"""Safe Jev-Omni teacher input preparation and CPU preflight helpers.

This module intentionally does not import or execute the upstream
``jev_omni.py`` file. It preserves the upstream prompt, decoder-capture/head
contract metadata, processor call shape, and forward probability semantics.
The real 24GB backbone loader is gated on CUDA and local safetensors only;
CPU tests use tiny fake models to verify the project-owned forward boundary.
"""

from __future__ import annotations

import gc
import inspect
import json
import tempfile
import wave
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from scripts.assets import selected_files, verify_file
from train.checkpoint import file_sha256
from train.teacher_head import (
    ALLOWED_SOURCE_REPO,
    PINNED_SOURCE_REVISION,
    load_expected_spec,
    tensor_bytes_sha256,
    validate_source,
)
from train.teacher_media import SAMPLE_RATE, prepared_teacher_audio

SCHEMA_VERSION = 1
OFFICIAL_JEV_OMNI_SHA256 = "11d761b0b6cefc8aac29757f43af4b2b02af8f9b6c6dad19834c0b14538180f0"
DEFAULT_SOURCE_DIR = Path("artifacts/quarantine/teacher/5addda86ddee081a68fb067477ea100c221b8917")
DEFAULT_MODEL_DIR = Path(
    "artifacts/models/akhilaaa3__Jev-Omni/5addda86ddee081a68fb067477ea100c221b8917"
)
DEFAULT_MODEL_ASSET_MANIFEST = Path("configs/project-model-assets.json")
DEFAULT_SAFE_HEAD = Path(
    "artifacts/teacher/jev-omni/5addda86ddee081a68fb067477ea100c221b8917/head.safetensors"
)
DEFAULT_HEAD_SPEC = Path("configs/teacher-head-jev-omni-spec.json")
DEFAULT_HEAD_CONVERSION_REPORT = Path("reports/teacher-head-conversion.json")
REQUIRED_SOURCE_FILES = (
    "config.json",
    "decision_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "jev_omni.py",
)


@dataclass(frozen=True)
class PreparedTeacherInput:
    modality: str
    option_count: int
    messages: list[dict[str, Any]]
    media_path: Path | None


@dataclass(frozen=True)
class ProcessorBuildResult:
    inputs: Mapping[str, Any]
    summary: dict[str, Any]


@dataclass(frozen=True)
class TeacherForwardResult:
    prediction: str
    prediction_index: int
    confidence: float
    probabilities: dict[str, float]


class TeacherForwardBlocked(RuntimeError):
    """Raised when the real teacher loader is intentionally blocked."""

    def __init__(self, message: str, report: Mapping[str, Any]):
        super().__init__(message)
        self.report = dict(report)


def official_teacher_prompt(state: str, question: str, options: Sequence[str]) -> str:
    """Return the exact upstream Jev-Omni decision prompt text."""
    checked = validate_teacher_options(options)
    choices = "\n".join(f"{index + 1}. {value}" for index, value in enumerate(checked))
    return (
        f"{state}\n\n---\n\nQUESTION: {question}\n\nOPTIONS:\n{choices}\n\n"
        f"Reply with only the number of the correct option (1-{len(checked)}).\n"
        "Output a single number and nothing else."
    )


def validate_teacher_options(options: Sequence[str]) -> tuple[str, ...]:
    if isinstance(options, str) or not 2 <= len(options) <= 256:
        raise ValueError("Jev-Omni teacher requires 2-256 options")
    checked = tuple(options)
    if any(not isinstance(item, str) or not item for item in checked):
        raise ValueError("teacher options must be nonempty strings")
    if len(set(checked)) != len(checked):
        raise ValueError("teacher options must be distinct")
    return checked


def _load_image(path: Path) -> Any:
    image_module = __import__("PIL.Image", fromlist=["Image"])
    image = image_module.open(path).convert("RGB")
    image.load()
    return image


def _video_frames(path: Path, *, count: int = 16) -> list[Any]:
    if not 1 <= count <= 256:
        raise ValueError("teacher video frame count must be in [1, 256]")
    cv2 = __import__("cv2")
    image_module = __import__("PIL.Image", fromlist=["Image"])
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"Could not decode video frame count: {path}")
        wanted = sorted({int(round((total - 1) * (index + 0.5) / count)) for index in range(count)})
        if len(wanted) != count:
            raise ValueError(
                f"pinned sampling protocol yields {len(wanted)} unique frames, requested {count}"
            )
        frames = []
        current = 0
        for index in wanted:
            while current < index and cap.grab():
                current += 1
            ok, frame = cap.read()
            current += 1
            if not ok:
                break
            frames.append(image_module.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    if len(frames) != count:
        raise ValueError(f"Could not decode the requested {count} teacher video frames: {path}")
    return frames


def _read_prepared_pcm16_wav(path: Path) -> Any:
    """Return the exact prepared mono16k PCM as float32 waveform samples."""
    import numpy as np

    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getframerate() != 16000
            or stream.getsampwidth() != 2
            or stream.getcomptype() != "NONE"
        ):
            raise ValueError("prepared teacher audio must be mono 16kHz PCM16 WAV")
        frames = stream.readframes(stream.getnframes())
        if len(frames) != stream.getnframes() * 2:
            raise ValueError("prepared teacher audio has incomplete PCM frames")
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / np.float32(32768.0)


@contextmanager
def prepared_teacher_input(
    *,
    state: str,
    question: str,
    options: Sequence[str],
    modality: str = "text",
    media: Path | None = None,
    video_frames: int = 16,
    temp_parent: Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> Iterator[PreparedTeacherInput]:
    """Prepare upstream-shaped content without calling the upstream ``predict`` helper."""
    if modality not in {"text", "image", "audio", "video"}:
        raise ValueError("teacher modality must be text, image, audio, or video")
    checked = validate_teacher_options(options)
    if modality == "text" and media is not None:
        raise ValueError("text teacher input must not include media")
    if modality != "text" and media is None:
        raise ValueError(f"{modality} teacher input requires media")
    media_path = media.resolve(strict=True) if media is not None else None
    content: list[dict[str, Any]] = []
    if modality == "image":
        assert media_path is not None
        content.append({"type": "image", "image": _load_image(media_path)})
        content.append({"type": "text", "text": official_teacher_prompt(state, question, checked)})
        yield PreparedTeacherInput(
            modality, len(checked), [{"role": "user", "content": content}], media_path
        )
        return
    if modality == "video":
        assert media_path is not None
        content.extend(
            {"type": "image", "image": frame}
            for frame in _video_frames(media_path, count=video_frames)
        )
        content.append({"type": "text", "text": official_teacher_prompt(state, question, checked)})
        yield PreparedTeacherInput(
            modality, len(checked), [{"role": "user", "content": content}], media_path
        )
        return
    if modality == "audio":
        assert media_path is not None
        with prepared_teacher_audio(
            media_path, temp_parent=temp_parent, ffmpeg_binary=ffmpeg_binary
        ) as audio:
            content.append({"type": "audio", "audio": _read_prepared_pcm16_wav(audio)})
            content.append(
                {"type": "text", "text": official_teacher_prompt(state, question, checked)}
            )
            yield PreparedTeacherInput(
                modality, len(checked), [{"role": "user", "content": content}], media_path
            )
        return
    content.append({"type": "text", "text": official_teacher_prompt(state, question, checked)})
    yield PreparedTeacherInput(modality, len(checked), [{"role": "user", "content": content}], None)


def _value_summary(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {"kind": "tensor", "shape": [int(dim) for dim in shape], "dtype": str(dtype)}
    if isinstance(value, list):
        return {"kind": "list", "length": len(value)}
    if isinstance(value, tuple):
        return {"kind": "tuple", "length": len(value)}
    return {"kind": type(value).__name__}


def build_teacher_processor_inputs(
    processor: Any, prepared: PreparedTeacherInput
) -> ProcessorBuildResult:
    """Call the official processor path while keeping reports payload-free."""
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if apply_chat_template is None:
        raise ValueError("teacher processor must expose apply_chat_template")
    raw = apply_chat_template(
        prepared.messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("teacher processor must return a nonempty mapping")
    summary = {
        "modality": prepared.modality,
        "option_count": prepared.option_count,
        "keys": {
            str(key): _value_summary(value)
            for key, value in sorted(raw.items(), key=lambda item: str(item[0]))
        },
        "thinking_disabled": True,
        "tokenize": True,
        "contains_payload_values": False,
    }
    return ProcessorBuildResult(inputs=raw, summary=summary)


def load_teacher_processor(source_dir: Path) -> Any:
    """Load only local processor/tokenizer files; never execute remote custom code."""
    source_dir = source_dir.resolve(strict=True)
    if not source_dir.is_dir():
        raise ValueError("teacher processor source must be a local directory")
    transformers = __import__("transformers", fromlist=["AutoProcessor"])
    return transformers.AutoProcessor.from_pretrained(
        str(source_dir),
        local_files_only=True,
        trust_remote_code=False,
    )


try:
    _TORCH_NN_MODULE = __import__("torch").nn.Module
except Exception:  # pragma: no cover - lets non-training envs import this module
    _TORCH_NN_MODULE = object


def _torch() -> Any:
    return __import__("torch")


def _safetensors_torch() -> Any:
    return __import__("safetensors.torch", fromlist=["load_file"])


def _teacher_count_tensor(option_count: int, *, device: Any) -> Any:
    return _torch().tensor([option_count], device=device)


class TeacherDecisionHead(_TORCH_NN_MODULE):  # type: ignore[misc, valid-type]
    """Converted Jev-Omni _Head256 with safetensors-only loading."""

    def __init__(self, hidden_size: int, output_classes: int = 256) -> None:
        super().__init__()
        torch = _torch()
        self.register_buffer("mu", torch.zeros(1, hidden_size, dtype=torch.float32))
        self.register_buffer("sd", torch.ones(1, hidden_size, dtype=torch.float32))
        self.linear = torch.nn.Linear(hidden_size, output_classes, dtype=torch.float32)

    def forward(self, features: Any, counts: Any) -> Any:
        torch = _torch()
        z = self.linear((features.float() - self.mu) / self.sd)
        mask = torch.arange(z.shape[-1], device=z.device)[None] >= counts[:, None]
        return z.masked_fill(mask, -1e30)


def load_safe_teacher_head_module(
    head_path: Path,
    spec_path: Path,
    *,
    expected_sha256: str,
    device: str = "cpu",
) -> TeacherDecisionHead:
    """Load the converted decision head without torch.load or upstream code."""
    report = inspect_safe_teacher_head(head_path, spec_path, expected_sha256=expected_sha256)
    tensors = _safetensors_torch().load_file(str(head_path.resolve(strict=True)), device=device)
    hidden_shape = report["tensors"].get("mu", {}).get("shape")
    if not isinstance(hidden_shape, list) or len(hidden_shape) != 2 or hidden_shape[0] != 1:
        raise ValueError("teacher head mu tensor must have shape [1, hidden]")
    linear_bias_shape = report["tensors"].get("linear.bias", {}).get("shape")
    if (
        not isinstance(linear_bias_shape, list)
        or len(linear_bias_shape) != 1
        or int(linear_bias_shape[0]) < 2
    ):
        raise ValueError("teacher head linear.bias tensor must define at least two outputs")
    head = TeacherDecisionHead(int(hidden_shape[1]), int(linear_bias_shape[0])).to(device).eval()
    head.load_state_dict(tensors)
    return cast(TeacherDecisionHead, head)


def find_teacher_decoder(model: Any) -> tuple[str, Any]:
    """Locate the official Gemma4 text backbone module used for hidden capture."""
    for path in ("model.language_model", "language_model.model", "model.text_model", "model"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and hasattr(node, "layers") and hasattr(node, "register_forward_hook"):
            return path, node
    raise RuntimeError("Could not locate the Gemma text backbone")


def _move_inputs_to_device(inputs: Mapping[str, Any], *, device: str) -> dict[str, Any]:
    torch = _torch()
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        to = getattr(value, "to", None)
        if to is None:
            moved[key] = value
        elif bool(torch.is_tensor(value)) and torch.is_floating_point(value) and device == "cuda":
            moved[key] = value.to(device, dtype=torch.bfloat16)
        else:
            moved[key] = value.to(device)
    return moved


class JevOmniTeacherForward:
    """Project-owned Jev-Omni forward adapter with explicit hook cleanup."""

    def __init__(
        self,
        *,
        model: Any,
        head: TeacherDecisionHead,
        processor: Any,
        decoder: Any,
        device: str,
    ) -> None:
        self.model = model
        self.head = head
        self.processor = processor
        self.device = device
        self._capture: dict[str, Any] = {}
        self._hook = decoder.register_forward_hook(self._capture_decoder_hidden)
        self._extra = (
            {"logits_to_keep": 1}
            if "logits_to_keep" in inspect.signature(model.forward).parameters
            else {}
        )

    def _capture_decoder_hidden(self, _module: Any, _args: Any, output: Any) -> None:
        hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        self._capture["hidden"] = hidden[:, -1].float()

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._capture.clear()

    def __enter__(self) -> JevOmniTeacherForward:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def predict_prepared(
        self,
        prepared: PreparedTeacherInput,
        inputs: Mapping[str, Any],
        options: Sequence[str],
    ) -> TeacherForwardResult:
        checked = validate_teacher_options(options)
        if len(checked) != prepared.option_count:
            raise ValueError("teacher options changed after input preparation")
        torch = _torch()
        self._capture.clear()
        try:
            with torch.inference_mode():
                moved = _move_inputs_to_device(inputs, device=self.device)
                autocast = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.device == "cuda"
                    else nullcontext()
                )
                with autocast:
                    self.model(**moved, use_cache=False, **self._extra)
                    hidden = self._capture.get("hidden")
                    if hidden is None:
                        raise RuntimeError("teacher decoder hidden capture was empty")
                    logits = self.head(
                        hidden, _teacher_count_tensor(len(checked), device=hidden.device)
                    )
                    probabilities = logits[0, : len(checked)].softmax(-1)
            values = [float(item) for item in probabilities.float().cpu().tolist()]
            best = max(range(len(values)), key=values.__getitem__)
            return TeacherForwardResult(
                prediction=checked[best],
                prediction_index=best,
                confidence=values[best],
                probabilities=dict(zip(checked, values, strict=True)),
            )
        finally:
            self._capture.clear()

    def predict(
        self,
        *,
        state: str,
        question: str,
        options: Sequence[str],
        media: Path | None = None,
        modality: str = "text",
        video_frames: int = 16,
        temp_parent: Path | None = None,
        ffmpeg_binary: str = "ffmpeg",
    ) -> TeacherForwardResult:
        with prepared_teacher_input(
            state=state,
            question=question,
            options=options,
            modality=modality,
            media=media,
            video_frames=video_frames,
            temp_parent=temp_parent,
            ffmpeg_binary=ffmpeg_binary,
        ) as prepared:
            inputs = build_teacher_processor_inputs(self.processor, prepared).inputs
            return self.predict_prepared(prepared, inputs, options)


def _teacher_manifest_model(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.resolve(strict=True).read_text())
    models = manifest.get("models", []) if isinstance(manifest, dict) else []
    matching = [
        model
        for model in models
        if model.get("repo_id") == ALLOWED_SOURCE_REPO
        and model.get("revision") == PINNED_SOURCE_REVISION
    ]
    if len(matching) != 1:
        raise ValueError("teacher model manifest must contain exactly the pinned Jev-Omni entry")
    selected = matching[0]
    if not isinstance(selected, dict):
        raise ValueError("teacher model manifest entry must be an object")
    return selected


def _required_teacher_files(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    files = model.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("teacher model manifest entry has no files")
    required = [file for file in files if isinstance(file, dict) and file.get("required") is True]
    if not required:
        raise ValueError("teacher model manifest entry has no required files")
    paths = [str(file.get("path")) for file in required]
    if len(paths) != len(set(paths)):
        raise ValueError("teacher model manifest has duplicate required paths")
    return required


def _checksum_fields(file: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": file["path"],
        "size": file.get("size", file.get("bytes")),
        "kind": file.get("kind"),
        "upstream_sha256": file.get("upstream_sha256") or file.get("lfs_sha256"),
        "actual_sha256": file.get("actual_sha256"),
        "git_blob_sha1": file.get("git_blob_sha1"),
    }


def _selected_files_manifest(
    model: Mapping[str, Any], required: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for file in required:
        path = str(file["path"])
        kind = str(file.get("kind"))
        if path == "head.pt":
            continue
        if path.endswith(".safetensors"):
            if kind != "weights":
                raise ValueError("teacher safetensors file is not marked as weights")
            selected.append({**_checksum_fields(file), "kind": "weights"})
        elif path in {
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "processor_config.json",
            "model.safetensors.index.json",
        }:
            selected.append({**_checksum_fields(file), "kind": "metadata"})
    return {
        "schema_version": SCHEMA_VERSION,
        "models": [
            {
                "repo_id": ALLOWED_SOURCE_REPO,
                "revision": PINNED_SOURCE_REVISION,
                "license": model.get("license"),
                "files": selected,
            }
        ],
    }


def _verify_manifest_file(model_dir: Path, file: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    path = str(file["path"])
    if not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("unsafe teacher model manifest path")
    normalized = _checksum_fields(file)
    if not isinstance(normalized.get("size"), int):
        raise ValueError(f"teacher model manifest file lacks size: {path}")
    target = model_dir / path
    if not target.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("teacher model file escapes pinned directory")
    return path, {
        "bytes": normalized["size"],
        "sha256": verify_file(target, normalized),
        "kind": normalized["kind"],
    }


def validate_teacher_model_assets(
    model_dir: Path = DEFAULT_MODEL_DIR,
    manifest_path: Path = DEFAULT_MODEL_ASSET_MANIFEST,
) -> dict[str, Any]:
    """Verify the full local Jev-Omni model snapshot before any CUDA load."""
    model = _teacher_manifest_model(manifest_path)
    validate_source(str(model.get("repo_id")), str(model.get("revision")))
    model_dir = model_dir.resolve(strict=True)
    expected_parent = ALLOWED_SOURCE_REPO.replace("/", "__")
    if model_dir.name != PINNED_SOURCE_REVISION or model_dir.parent.name != expected_parent:
        raise ValueError("teacher model_dir must be the pinned repo/revision directory")
    required = _required_teacher_files(model)
    selected = selected_files(_selected_files_manifest(model, required), metadata_only=False)
    selected_paths = {file["path"] for _, file in selected}
    hashes: dict[str, dict[str, Any]] = {}
    for _, file in selected:
        path, info = _verify_manifest_file(model_dir, file)
        hashes[path] = info
    raw_head = [file for file in required if file.get("path") == "head.pt"]
    if len(raw_head) != 1:
        raise ValueError("teacher manifest must retain exactly one raw head.pt provenance entry")
    if raw_head[0].get("kind") == "weights":
        raise ValueError("raw head.pt must not be admitted as model weights")
    path, info = _verify_manifest_file(model_dir, raw_head[0])
    hashes[path] = {**info, "runtime_loaded": False}
    for file in required:
        path = str(file["path"])
        if path in selected_paths or path == "head.pt":
            continue
        path, info = _verify_manifest_file(model_dir, file)
        hashes[path] = info
    expected_weights = {file["path"] for _, file in selected if file["kind"] == "weights"}
    actual_weights = {path.name for path in model_dir.glob("model*.safetensors")}
    if actual_weights != expected_weights:
        raise ValueError("unverified or missing teacher safetensors files in model directory")
    index = model_dir / "model.safetensors.index.json"
    index_in_manifest = "model.safetensors.index.json" in hashes
    if index.is_file():
        if not index_in_manifest:
            raise ValueError("teacher safetensors index exists but is not manifest verified")
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(not isinstance(value, str) for value in weight_map.values())
            or set(weight_map.values()) != expected_weights
        ):
            raise ValueError("teacher safetensors index references unverified shards")
    elif expected_weights != {"model.safetensors"}:
        raise ValueError("teacher sharded safetensors require a verified index")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("architectures") != ["Gemma4UnifiedForConditionalGeneration"]:
        raise ValueError("teacher model architecture mismatch")
    if config.get("model_type") != "gemma4_unified":
        raise ValueError("teacher model type mismatch")
    return {
        "model_dir": str(model_dir),
        "manifest": str(manifest_path.resolve(strict=True)),
        "source_repo": ALLOWED_SOURCE_REPO,
        "source_revision": PINNED_SOURCE_REVISION,
        "license": model.get("license"),
        "files_verified": len(hashes),
        "bytes_verified": sum(int(item["bytes"]) for item in hashes.values()),
        "weights": sorted(expected_weights),
        "raw_head_verified_not_loaded": True,
        "index_verified": index.is_file(),
        "files": hashes,
    }


def teacher_forward_readiness_report(
    *,
    protocol_source_dir: Path = DEFAULT_SOURCE_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    model_asset_manifest: Path = DEFAULT_MODEL_ASSET_MANIFEST,
    head_path: Path = DEFAULT_SAFE_HEAD,
    spec_path: Path = DEFAULT_HEAD_SPEC,
    conversion_report: Path = DEFAULT_HEAD_CONVERSION_REPORT,
    expected_head_sha256: str | None = None,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_utc": started.isoformat(),
        "model_forward_run": False,
        "model_backbone_loaded": False,
        "student_head_used": False,
        "trust_remote_code": False,
        "local_files_only": True,
        "use_safetensors_for_model": True,
        "use_safetensors_for_head": True,
        "upstream_custom_code_executed": False,
        "official_model_class": "Gemma4UnifiedForConditionalGeneration",
        "loader_stage": "pre_from_pretrained",
    }
    try:
        expected_head_sha256 = expected_head_sha256 or expected_head_sha256_from_report(
            conversion_report
        )
        report["protocol_source"] = validate_teacher_source_tree(
            protocol_source_dir, expected_jev_omni_sha256=OFFICIAL_JEV_OMNI_SHA256
        )
        report["model_assets"] = validate_teacher_model_assets(model_dir, model_asset_manifest)
        report["head"] = inspect_safe_teacher_head(
            head_path, spec_path, expected_sha256=expected_head_sha256
        )
        torch = _torch()
        report["cuda_available"] = bool(torch.cuda.is_available())
        if not report["cuda_available"]:
            raise TeacherForwardBlocked(
                "teacher forward loader requires CUDA before from_pretrained", report
            )
        report["status"] = "ready_for_explicit_cuda_load"
        report["exit_code"] = 0
    except TeacherForwardBlocked as exc:
        report.update(exc.report)
        report["status"] = "blocked"
        report["exit_code"] = 1
        report["blocker"] = type(exc).__name__
        report["error"] = str(exc)
    except Exception as exc:
        report["status"] = "blocked"
        report["exit_code"] = 1
        report["blocker"] = type(exc).__name__
        report["error"] = str(exc)
    finished = datetime.now(UTC)
    report["finished_utc"] = finished.isoformat()
    report["elapsed_seconds"] = round((finished - started).total_seconds(), 3)
    return report


def load_teacher_forward_bundle(
    *,
    protocol_source_dir: Path = DEFAULT_SOURCE_DIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    model_asset_manifest: Path = DEFAULT_MODEL_ASSET_MANIFEST,
    head_path: Path = DEFAULT_SAFE_HEAD,
    spec_path: Path = DEFAULT_HEAD_SPEC,
    conversion_report: Path = DEFAULT_HEAD_CONVERSION_REPORT,
    expected_head_sha256: str | None = None,
    device: str = "cuda",
) -> JevOmniTeacherForward:
    """Load the real teacher only from local pinned safetensors on CUDA."""
    if device != "cuda":
        raise ValueError("teacher forward loader only admits the CUDA device")
    readiness = teacher_forward_readiness_report(
        protocol_source_dir=protocol_source_dir,
        model_dir=model_dir,
        model_asset_manifest=model_asset_manifest,
        head_path=head_path,
        spec_path=spec_path,
        conversion_report=conversion_report,
        expected_head_sha256=expected_head_sha256,
    )
    if readiness.get("status") != "ready_for_explicit_cuda_load":
        raise TeacherForwardBlocked(
            str(readiness.get("error", "teacher forward blocked")), readiness
        )
    transformers = __import__("transformers", fromlist=["Gemma4UnifiedForConditionalGeneration"])
    model_class = transformers.Gemma4UnifiedForConditionalGeneration
    torch = _torch()
    model = model_class.from_pretrained(
        str(model_dir.resolve(strict=True)),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    expected_head_sha256 = expected_head_sha256 or expected_head_sha256_from_report(
        conversion_report
    )
    head = load_safe_teacher_head_module(
        head_path, spec_path, expected_sha256=expected_head_sha256, device=device
    )
    processor = load_teacher_processor(model_dir)
    _decoder_path, decoder = find_teacher_decoder(model)
    return JevOmniTeacherForward(
        model=model, head=head, processor=processor, decoder=decoder, device=device
    )


def validate_teacher_source_tree(
    source_dir: Path,
    *,
    expected_jev_omni_sha256: str = OFFICIAL_JEV_OMNI_SHA256,
    source_repo: str = ALLOWED_SOURCE_REPO,
    source_revision: str = PINNED_SOURCE_REVISION,
) -> dict[str, Any]:
    validate_source(source_repo, source_revision)
    source_dir = source_dir.resolve(strict=True)
    if not source_dir.is_dir():
        raise ValueError("teacher source must be a local directory")
    files: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_SOURCE_FILES:
        path = source_dir / name
        if not path.is_file():
            raise ValueError(f"missing teacher source file: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if files["jev_omni.py"]["sha256"] != expected_jev_omni_sha256:
        raise ValueError("teacher upstream jev_omni.py SHA256 mismatch")
    decision = json.loads((source_dir / "decision_config.json").read_text())
    processor = json.loads((source_dir / "processor_config.json").read_text())
    config = json.loads((source_dir / "config.json").read_text())
    if decision.get("hidden_size") != 3840 or decision.get("output_classes") != 256:
        raise ValueError("teacher decision head config does not match the pinned contract")
    if decision.get("dtype") != "float32":
        raise ValueError("teacher decision head dtype must remain float32")
    if processor.get("processor_class") != "Gemma4UnifiedProcessor":
        raise ValueError("teacher processor class mismatch")
    if config.get("architectures") != ["Gemma4UnifiedForConditionalGeneration"]:
        raise ValueError("teacher model architecture mismatch")
    return {
        "source_repo": source_repo,
        "source_revision": source_revision,
        "files": files,
        "decision_config": {
            "hidden_size": decision["hidden_size"],
            "output_classes": decision["output_classes"],
            "dtype": decision["dtype"],
        },
        "processor_config": {
            "processor_class": processor["processor_class"],
            "audio_sampling_rate": processor["feature_extractor"]["sampling_rate"],
            "video_num_frames_config": processor["video_processor"]["num_frames"],
        },
        "config": {"architectures": config["architectures"], "model_type": config["model_type"]},
        "upstream_custom_code_executed": False,
    }


def inspect_safe_teacher_head(
    head_path: Path,
    spec_path: Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    if head_path.suffix != ".safetensors":
        raise ValueError("teacher preflight requires the converted safetensors head")
    head_path = head_path.resolve(strict=True)
    spec = load_expected_spec(spec_path.resolve(strict=True))
    if spec is None:
        raise ValueError("teacher preflight requires an expected head spec")
    actual_sha256 = file_sha256(head_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("teacher safetensors head SHA256 mismatch")
    safetensors = __import__("safetensors.torch", fromlist=["load_file"])
    torch = __import__("torch")
    tensors = safetensors.load_file(str(head_path), device="cpu")
    if set(tensors) != set(spec):
        raise ValueError("teacher safetensors head keys mismatch")
    summary: dict[str, dict[str, Any]] = {}
    for key in sorted(tensors):
        tensor = tensors[key]
        expected = spec[key]
        shape = [int(dim) for dim in tensor.shape]
        dtype = str(tensor.dtype)
        if shape != expected["shape"] or dtype != expected["dtype"]:
            raise ValueError(f"teacher safetensors head tensor mismatch: {key}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"teacher safetensors head tensor has nonfinite values: {key}")
        summary[key] = {
            "shape": shape,
            "dtype": dtype,
            "sha256": tensor_bytes_sha256(tensor),
        }
    return {
        "path": str(head_path),
        "bytes": head_path.stat().st_size,
        "sha256": actual_sha256,
        "tensor_count": len(summary),
        "tensors": summary,
        "loaded_with": "safetensors.torch.load_file",
        "torch_load_used": False,
    }


def expected_head_sha256_from_report(path: Path = DEFAULT_HEAD_CONVERSION_REPORT) -> str:
    report = json.loads(path.resolve(strict=True).read_text())
    sha256 = report.get("output_sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("teacher head conversion report does not contain output_sha256")
    if report.get("status") != "converted" or report.get("exit_code") != 0:
        raise ValueError("teacher head conversion report is not a passing conversion")
    return sha256


def _write_synthetic_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x00" * int(SAMPLE_RATE / 10))


def synthetic_processor_preflight(
    processor: Any,
    *,
    modality: str = "audio",
    temp_parent: Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="teacher-preflight-", dir=temp_parent) as directory:
        root = Path(directory)
        media: Path | None = None
        if modality == "audio":
            media = root / "synthetic.wav"
            _write_synthetic_wav(media)
        elif modality == "image":
            image_module = __import__("PIL.Image", fromlist=["Image"])
            media = root / "synthetic.png"
            image_module.new("RGB", (16, 16), (0, 0, 0)).save(media)
        elif modality != "text":
            raise ValueError("synthetic preflight supports text, image, or audio")
        with prepared_teacher_input(
            state="Synthetic teacher CPU preflight state.",
            question="Which option is valid?",
            options=("alpha", "beta"),
            modality=modality,
            media=media,
            temp_parent=root,
            ffmpeg_binary=ffmpeg_binary,
        ) as prepared:
            return build_teacher_processor_inputs(processor, prepared).summary


def run_teacher_preflight(
    *,
    source_dir: Path = DEFAULT_SOURCE_DIR,
    head_path: Path = DEFAULT_SAFE_HEAD,
    spec_path: Path = DEFAULT_HEAD_SPEC,
    conversion_report: Path = DEFAULT_HEAD_CONVERSION_REPORT,
    synthetic_modality: str = "audio",
    expected_head_sha256: str | None = None,
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, Any]:
    started = datetime.now(UTC)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_utc": started.isoformat(),
        "gpu_used": False,
        "model_forward_run": False,
        "model_backbone_loaded": False,
        "student_head_used": False,
        "trust_remote_code": False,
        "local_files_only": True,
        "use_safetensors_for_head": True,
        "source_repo": ALLOWED_SOURCE_REPO,
        "source_revision": PINNED_SOURCE_REVISION,
        "source_scope": (
            "source-only structural check; only jev_omni.py and the converted head are "
            "hash anchored here, so this report is not full model-source admission"
        ),
    }
    try:
        expected_head_sha256 = expected_head_sha256 or expected_head_sha256_from_report(
            conversion_report
        )
        report["source"] = validate_teacher_source_tree(source_dir)
        report["head"] = inspect_safe_teacher_head(
            head_path, spec_path, expected_sha256=expected_head_sha256
        )
        gc.collect()
        processor = load_teacher_processor(source_dir)
        report["processor"] = {
            "class": type(processor).__name__,
            "synthetic_input": synthetic_processor_preflight(
                processor,
                modality=synthetic_modality,
                ffmpeg_binary=ffmpeg_binary,
            ),
        }
        report["forward_integration_status"] = (
            "project_owned_forward_entry_ready; real 24GB load remains CUDA gated"
        )
        report["status"] = "passed"
        report["exit_code"] = 0
    except Exception as exc:
        report["status"] = "blocked"
        report["exit_code"] = 1
        report["blocker"] = type(exc).__name__
        report["error"] = str(exc)
    finished = datetime.now(UTC)
    report["finished_utc"] = finished.isoformat()
    report["elapsed_seconds"] = round((finished - started).total_seconds(), 3)
    return report
