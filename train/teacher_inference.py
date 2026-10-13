"""Safe Jev-Omni teacher input preparation and CPU preflight helpers.

This module intentionally does not import or execute the upstream
``jev_omni.py`` file. It preserves the upstream prompt, decoder-capture/head
contract metadata, and processor call shape. Full teacher forward integration
is still unwritten in this slice and requires separate implementation plus GPU
evidence.
"""

from __future__ import annotations

import gc
import json
import tempfile
import wave
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
DEFAULT_SOURCE_DIR = Path(
    "artifacts/quarantine/teacher/5addda86ddee081a68fb067477ea100c221b8917"
)
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
    import numpy as np  # type: ignore[import-not-found]

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
        report["forward_integration_status"] = "missing_unwritten_in_this_slice"
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
