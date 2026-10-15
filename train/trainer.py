"""Bounded text/image/audio/video decision LoRA smoke; no quality or GPU-parity claims."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, overload

from pydantic import Field, model_validator

from data.schema import (
    Contract,
    HardTarget,
    InsufficientTarget,
    Prefix,
    Sample,
    SoftTarget,
    answer_tokens,
    validate_pool_isolation,
)
from data.synth import generate_samples
from scripts.assets import selected_files, verify_file, write_json
from scripts.check_tokens import OfficialTokenizerAdapter
from train.checkpoint import (
    file_sha256,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    verify_resume_request,
)
from train.monitoring import SwanLabConfig, Telemetry, start_swanlab
from train.multimodal import (
    ImageMaxSoftTokens,
    configure_image_processor,
    load_media,
    resolve_media,
)
from train.template import check_answer_tokens, render_payload, render_prompt


class SmokeConfig(Contract):
    seed: int = Field(default=42, ge=0)
    sample_count: Literal[100] = 100
    steps: Literal[10] = 10
    accumulation: Literal[10] = 10
    max_length: int = Field(default=2048, ge=32, le=4096)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.001)
    rank: Literal[16] = 16
    alpha: Literal[32] = 32
    dtype: Literal["bfloat16"] = "bfloat16"
    modality: Literal["text"] = "text"

    @model_validator(mode="after")
    def finite_values(self) -> SmokeConfig:
        if not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite")
        return self


class LoopConfig(Contract):
    """S1-scale training over a frozen pool; P0 smoke keeps its fixed Literals."""

    seed: int = Field(default=42, ge=0)
    sample_count: int = Field(ge=2, le=100_000)
    steps: int = Field(ge=1, le=100_000)
    accumulation: int = Field(ge=1, le=64)
    max_length: int = Field(default=2048, ge=32, le=4096)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.001)
    rank: Literal[16] = 16
    alpha: Literal[32] = 32
    dtype: Literal["bfloat16"] = "bfloat16"
    modality: Literal["text", "image", "screenshot", "audio", "video", "mixed"] = "text"
    video_num_frames: Literal[8, 16] = 16
    image_max_soft_tokens: ImageMaxSoftTokens | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    media_root: str | None = None
    media_roots: list[str] | None = None
    midway_step: int | None = Field(default=None, ge=1)
    expected_data_version: str | None = None
    expected_data_versions: list[str] | None = Field(default=None, min_length=1)
    checkpoint_every: int | None = Field(default=None, ge=1)
    brier_weight: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    preparation_mode: Literal["eager", "lazy"] = "eager"
    preparation_cache_bytes: int = Field(default=256 * 1024 * 1024, ge=0)

    @model_validator(mode="after")
    def validate_loop(self) -> LoopConfig:
        if not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite")
        if self.midway_step is not None and self.midway_step > self.steps:
            raise ValueError("midway_step must not exceed steps")
        if self.modality == "text":
            if self.preparation_mode == "lazy":
                raise ValueError("lazy preparation is reserved for media training")
            if self.media_roots is not None:
                raise ValueError("media_roots is reserved for media training")
        elif self.media_roots is not None:
            if self.media_root:
                raise ValueError("media_root and media_roots are mutually exclusive")
            if not self.media_roots or any(not root.strip() for root in self.media_roots):
                raise ValueError("media_roots must be nonempty strings when provided")
        elif not (self.media_root or "").strip():
            raise ValueError(f"{self.modality} training requires a nonempty media_root")
        if self.modality == "mixed":
            if self.expected_data_version is None and self.expected_data_versions is None:
                raise ValueError("mixed training requires expected_data_versions")
        elif self.expected_data_versions is not None:
            raise ValueError("expected_data_versions is reserved for mixed training")
        return self


TrainConfig = SmokeConfig | LoopConfig

MEDIA_TYPES = ("image", "audio", "video")
MEDIA_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "image": ("pixel_values",),
    "audio": ("input_features", "input_features_mask"),
    "video": ("pixel_values_videos", "video_position_ids"),
}
# Everything a prepared media example may carry into the forward pass.
MEDIA_FORWARD_KEYS = (
    "pixel_values",
    "image_position_ids",
    "input_features",
    "input_features_mask",
    "pixel_values_videos",
    "video_position_ids",
    "num_frames_per_video",
)


def cohort_media_type(modality: str) -> str:
    """Screenshot cohorts reuse image media; audio and video carry their own type."""
    return "image" if modality in {"image", "screenshot"} else modality


def smoke_samples(
    config: TrainConfig,
    data_path: Path | None = None,
    *,
    expected_data_version: str | None = None,
    expected_data_versions: list[str] | None = None,
) -> list[Sample]:
    if data_path is None and config.modality != "text":
        raise ValueError("non-text cohorts require a frozen data file")
    if data_path is None:
        samples = generate_samples(config.sample_count, config.seed)
        # Keep exactly one question per source sample, balanced across four target tasks.
        for index, sample in enumerate(samples):
            sample.questions = [sample.questions[index % len(sample.questions)]]
    else:
        samples = [
            Sample.model_validate_json(line)
            for line in data_path.read_text().splitlines()
            if line.strip()
        ]
    if len(samples) != config.sample_count or any(len(s.questions) != 1 for s in samples):
        raise ValueError(
            f"training requires exactly {config.sample_count} samples, one question each"
        )
    validate_pool_isolation(samples)
    mixed_versions: frozenset[str] | None = None
    if config.modality == "mixed":
        versions = list(expected_data_versions or [])
        if expected_data_version is not None:
            versions.append(expected_data_version)
        if not versions:
            raise ValueError("mixed training requires expected_data_versions")
        mixed_versions = frozenset(versions)
    for sample in samples:
        if config.modality == "mixed":
            if sample.meta.pool != "train":
                raise ValueError("smoke accepts training-pool samples")
            media = sample.prefix.media
            if media:
                if (
                    len(media) != 1
                    or media[0].type not in MEDIA_TYPES
                    or media[0].type not in sample.meta.modality
                ):
                    raise ValueError(
                        "mixed media samples require exactly one media item "
                        "matching the declared modality"
                    )
            elif sample.meta.modality != ["text"]:
                raise ValueError("mixed text samples must be text-only")
        elif config.modality != "text":
            if sample.meta.pool != "train":
                raise ValueError("smoke accepts training-pool samples")
            expected = cohort_media_type(config.modality)
            if (
                len(sample.prefix.media) != 1
                or sample.prefix.media[0].type != expected
                or expected not in sample.meta.modality
            ):
                raise ValueError(
                    f"{config.modality} training requires exactly one {expected} per sample"
                )
        elif sample.meta.pool != "train" or sample.meta.modality != ["text"] or sample.prefix.media:
            raise ValueError("smoke accepts text-only training-pool samples")
        if mixed_versions is not None:
            if sample.meta.data_version not in mixed_versions:
                raise ValueError("mixed training data_version is not in the frozen allocation")
        elif (
            expected_data_version is not None and sample.meta.data_version != expected_data_version
        ):
            raise ValueError("training data_version does not match the frozen allocation")
    return samples


def data_bytes(samples: list[Sample]) -> bytes:
    return ("\n".join(s.model_dump_json() for s in samples) + "\n").encode()


def prepare_examples(
    samples: list[Sample], tokenizer: Any, max_length: int
) -> list[dict[str, Any]]:
    examples = []
    for sample in samples:
        question = sample.questions[0]
        tokens = (*answer_tokens(question), "U")
        mapping = check_answer_tokens(tokenizer, question)
        prompt = render_prompt(tokenizer, sample.prefix, question)
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not input_ids or len(input_ids) > max_length:
            raise ValueError("empty or over-length prompt; truncation is forbidden")
        for token in tokens:
            if tokenizer.encode(prompt + token, add_special_tokens=False) != input_ids + [
                mapping[token]
            ]:
                raise ValueError(f"answer token changes at generation boundary: {token}")
        target = question.target
        if isinstance(target, HardTarget):
            expected = [float(t == str(target.value)) for t in tokens]
        elif isinstance(target, SoftTarget):
            expected = [*target.probs, 0.0]
        elif isinstance(target, InsufficientTarget):
            expected = [0.0] * (len(tokens) - 1) + [1.0]
        else:
            raise ValueError("sample lacks supervision")
        examples.append(
            {
                "id": sample.id,
                "input_ids": input_ids,
                "token_ids": [mapping[t] for t in tokens],
                "target": expected,
            }
        )
    return examples


def verified_media_file(
    media_root: Path, uri: str, sha256: str, *, extra_roots: list[Path] | None = None
) -> Path:
    """Streaming-sha-verified local media path; multi-root tries sha matches in order."""
    roots = [media_root, *(extra_roots or [])]
    failures: list[str] = []
    for root in roots:
        try:
            return resolve_media(root, uri, sha256, max_bytes=64 * 1024 * 1024)
        except (ValueError, OSError) as exc:
            failures.append(f"{root / uri}: {type(exc).__name__}: {exc}")
    raise ValueError(
        "media resolution failed for "
        + uri
        + " across "
        + str(len(roots))
        + " roots: "
        + "; ".join(failures)
    )


def media_processor_kwargs(media_type: str, loaded: Any) -> dict[str, Any]:
    """Official processor inputs per media type, isomorphic to s0_baseline_cpu."""
    if media_type == "image":
        return {"images": [[loaded]]}
    if media_type == "audio":
        return {"audio": [loaded], "sampling_rate": 16000}
    if media_type == "video":
        return {
            "videos": [loaded["frames"]],
            "video_metadata": [loaded["metadata"]],
            "do_sample_frames": False,
        }
    raise ValueError(f"unsupported media type: {media_type}")


def prepare_media_example(
    sample: Sample,
    processor: Any,
    media_root: Path,
    *extra_roots: Path,
    max_length: int,
    video_num_frames: int = 16,
) -> dict[str, Any]:
    """Official-processor rendering for one media sample, isomorphic to s0_baseline_cpu."""
    if video_num_frames not in (8, 16):
        raise ValueError("video_num_frames must be 8 or 16")
    question = sample.questions[0]
    if len(sample.prefix.media) != 1 or sample.prefix.media[0].type not in MEDIA_TYPES:
        raise ValueError(f"sample {sample.id} must carry exactly one image, audio or video item")
    item = sample.prefix.media[0]
    media_type = item.type
    try:
        media_path = verified_media_file(
            media_root, item.uri, item.sha256, extra_roots=list(extra_roots)
        )
    except ValueError as exc:
        raise ValueError(f"sample {sample.id}: {exc}") from exc
    loaded = load_media(
        media_path,
        media_type,
        num_frames=video_num_frames,
        max_seconds=30 if media_type == "audio" else 60,
    )
    payload = render_payload(Prefix(state=sample.prefix.state), question)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": media_type}, {"type": "text", "text": payload}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"official {media_type} template returned empty prompt")
    tokenizer = processor.tokenizer
    tokens = (*answer_tokens(question), "U")
    mapping = check_answer_tokens(tokenizer, question)
    inputs = processor(
        text=[prompt],
        return_tensors="pt",
        truncation=False,
        **media_processor_kwargs(media_type, loaded),
    )
    input_ids = inputs["input_ids"][0].tolist()
    missing = [key for key in MEDIA_REQUIRED_KEYS[media_type] if key not in inputs]
    if missing:
        raise ValueError(f"processor omitted {', '.join(missing)} for {media_type}")
    if not input_ids or len(input_ids) > max_length:
        raise ValueError("empty or over-length multimodal prompt; truncation is forbidden")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    for token in tokens:
        if tokenizer.encode(prompt + token, add_special_tokens=False) != prompt_ids + [
            mapping[token]
        ]:
            raise ValueError(f"answer token changes at generation boundary: {token}")
    target = question.target
    if isinstance(target, HardTarget):
        expected = [float(t == str(target.value)) for t in tokens]
    elif isinstance(target, SoftTarget):
        expected = [*target.probs, 0.0]
    elif isinstance(target, InsufficientTarget):
        expected = [0.0] * (len(tokens) - 1) + [1.0]
    else:
        raise ValueError("sample lacks supervision")
    example = {
        "id": sample.id,
        "input_ids": input_ids,
        "token_ids": [mapping[t] for t in tokens],
        "answer_token_ids": mapping,
        "target": expected,
    }
    for key in MEDIA_FORWARD_KEYS:
        if key in inputs:
            example[key] = inputs[key]
    if media_type == "video":
        example["video_num_frames"] = video_num_frames
    return example


def prepare_multimodal_examples(
    samples: list[Sample],
    model_dir: Path,
    media_root: Path,
    max_length: int,
    extra_roots: list[Path] | None = None,
    video_num_frames: int = 16,
    image_max_soft_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Cohort wrapper: load the official processor once, render every sample."""
    transformers = importlib.import_module("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    configure_image_processor(processor, image_max_soft_tokens)
    return [
        prepare_media_example(
            sample,
            processor,
            media_root,
            *(extra_roots or []),
            max_length=max_length,
            video_num_frames=video_num_frames,
        )
        for sample in samples
    ]


def prepare_mixed_example(
    sample: Sample,
    tokenizer: Any,
    processor: Any,
    media_root: Path,
    *,
    max_length: int,
    extra_roots: list[Path] | None = None,
    video_num_frames: int = 16,
) -> dict[str, Any]:
    """Route one sample by its own media; mixed text rows stay tokenizer-only."""
    if sample.prefix.media:
        return prepare_media_example(
            sample,
            processor,
            media_root,
            *(extra_roots or []),
            max_length=max_length,
            video_num_frames=video_num_frames,
        )
    return prepare_examples([sample], tokenizer, max_length)[0]


def prepare_mixed_examples(
    samples: list[Sample],
    model_dir: Path,
    media_root: Path,
    max_length: int,
    extra_roots: list[Path] | None = None,
    video_num_frames: int = 16,
    image_max_soft_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Mixed-cohort wrapper: one tokenizer plus one processor, per-sample routing."""
    transformers = importlib.import_module("transformers")
    processor = transformers.AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer = OfficialTokenizerAdapter(model_dir)
    configure_image_processor(processor, image_max_soft_tokens)
    return [
        prepare_mixed_example(
            sample,
            tokenizer,
            processor,
            media_root,
            max_length=max_length,
            extra_roots=extra_roots,
            video_num_frames=video_num_frames,
        )
        for sample in samples
    ]


class BoundedPreparedExamples(Sequence[dict[str, Any]]):
    """On-demand official inputs; only tensor storage retained by this cache is bounded.

    One current example and processor/decode temporaries may exceed the cache budget.
    Sample metadata and the set of observed indices are cheap, eager bookkeeping.
    """

    def __init__(
        self,
        samples: list[Sample],
        prepare: Callable[[Sample], dict[str, Any]],
        cache_bytes: int,
        report: dict[str, Any],
    ) -> None:
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self.samples = samples
        self.prepare = prepare
        self.budget = cache_bytes
        self.cache: OrderedDict[int, tuple[dict[str, Any], int]] = OrderedDict()
        self.observed: set[int] = set()
        self.stats: dict[str, Any] = {
            "mode": "lazy",
            "pool_examples": len(samples),
            "prepared_examples": 0,
            "preparation_calls": 0,
            "cache_budget_bytes": cache_bytes,
            "cache_retained_bytes": 0,
            "cache_peak_bytes": 0,
            "observed_max_prompt_tokens": 0,
            "media_validation": "partial; unvisited samples validated on access",
        }
        report["preparation"] = self.stats

    def __len__(self) -> int:
        return len(self.samples)

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index][0]
        sample = self.samples[index]
        try:
            example = self.prepare(sample)
        except Exception as exc:
            self.stats["failed_sample_id"] = sample.id
            self.stats["failed_sample_index"] = index
            raise ValueError(f"sample {sample.id} preparation failed: {exc}") from exc
        self.observed.add(index)
        self.stats["prepared_examples"] = len(self.observed)
        self.stats["preparation_calls"] += 1
        self.stats["observed_max_prompt_tokens"] = max(
            self.stats["observed_max_prompt_tokens"], len(example["input_ids"])
        )
        # Account storage rather than numel: views can retain a larger backing tensor.
        storages = {
            value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
            for value in example.values()
            if hasattr(value, "untyped_storage")
        }
        size = sum(storages.values())
        if size <= self.budget and self.budget > 0:
            while self.cache and self.stats["cache_retained_bytes"] + size > self.budget:
                _, (_, evicted_size) = self.cache.popitem(last=False)
                self.stats["cache_retained_bytes"] -= evicted_size
            self.cache[index] = (example, size)
            self.stats["cache_retained_bytes"] += size
            self.stats["cache_peak_bytes"] = max(
                self.stats["cache_peak_bytes"], self.stats["cache_retained_bytes"]
            )
        return example

    def validate_representatives(self) -> None:
        """Validate the first sample of each declared modality, without scanning tensors."""
        seen: set[str] = set()
        for index, sample in enumerate(self.samples):
            kind = sample.prefix.media[0].type if sample.prefix.media else "text"
            if kind not in seen:
                self[index]
                seen.add(kind)
        self.stats["preflight_prepared_examples"] = len(self.observed)


def decision_loss(
    logits: Any, token_ids: list[int], target: list[float], *, brier_weight: float = 0.0
) -> Any:
    """CE plus optional summed Brier on the joint legal-answer/U distribution in FP32."""
    torch = importlib.import_module("torch")
    if not math.isfinite(brier_weight) or brier_weight < 0:
        raise ValueError("brier_weight must be finite and nonnegative")
    if logits.ndim != 1 or not 3 <= len(token_ids) <= 21 or len(token_ids) != len(target):
        raise ValueError("loss expects one vocabulary vector and 2–20 answers plus U")
    if len(set(token_ids)) != len(token_ids) or any(
        type(i) is not int or i < 0 or i >= logits.shape[0] for i in token_ids
    ):
        raise ValueError("option token IDs must be distinct vocabulary indices")
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in target) or not math.isclose(
        math.fsum(target), 1.0, abs_tol=1e-9
    ):
        raise ValueError("target must be a finite probability distribution including U")
    selected = logits[token_ids].float()
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("nonfinite legal-token logits")
    labels = torch.tensor(target, device=selected.device, dtype=torch.float32)
    ce = -(labels * selected.log_softmax(dim=-1)).sum()
    if brier_weight == 0:
        return ce
    return ce + brier_weight * (selected.softmax(dim=-1) - labels).square().sum()


def verified_assets(model_dir: Path, manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    models = manifest.get("models", [])
    matching = [m for m in models if m.get("revision") == model_dir.name]
    if len(matching) != 1 or matching[0].get("repo_id") != "google/gemma-4-E4B-it":
        raise ValueError("model directory must identify the manifest's pinned E4B revision")
    model = matching[0]
    if not re.fullmatch(r"[0-9a-f]{40}", model_dir.name):
        raise ValueError("model revision must be immutable")
    scoped = {**manifest, "models": [model]}
    files = selected_files(scoped, metadata_only=False)
    if not any(file["kind"] == "weights" for _, file in files):
        raise ValueError("manifest contains no model weights")
    hashes = {}
    for _, file in files:
        path = model_dir / file["path"]
        if not path.resolve().is_relative_to(model_dir.resolve()):
            raise ValueError("model file escapes pinned directory")
        hashes[file["path"]] = verify_file(path, file)
    expected_weights = {f["path"] for _, f in files if f["kind"] == "weights"}
    if {p.name for p in model_dir.glob("model*.safetensors")} != expected_weights:
        raise ValueError("unverified or missing safetensors files in model directory")
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        if not shards or shards != expected_weights:
            raise ValueError("safetensors index references unverified shards")
    model_config = json.loads((model_dir / "config.json").read_text())
    if model_config.get("model_type") != "gemma4" or model_config.get("architectures") != [
        "Gemma4ForConditionalGeneration"
    ]:
        raise ValueError("unsupported model architecture")
    return hashes


def runtime_requirements() -> dict[str, str]:
    versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        versions[package] = importlib.metadata.version(package)
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
        raise ValueError("installed Transformers lacks Gemma4ForConditionalGeneration")
    if not hasattr(peft, "get_peft_model") or not hasattr(peft, "LoraConfig"):
        raise ValueError("installed PEFT lacks required LoRA APIs")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for the real Gemma smoke; CPU/MPS is not P0")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device must support bfloat16")
    return versions


def verified_init_adapter(path: Path, sha256: str) -> dict[str, str]:
    """Pin local standard PEFT safetensors and its configuration before model loading."""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("init adapter requires a lowercase SHA256")
    path = path.resolve()
    if file_sha256(path / "adapter_model.safetensors") != sha256:
        raise ValueError("init adapter weights hash mismatch")
    return {
        "path": str(path),
        "weights_sha256": sha256,
        "config_sha256": file_sha256(path / "adapter_config.json"),
    }


def attach_lora(model: Any, config: TrainConfig, init_adapter: dict[str, str] | None = None) -> Any:
    """Freeze original weights; optionally initialize only compatible decoder LoRA."""
    peft = importlib.import_module("peft")
    projections = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [
        name
        for name, _ in model.named_modules()
        if name.startswith("model.language_model.layers.") and name.split(".")[-1] in projections
    ]
    if {name.split(".")[-1] for name in targets} != projections:
        raise ValueError("cannot locate all intended Gemma decoder projections")
    lora_config = peft.LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=0.0,
        target_modules=targets,
        bias="none",
        inference_mode=False,
    )
    if init_adapter is not None:
        path = Path(init_adapter["path"])
        if verified_init_adapter(path, init_adapter["weights_sha256"]) != init_adapter:
            raise ValueError("init adapter config hash mismatch")
        # Compare every behavior-bearing field against the existing decision LoRA.
        # Serialization metadata and saved inference mode do not change its math.
        saved = json.loads((path / "adapter_config.json").read_text())
        expected = lora_config.to_dict()
        metadata = {
            "auto_mapping",
            "peft_version",
            "base_model_name_or_path",
            "revision",
            "inference_mode",
        }
        for key in saved.keys() | expected.keys():
            if key in metadata:
                continue
            actual = saved.get(key, expected.get(key))
            wanted = expected.get(key)
            if key == "target_modules":
                actual = set(actual) if isinstance(actual, list) else actual
            if actual != wanted:
                raise ValueError(f"init adapter config mismatch: {key}")
        if not {"peft_type", "r", "lora_alpha", "target_modules"} <= saved.keys():
            raise ValueError("init adapter config is incomplete")
    model.requires_grad_(False)
    adapted = peft.get_peft_model(model, lora_config)
    trainable = [name for name, p in adapted.named_parameters() if p.requires_grad]
    if not trainable or any(
        ".lora_" not in name or ".language_model.layers." not in name for name in trainable
    ):
        raise ValueError("trainable parameters extend beyond decoder LoRA")
    if init_adapter is not None:
        tensors = importlib.import_module("safetensors.torch").load_file(
            str(Path(init_adapter["path"]) / "adapter_model.safetensors")
        )
        expected_state = peft.get_peft_model_state_dict(adapted)
        if tensors.keys() != expected_state.keys() or any(
            tensors[name].shape != parameter.shape for name, parameter in expected_state.items()
        ):
            raise ValueError("init adapter tensor keys or shapes mismatch")
        torch = importlib.import_module("torch")
        if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
            raise ValueError("init adapter contains nonfinite weights")
        peft.set_peft_model_state_dict(adapted, tensors)
    return adapted


def log_training_metrics(
    telemetry: Telemetry,
    config: TrainConfig,
    report: dict[str, Any],
    *,
    mean_loss: float,
    grad_norm: float,
) -> None:
    telemetry.log(
        {
            "train/loss": mean_loss,
            "train/learning_rate": config.learning_rate,
            "train/grad_norm": grad_norm,
            "train/steps_completed": float(report["steps_completed"]),
        },
        step=int(report["steps_completed"]),
    )


def train_smoke(
    config: TrainConfig,
    model_dir: Path,
    examples: Sequence[dict[str, Any]],
    output_dir: Path,
    report: dict[str, Any],
    telemetry: Telemetry,
    resume_from: Path | None = None,
    resume_step: int = 0,
) -> None:
    """Execute 10 optimizer steps, each accumulating 10 single-sequence examples."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = attach_lora(base, config, report.get("init_adapter")).to("cuda")
    for name, parameter in model.named_parameters():
        if ".vision_tower." in name or ".audio_tower." in name:
            if parameter.requires_grad:
                raise ValueError(f"encoder tower must stay frozen under decision LoRA: {name}")
    model.train()
    # Encoders stay in eval mode as well as frozen; media cohorts only pass pixels through.
    for name, module in model.named_modules():
        if name.endswith(("vision_tower", "audio_tower")):
            module.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=0.0)
    report["trainable_parameters"] = sum(p.numel() for p in parameters)
    report["device"] = torch.cuda.get_device_name(0)
    report["cuda_version"] = torch.version.cuda
    order = list(range(len(examples)))
    random.Random(config.seed).shuffle(order)
    start_step = resume_step
    report["steps_completed"] = 0
    report["losses"] = []
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        restore_checkpoint(model, optimizer, checkpoint)
        report["steps_completed"] = start_step
        report["losses"] = list(checkpoint["losses"])
        report["resumed_from"] = str(resume_from)
    telemetry.log(
        {
            "validation/examples": float(report["examples"]),
            **(
                {
                    "validation/observed_max_prompt_tokens": float(
                        report["preparation"]["observed_max_prompt_tokens"]
                    )
                }
                if isinstance(examples, BoundedPreparedExamples)
                else {"validation/max_prompt_tokens": float(report["max_prompt_tokens"])}
            ),
            "train/learning_rate": config.learning_rate,
        },
        step=start_step,
    )
    for step in range(start_step, config.steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for offset in range(config.accumulation):
            draw = step * config.accumulation + offset
            example = examples[order[draw % len(examples)]]
            ids = torch.tensor([example["input_ids"]], device="cuda", dtype=torch.long)
            # Only whitelisted modality keys the official processor produced get forwarded.
            media_inputs = {
                key: example[key].to(ids.device) for key in MEDIA_FORWARD_KEYS if key in example
            }
            output = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
                **media_inputs,
            )
            loss = decision_loss(
                output.logits[0, -1],
                example["token_ids"],
                example["target"],
                brier_weight=getattr(config, "brier_weight", 0.0),
            )
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            (loss / config.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if float(norm) <= 0:
            raise ValueError("no LoRA gradient reached the decision loss")
        optimizer.step()
        report["steps_completed"] = step + 1
        mean_loss = math.fsum(losses) / len(losses)
        report["losses"].append(mean_loss)
        log_training_metrics(telemetry, config, report, mean_loss=mean_loss, grad_norm=float(norm))
        midway = getattr(config, "midway_step", None)
        if midway is not None and report["steps_completed"] == midway:
            midway_dir = output_dir / f"adapter-at-step-{midway:04d}"
            model.save_pretrained(
                str(midway_dir), safe_serialization=True, save_embedding_layers=False
            )
            midway_adapter = midway_dir / "adapter_model.safetensors"
            if not midway_adapter.is_file():
                raise ValueError("PEFT did not save the midway safetensors adapter")
            report.setdefault("midway_adapters", []).append(
                {
                    "step": midway,
                    "path": str(midway_dir),
                    "sha256": hashlib.sha256(midway_adapter.read_bytes()).hexdigest(),
                }
            )
            write_json(output_dir / "run.json", report)
        every = getattr(config, "checkpoint_every", None)
        if (
            every is None
            or report["steps_completed"] % every == 0
            or report["steps_completed"] == config.steps
        ):
            checkpoint_path = save_checkpoint(
                output_dir,
                step=report["steps_completed"],
                config=report["config"],
                model_revision=report["model_revision"],
                data_sha256=report["data_sha256"],
                asset_hashes=report["asset_hashes"],
                order=order,
                losses=report["losses"],
                model=model,
                optimizer=optimizer,
            )
            report["checkpoint"] = str(checkpoint_path)
        write_json(output_dir / "run.json", report)
    model.save_pretrained(
        str(output_dir / "adapter"), safe_serialization=True, save_embedding_layers=False
    )
    adapter = output_dir / "adapter" / "adapter_model.safetensors"
    if not adapter.is_file():
        raise ValueError("PEFT did not save a safetensors adapter")
    report["adapter_sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()


def code_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def run(
    config: TrainConfig,
    model_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    execute: bool,
    data_path: Path | None = None,
    monitoring: SwanLabConfig | None = None,
    resume_from: Path | None = None,
    scope: str | None = None,
    init_adapter: Path | None = None,
    init_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    report: dict[str, Any] = {
        "status": "blocked",
        "exit_code": 1,
        "code_revision": code_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": config.model_dump(),
        "model_revision": model_dir.name,
        "scope": scope
        or "text-only real-model plumbing smoke; not P0 completion or quality evidence",
        "resume_supported": True,
        "cost": None,
        "steps_completed": 0,
    }
    # Historical CE checkpoints omitted this field. Keep that identity for zero weight;
    # nonzero weights remain in config and are checked by strict resume equality.
    if report["config"].get("brier_weight") == 0:
        report["config"].pop("brier_weight")
    if report["config"].get("preparation_mode") == "eager":
        report["config"].pop("preparation_mode")
        report["config"].pop("preparation_cache_bytes")
    monitoring_config = monitoring or SwanLabConfig.from_env()
    report["monitoring"] = monitoring_config.describe()
    created = False
    accepted_resume = False
    telemetry: Telemetry | None = None
    try:
        if (init_adapter is None) != (init_adapter_sha256 is None):
            raise ValueError("init-adapter and init-adapter-sha256 must be supplied together")
        if init_adapter is not None and init_adapter_sha256 is not None:
            identity = verified_init_adapter(init_adapter, init_adapter_sha256)
            report["init_adapter"] = identity
            # Checkpoint config equality requires the same source on every resume.
            # Leave legacy config byte-for-byte unchanged when no source is provided.
            report["config"]["init_adapter"] = identity
        if execute:
            if resume_from is None:
                output_dir.mkdir(parents=True, exist_ok=False)
                created = True
            else:
                if not output_dir.is_dir():
                    raise ValueError("resume requires an existing output directory")
        samples = smoke_samples(
            config,
            data_path,
            expected_data_version=getattr(config, "expected_data_version", None),
            expected_data_versions=getattr(config, "expected_data_versions", None),
        )
        report["data_sha256"] = hashlib.sha256(data_bytes(samples)).hexdigest()
        blockers = []
        operations: list[tuple[str, Callable[[], dict[str, str]]]] = [
            ("asset_hashes", lambda: verified_assets(model_dir, manifest_path)),
            ("versions", runtime_requirements),
        ]
        for field, operation in operations:
            try:
                report[field] = operation()
            except Exception as exc:
                blockers.append(f"{field}: {type(exc).__name__}: {exc}")
        examples: Sequence[dict[str, Any]] = []
        try:
            if config.modality == "text":
                adapter = OfficialTokenizerAdapter(model_dir)
                examples = prepare_examples(samples, adapter, config.max_length)
            else:
                if not config.media_root and not config.media_roots:
                    raise ValueError("non-text training requires media_root or media_roots")
                if config.media_roots:
                    # The primary root must not repeat inside the candidate list.
                    primary_root = Path(config.media_roots[0])
                    extra_roots = [Path(root) for root in config.media_roots[1:]]
                else:
                    primary_root = Path(config.media_root or "")
                    extra_roots = []
                video_num_frames = getattr(config, "video_num_frames", 16)
                if getattr(config, "preparation_mode", "eager") == "lazy":
                    processor = importlib.import_module(
                        "transformers"
                    ).AutoProcessor.from_pretrained(
                        str(model_dir), local_files_only=True, trust_remote_code=False
                    )
                    configure_image_processor(processor, config.image_max_soft_tokens)
                    adapter = OfficialTokenizerAdapter(model_dir)
                    examples = BoundedPreparedExamples(
                        samples,
                        lambda sample: prepare_mixed_example(
                            sample,
                            adapter,
                            processor,
                            primary_root,
                            max_length=config.max_length,
                            extra_roots=extra_roots,
                            video_num_frames=video_num_frames,
                        ),
                        getattr(config, "preparation_cache_bytes", 256 * 1024 * 1024),
                        report,
                    )
                    examples.validate_representatives()
                elif config.modality == "mixed":
                    examples = prepare_mixed_examples(
                        samples,
                        model_dir,
                        primary_root,
                        config.max_length,
                        extra_roots=extra_roots,
                        video_num_frames=video_num_frames,
                        image_max_soft_tokens=config.image_max_soft_tokens,
                    )
                else:
                    examples = prepare_multimodal_examples(
                        samples,
                        model_dir,
                        primary_root,
                        config.max_length,
                        extra_roots,
                        video_num_frames=video_num_frames,
                        image_max_soft_tokens=config.image_max_soft_tokens,
                    )
            report["examples"] = len(examples)
            if not isinstance(examples, BoundedPreparedExamples):
                report["max_prompt_tokens"] = max(len(e["input_ids"]) for e in examples)
                report["preparation"] = {
                    "mode": "eager",
                    "prepared_examples": len(examples),
                    "media_validation": "all examples prepared",
                }
        except Exception as exc:
            blockers.append(f"examples: {type(exc).__name__}: {exc}")
        blockers.extend(monitoring_config.blockers())
        report["blockers"] = blockers
        if blockers:
            raise ValueError("training preflight has blockers")
        order = list(range(len(examples)))
        random.Random(config.seed).shuffle(order)
        resume_step = 0
        if execute and resume_from is not None:
            payload = data_bytes(samples)
            data_output = output_dir / "data.jsonl"
            if (
                not data_output.is_file()
                or hashlib.sha256(data_output.read_bytes()).hexdigest() != report["data_sha256"]
            ):
                raise ValueError("resume data.jsonl is missing or does not match current data")
            resume_metadata = verify_resume_request(
                output_dir,
                resume_from,
                config=report["config"],
                model_revision=report["model_revision"],
                data_sha256=report["data_sha256"],
                asset_hashes=report["asset_hashes"],
                order=order,
            )
            resume_step = int(resume_metadata["step"])
            accepted_resume = True
        if execute:
            payload = data_bytes(samples)
            data_output = output_dir / "data.jsonl"
            if resume_from is None:
                data_output.write_bytes(payload)
            report["status"] = "running"
            write_json(output_dir / "run.json", report)
            telemetry = start_swanlab(monitoring_config, report)
            try:
                train_smoke(
                    config,
                    model_dir,
                    examples,
                    output_dir,
                    report,
                    telemetry,
                    resume_from=resume_from,
                    resume_step=resume_step,
                )
            finally:
                telemetry.finish(report)
        report.update(status="completed" if execute else "ready", exit_code=0)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            status="failed" if report["status"] == "running" else "blocked",
            error=f"{type(exc).__name__}: {exc}",
        )
    report["duration_seconds"] = time.monotonic() - start
    if created or accepted_resume:
        write_json(output_dir / "run.json", report)
    return report
