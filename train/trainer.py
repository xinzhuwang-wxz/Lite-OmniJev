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
    canonicalize_media,
    configure_image_processor,
    load_media,
    resolve_media,
)
from train.pair_objective import claim_pair_loss
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
    reference_sidecar: str | None = Field(default=None, exclude_if=lambda v: v is None)
    reference_sidecar_sha256: str | None = Field(default=None, exclude_if=lambda v: v is None)
    reference_ledger: str | None = Field(default=None, exclude_if=lambda v: v is None)
    reference_ledger_sha256: str | None = Field(default=None, exclude_if=lambda v: v is None)
    reference_kl_weight: float = Field(
        default=0.0, allow_inf_nan=False, exclude_if=lambda v: v == 0
    )
    reference_replay_data: str | None = Field(default=None, exclude_if=lambda v: v is None)
    reference_replay_sha256: str | None = Field(default=None, exclude_if=lambda v: v is None)
    pair_loss_weight: float = Field(default=0.0, allow_inf_nan=False, exclude_if=lambda v: v == 0)
    pair_sidecar: str | None = Field(default=None, exclude_if=lambda v: v is None)
    pair_sidecar_sha256: str | None = Field(default=None, exclude_if=lambda v: v is None)
    pair_ledger: str | None = Field(default=None, exclude_if=lambda v: v is None)
    pair_ledger_sha256: str | None = Field(default=None, exclude_if=lambda v: v is None)
    preparation_mode: Literal["eager", "lazy"] = "eager"
    preparation_cache_bytes: int = Field(default=256 * 1024 * 1024, ge=0)

    @model_validator(mode="after")
    def validate_loop(self) -> LoopConfig:
        if self.pair_loss_weight not in (0.0, 0.1):
            raise ValueError("claim pair pilot weight must be 0 or 0.1")
        pair_inputs = (
            self.pair_sidecar,
            self.pair_sidecar_sha256,
            self.pair_ledger,
            self.pair_ledger_sha256,
        )
        if self.pair_loss_weight:
            if self.reference_kl_weight or self.brier_weight:
                raise ValueError("claim pair pilot requires pure gold CE without KL/Brier")
            if (self.seed, self.steps, self.accumulation, self.sample_count, self.modality) != (
                42,
                10,
                12,
                120,
                "mixed",
            ):
                raise ValueError(
                    "claim pair pilot requires seed42/mixed/10steps/accum12/120samples"
                )
            if not all(pair_inputs):
                raise ValueError("claim pair pilot requires SHA-bound sidecar and TRAIN ledger")
        elif any(pair_inputs):
            raise ValueError("claim pair inputs require nonzero pair weight")
        if self.reference_kl_weight not in (0.0, 0.5):
            raise ValueError("reference pilot weight must be 0 or 0.5")
        if self.reference_kl_weight:
            if self.brier_weight != 0:
                raise ValueError("reference pilot keeps gold CE without Brier")
            if self.seed != 42 or self.steps * self.accumulation != 300 or self.sample_count != 300:
                raise ValueError("reference pilot requires seed42 and exactly 300 unique draws")
            if not all(
                (
                    self.reference_sidecar,
                    self.reference_sidecar_sha256,
                    self.reference_ledger,
                    self.reference_ledger_sha256,
                    self.reference_replay_data,
                    self.reference_replay_sha256,
                )
            ):
                raise ValueError("reference pilot requires SHA-bound sidecar and TRAIN ledger")
        elif any(
            (
                self.reference_sidecar,
                self.reference_sidecar_sha256,
                self.reference_ledger,
                self.reference_ledger_sha256,
                self.reference_replay_data,
                self.reference_replay_sha256,
            )
        ):
            raise ValueError("reference inputs require nonzero reference KL weight")
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
    "mm_token_type_ids",
    "pixel_values",
    "image_position_ids",
    "input_features",
    "input_features_mask",
    "pixel_values_videos",
    "video_position_ids",
    "num_frames_per_video",
)


def validated_media_types(sample: Sample) -> tuple[str, ...]:
    """Admit single media or one image/audio pair, preserving template order."""
    kinds = tuple(item.type for item in sample.prefix.media)
    supported = (len(kinds) == 1 and kinds[0] in MEDIA_TYPES) or (
        len(kinds) == 2 and set(kinds) == {"image", "audio"}
    )
    declared = all(kind in sample.meta.modality for kind in kinds)
    if len(kinds) == 2:
        declared = declared and set(sample.meta.modality) == set(kinds)
    if not supported or not declared:
        raise ValueError(
            f"sample {sample.id} must carry exactly one image, audio or video item "
            "or one image+audio pair matching the declared modality"
        )
    return kinds


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
                try:
                    validated_media_types(sample)
                except ValueError as exc:
                    raise ValueError(
                        "mixed media samples require exactly one media item or an image+audio "
                        "pair matching the declared modality"
                    ) from exc
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
                "observed_modalities": list(sample.meta.modality),
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
    media_types = validated_media_types(sample)
    processor_kwargs: dict[str, Any] = {}
    for item in sample.prefix.media:
        try:
            media_path = verified_media_file(
                media_root, item.uri, item.sha256, extra_roots=list(extra_roots)
            )
        except ValueError as exc:
            raise ValueError(f"sample {sample.id}: {exc}") from exc
        loaded = load_media(
            media_path,
            item.type,
            num_frames=video_num_frames,
            max_seconds=30 if item.type == "audio" else 60,
        )
        if len(media_types) == 2 and item.type == "audio":
            loaded = canonicalize_media("audio", loaded).hf_value
        processor_kwargs.update(media_processor_kwargs(item.type, loaded))
    if len(media_types) == 2:
        processor_kwargs["add_special_tokens"] = False
    payload = render_payload(Prefix(state=sample.prefix.state), question)
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": kind} for kind in media_types]
                + [{"type": "text", "text": payload}],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("official media template returned empty prompt")
    tokenizer = processor.tokenizer
    tokens = (*answer_tokens(question), "U")
    mapping = check_answer_tokens(tokenizer, question)
    inputs = processor(
        text=[prompt],
        return_tensors="pt",
        truncation=False,
        **processor_kwargs,
    )
    input_ids = inputs["input_ids"][0].tolist()
    missing = [
        key for kind in media_types for key in MEDIA_REQUIRED_KEYS[kind] if key not in inputs
    ]
    if missing:
        raise ValueError(f"processor omitted {', '.join(missing)} for {media_types}")
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
        "observed_modalities": list(sample.meta.modality),
        "input_ids": input_ids,
        "token_ids": [mapping[t] for t in tokens],
        "answer_token_ids": mapping,
        "target": expected,
    }
    for key in MEDIA_FORWARD_KEYS:
        if key in inputs:
            example[key] = inputs[key]
    if "video" in media_types:
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
    logits: Any,
    token_ids: list[int],
    target: list[float],
    *,
    brier_weight: float = 0.0,
    reference_probs: Any = None,
    reference_kl_weight: float = 0.0,
    reference_enabled: bool = False,
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
    loss = (
        ce
        if brier_weight == 0
        else ce + brier_weight * (selected.softmax(dim=-1) - labels).square().sum()
    )
    if reference_kl_weight == 0:
        return loss
    if reference_kl_weight != 0.5:
        raise ValueError("reference pilot weight must be 0 or 0.5")
    if reference_probs is None:
        raise ValueError("reference distribution is required")
    q = torch.as_tensor(reference_probs, device=selected.device, dtype=torch.float32).detach()
    if q.shape != selected.shape or not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
        raise ValueError("invalid reference distribution")
    if not bool(torch.isclose(q.sum(), q.new_tensor(1.0), atol=1e-6, rtol=0)):
        raise ValueError("reference distribution must sum to one including U")
    gold = [i for i, value in enumerate(target) if value == 1.0]
    winners = (q == q.max()).nonzero().flatten().tolist()
    enabled = reference_enabled and len(gold) == 1 and winners == gold
    if not enabled:
        return loss
    positive = q > 0
    kl = (q[positive] * (q[positive].log() - selected.log_softmax(-1)[positive])).sum()
    return loss + reference_kl_weight * kl


REFERENCE_FIELDS = (
    "reference_sidecar",
    "reference_sidecar_sha256",
    "reference_ledger",
    "reference_ledger_sha256",
    "reference_kl_weight",
    "reference_replay_data",
    "reference_replay_sha256",
)


def reference_prepared_hash(example: dict[str, Any]) -> str:
    """Bind actual IDs, gold, modality and every forwarded CPU tensor byte."""
    record = {}
    for key in sorted(example):
        value = example[key]
        if hasattr(value, "detach"):
            tensor = value.detach().cpu().contiguous()
            torch = importlib.import_module("torch")
            value = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(
                    tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        record[key] = value
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reference_bound_file(path: Path, expected_sha256: str | None) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("reference input must be an absolute regular file, not a symlink")
    if (
        not expected_sha256
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or file_sha256(path) != expected_sha256
    ):
        raise ValueError("reference input SHA mismatch")
    return path


def reference_header(
    config: LoopConfig,
    samples: list[Sample],
    identity: dict[str, str],
    model_hashes: dict[str, str],
) -> dict[str, Any]:
    if any(sample.meta.pool != "train" for sample in samples):
        raise ValueError("reference accepts TRAIN only")
    replay_path = Path(config.reference_replay_data or "")
    reference_bound_file(replay_path, config.reference_replay_sha256)
    replay_bytes = replay_path.read_bytes()
    if hashlib.sha256(replay_bytes).hexdigest() != config.reference_replay_sha256:
        raise ValueError("reference replay data changed while reading")
    replay = [
        Sample.model_validate_json(line)
        for line in replay_bytes.decode().splitlines()
        if line.strip()
    ]
    if any(sample.meta.pool != "train" for sample in replay):
        raise ValueError("reference replay source must be TRAIN only")
    originals = {sample.id: data_bytes([sample]) for sample in replay}
    if len(originals) != len(replay) or any(
        "video" not in sample.meta.modality and originals.get(sample.id) != data_bytes([sample])
        for sample in samples
    ):
        raise ValueError("non-video samples must match frozen TRAIN replay exactly")
    return {
        "schema": "train-reference-v1",
        "data_sha256": hashlib.sha256(data_bytes(samples)).hexdigest(),
        "reference_adapter": dict(identity),
        "asset_hashes": dict(model_hashes),
        "config": {k: v for k, v in config.model_dump().items() if k not in REFERENCE_FIELDS},
        "ledger_sha256": config.reference_ledger_sha256,
        "replay_data_sha256": config.reference_replay_sha256,
    }


def reference_order(config: TrainConfig, count: int) -> list[int]:
    order = list(range(count))
    random.Random(config.seed).shuffle(order)
    return [order[draw % count] for draw in range(config.steps * config.accumulation)]


def reference_row(
    draw: int, index: int, sample: Sample, example: dict[str, Any], probabilities: list[float]
) -> dict[str, Any]:
    return {
        "draw": draw,
        "sample_index": index,
        "id": sample.id,
        "sample_sha256": hashlib.sha256(data_bytes([sample])).hexdigest(),
        "prepared_sha256": reference_prepared_hash(example),
        "token_ids": example["token_ids"],
        "reference_probs": probabilities,
    }


def validate_reference_row(
    row: dict[str, Any], draw: int, index: int, sample: Sample, example: dict[str, Any]
) -> bool:
    if (
        sample.meta.pool != "train"
        or type(row.get("draw")) is not int
        or type(row.get("sample_index")) is not int
    ):
        raise ValueError("reference requires TRAIN and integer draw identity")
    expected = reference_row(draw, index, sample, example, row.get("reference_probs", []))
    if row != expected or len(set(row["token_ids"])) != len(row["token_ids"]):
        raise ValueError("reference draw/sample/input/token binding mismatch")
    probs = row["reference_probs"]
    if (
        len(probs) != len(example["target"])
        or any(type(p) not in (float, int) or not math.isfinite(p) or p < 0 or p > 1 for p in probs)
        or not math.isclose(math.fsum(probs), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("invalid reference probabilities including U")
    mapping = example.get("answer_token_ids")
    if mapping is None:
        # prepare_examples already checked these IDs at the official generation
        # boundary. Reconstruct locally; never mutate/hash a different prepared example.
        if (
            sample.meta.modality != ["text"]
            or sample.prefix.media
            or len(sample.questions) != 1
            or any(key in example for key in MEDIA_FORWARD_KEYS)
        ):
            raise ValueError("media reference requires original answer-token mapping")
        labels = (*answer_tokens(sample.questions[0]), "U")
        ids = example["token_ids"]
        if len(ids) != len(labels) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("text reference answer length/type mismatch")
        mapping = dict(zip(labels, ids, strict=True))
    if (
        not mapping
        or list(mapping)[-1] != "U"
        or list(mapping.values()) != row["token_ids"]
        or len(set(mapping.values())) != len(mapping)
    ):
        raise ValueError("reference requires ordered complete answers including U")
    gold = [i for i, p in enumerate(example["target"]) if p == 1.0]
    winners = [i for i, p in enumerate(probs) if p == max(probs)]
    return "video" not in sample.meta.modality and len(gold) == 1 and winners == gold


def load_reference(
    config: LoopConfig,
    samples: list[Sample],
    examples: Sequence[dict[str, Any]],
    report: dict[str, Any],
    data_path: Path | None,
) -> list[dict[str, Any]]:
    from scripts.train_s1 import verify_data_is_train_pool

    if len(samples) != 300 or len(examples) != 300 or len({sample.id for sample in samples}) != 300:
        raise ValueError("reference pilot requires 300 unique prepared TRAIN samples")
    if data_path is None or not config.reference_ledger:
        raise ValueError("reference requires explicit TRAIN data and ledger")
    ledger = Path(config.reference_ledger)
    reference_bound_file(ledger, config.reference_ledger_sha256)
    verify_data_is_train_pool(data_path, [ledger])
    path = Path(config.reference_sidecar or "")
    reference_bound_file(path, config.reference_sidecar_sha256)
    payload_bytes = path.read_bytes()
    if hashlib.sha256(payload_bytes).hexdigest() != config.reference_sidecar_sha256:
        raise ValueError("reference sidecar changed while reading")
    payload = json.loads(payload_bytes)
    if set(payload) != {"header", "rows"} or payload["header"] != reference_header(
        config, samples, report["init_adapter"], report["asset_hashes"]
    ):
        raise ValueError("reference header/data/adapter/config binding mismatch")
    order = reference_order(config, len(examples))
    if len(payload["rows"]) != len(order):
        raise ValueError("missing or extra reference draws")
    for draw, index in enumerate(order):
        validate_reference_row(payload["rows"][draw], draw, index, samples[index], examples[index])
    result: list[dict[str, Any]] = payload["rows"]
    return result


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
    step_metrics: dict[str, float] | None = None,
) -> None:
    if step_metrics is not None:
        history = report.setdefault("step_metrics", [])
        history.append({"step": int(report["steps_completed"]), **step_metrics})
        del history[:-256]
    telemetry.log(
        {
            **{f"train/{key}": value for key, value in (step_metrics or {}).items()},
            "train/loss": mean_loss,
            "train/learning_rate": config.learning_rate,
            "train/grad_norm": grad_norm,
            "train/steps_completed": float(report["steps_completed"]),
        },
        step=int(report["steps_completed"]),
    )


def claim_pair_row(
    positive_draw: int,
    positive_index: int,
    negative_index: int,
    positive: Sample,
    negative: Sample,
    positive_example: dict[str, Any],
    negative_example: dict[str, Any],
) -> dict[str, Any]:
    """Bind controlled claims to actual shared image/audio components, not family names alone."""
    if positive.meta.pool != "train" or negative.meta.pool != "train":
        raise ValueError("claim pairs accept TRAIN only")
    if positive.meta.family_id != negative.meta.family_id or positive.id == negative.id:
        raise ValueError("claim pairs require distinct claims from the same family")
    components = sorted((m.type, m.sha256) for m in positive.prefix.media)
    if components != sorted((m.type, m.sha256) for m in negative.prefix.media) or [
        kind for kind, _ in components
    ] != ["audio", "image"]:
        raise ValueError("claim pairs require identical image and audio component hashes")
    for sample, example, gold in (
        (positive, positive_example, "yes"),
        (negative, negative_example, "no"),
    ):
        question = sample.questions[0]
        mapping = example.get("answer_token_ids", {})
        if (
            len(sample.questions) != 1
            or question.type != "noul"
            or not isinstance(question.target, HardTarget)
            or question.target.value != gold
            or set(mapping) != {"yes", "no", "U"}
            or len(set(mapping.values())) != 3
            or any(type(v) is not int or v < 0 for v in mapping.values())
            or example["token_ids"] != [mapping[t] for t in ("yes", "no", "U")]
            or example["target"] != [float(t == gold) for t in ("yes", "no", "U")]
            or not all(k in example for k in ("pixel_values", "input_features"))
            or "pixel_values_videos" in example
        ):
            raise ValueError("claim pair gold/mapping/prepared crossmodal binding mismatch")
    return {
        "positive_draw": positive_draw,
        "negative_draw": positive_draw + 1,
        "positive_sample_index": positive_index,
        "negative_sample_index": negative_index,
        "positive_sample_sha256": hashlib.sha256(data_bytes([positive])).hexdigest(),
        "negative_sample_sha256": hashlib.sha256(data_bytes([negative])).hexdigest(),
        "positive_prepared_sha256": reference_prepared_hash(positive_example),
        "negative_prepared_sha256": reference_prepared_hash(negative_example),
        "family_id": positive.meta.family_id,
        "components": [{"type": kind, "sha256": digest} for kind, digest in components],
    }


def load_claim_pairs(
    config: LoopConfig,
    samples: list[Sample],
    examples: Sequence[dict[str, Any]],
    data_path: Path | None,
) -> dict[int, dict[str, Any]]:
    """Audit every fixed pair before model loading, with external SHA-pinned TRAIN ledger."""
    from scripts.train_s1 import verify_data_is_train_pool

    if (
        data_path is None
        or len(samples) != 120
        or len(examples) != 120
        or any(s.meta.pool != "train" for s in samples)
    ):
        raise ValueError("claim pair pilot requires explicit TRAIN data")
    ledger = reference_bound_file(Path(config.pair_ledger or ""), config.pair_ledger_sha256)
    verify_data_is_train_pool(data_path, [ledger])
    sidecar = reference_bound_file(Path(config.pair_sidecar or ""), config.pair_sidecar_sha256)
    sidecar_bytes = sidecar.read_bytes()
    if hashlib.sha256(sidecar_bytes).hexdigest() != config.pair_sidecar_sha256:
        raise ValueError("claim pair sidecar changed while reading")
    payload = json.loads(sidecar_bytes)
    header = {
        "schema": "train-claim-pairs-v1",
        "data_sha256": hashlib.sha256(data_bytes(samples)).hexdigest(),
        "ledger_sha256": config.pair_ledger_sha256,
        "seed": config.seed,
        "steps": config.steps,
        "accumulation": config.accumulation,
    }
    if set(payload) != {"header", "pairs"} or payload["header"] != header:
        raise ValueError("claim pair header/data/ledger/config mismatch")
    order = reference_order(config, len(examples))
    expected = []
    for step in range(config.steps):
        for offset in (0, 2):
            draw = step * 12 + offset
            pi, ni = order[draw], order[draw + 1]
            expected.append(
                claim_pair_row(draw, pi, ni, samples[pi], samples[ni], examples[pi], examples[ni])
            )
    if json.dumps(payload["pairs"], sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError("claim pair mask/draw/sample/prepared/component binding mismatch")
    if file_sha256(ledger) != config.pair_ledger_sha256:
        raise ValueError("claim pair ledger changed during audit")
    return {row["positive_draw"]: row for row in expected}


def validate_invocation_steps(invocation_steps: int | None) -> None:
    """Execution limits never participate in immutable training/checkpoint identity."""
    if invocation_steps is not None and (
        type(invocation_steps) is not int or invocation_steps <= 0
    ):
        raise ValueError("invocation_steps must be a positive integer, not bool")


def train_smoke(
    config: TrainConfig,
    model_dir: Path,
    examples: Sequence[dict[str, Any]],
    output_dir: Path,
    report: dict[str, Any],
    telemetry: Telemetry,
    resume_from: Path | None = None,
    resume_step: int = 0,
    reference_rows: list[dict[str, Any]] | None = None,
    reference_samples: list[Sample] | None = None,
    pair_rows: dict[int, dict[str, Any]] | None = None,
    invocation_steps: int | None = None,
) -> None:
    """Train through a global immutable trajectory, optionally stopping at a committed step."""
    validate_invocation_steps(invocation_steps)
    if invocation_steps is not None and resume_step >= config.steps:
        raise ValueError("invocation checkpoint already reached the global step target")
    end_step = (
        config.steps
        if invocation_steps is None
        else min(config.steps, resume_step + invocation_steps)
    )
    pair_weight = getattr(config, "pair_loss_weight", 0.0)
    if pair_weight and (
        pair_rows is None
        or sorted(pair_rows)
        != [step * 12 + offset for step in range(config.steps) for offset in (0, 2)]
    ):
        raise ValueError("claim pair pilot fixed mask binding missing")
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
    report["step_metrics_boundary"] = {
        "step_wall_seconds": (
            "optimizer step wall time including input access and CUDA synchronization; "
            "excludes logging/checkpoint writes"
        ),
        "cpu_prepare_wait_seconds": (
            "elapsed host time accessing prepared examples, including lazy decode/cache lookup"
        ),
        "invocation_samples_seen": "actual draws in this invocation; resets on resume",
        "peak_cuda_bytes": "allocator peak since process peak reset, not per-step allocation",
        "modality_exposure": (
            "sample.meta.modality labels only; screenshot rows declared image stay image; "
            "counts are sample exposures, not media objects; absent labels are not inferred"
        ),
        "modality_loss": (
            "decision loss grouped by declared input modality; "
            "crossmodal also contributes to image/audio"
        ),
        "history_limit": 256,
    }
    samples_seen = 0
    for step in range(start_step, end_step):
        step_started = time.perf_counter()
        prepare_wait = 0.0
        input_tokens = 0
        modality_losses: dict[str, list[float]] = {}
        optimizer.zero_grad(set_to_none=True)
        losses = []
        pair_losses: list[float] = []
        pending_pair: tuple[Any, Any, dict[str, Any]] | None = None
        reference_ce, reference_kl, reference_enabled_count = [], [], 0
        for offset in range(config.accumulation):
            draw = step * config.accumulation + offset
            prepare_started = time.perf_counter()
            example = examples[order[draw % len(examples)]]
            prepare_wait += time.perf_counter() - prepare_started
            input_tokens += len(example["input_ids"])
            samples_seen += 1
            pair_row = (
                pair_rows.get(draw - (offset % 2))
                if pair_weight and offset < 4 and pair_rows is not None
                else None
            )
            if pair_weight and offset < 4:
                role = "positive" if offset % 2 == 0 else "negative"
                if (
                    pair_row is None
                    or reference_prepared_hash(example) != pair_row[role + "_prepared_sha256"]
                ):
                    raise ValueError("claim pair runtime prepared/draw binding mismatch")
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
            if getattr(config, "reference_kl_weight", 0):
                index = order[draw % len(examples)]
                if reference_rows is None or reference_samples is None:
                    raise ValueError("reference validation missing")
                reference = reference_rows[draw]
                enabled = validate_reference_row(
                    reference, draw, index, reference_samples[index], example
                )
                ce = loss
                loss = decision_loss(
                    output.logits[0, -1],
                    example["token_ids"],
                    example["target"],
                    brier_weight=getattr(config, "brier_weight", 0.0),
                    reference_probs=reference["reference_probs"],
                    reference_kl_weight=0.5,
                    reference_enabled=enabled,
                )
                reference_ce.append(float(ce.detach().cpu()))
                reference_kl.append(float(((loss - ce) / 0.5).detach().cpu()))
                reference_enabled_count += int(enabled)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training loss")
            if pair_weight and offset < 4:
                if offset % 2 == 0:
                    pending_pair = (output.logits[0, -1], loss, example)
                else:
                    if pending_pair is None or pair_row is None:
                        raise ValueError("claim pair positive graph missing")
                    positive_logits, positive_ce, positive_example = pending_pair
                    pair_loss = claim_pair_loss(
                        positive_logits,
                        output.logits[0, -1],
                        positive_token_ids=positive_example["answer_token_ids"],
                        negative_token_ids=example["answer_token_ids"],
                        positive_gold="yes",
                        negative_gold="no",
                        positive_family_id=pair_row["family_id"],
                        negative_family_id=pair_row["family_id"],
                    )
                    (
                        (positive_ce + loss + pair_weight * pair_loss) / config.accumulation
                    ).backward()
                    pair_losses.append(float(pair_loss.detach().cpu()))
                    pending_pair = None
                    del positive_logits, positive_ce, positive_example, pair_loss
            else:
                (loss / config.accumulation).backward()
            observed_loss = float(loss.detach().cpu())
            losses.append(observed_loss)
            modalities = set(example.get("observed_modalities", [])) & {
                "text",
                "image",
                "screenshot",
                "audio",
                "video",
            }
            if {"image", "audio"} <= modalities:
                modalities.add("crossmodal")
            for modality in modalities:
                modality_losses.setdefault(modality, []).append(observed_loss)
        if pair_weight and (len(pair_losses) != 2 or pending_pair is not None):
            raise ValueError("claim pair step requires exactly two completed pairs")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        if float(norm) <= 0:
            raise ValueError("no LoRA gradient reached the decision loss")
        optimizer.step()
        report["steps_completed"] = step + 1
        mean_loss = math.fsum(losses) / len(losses)
        if pair_weight:
            mean_loss += pair_weight * math.fsum(pair_losses) / config.accumulation
        report["losses"].append(mean_loss)
        torch.cuda.synchronize()
        step_metrics = {
            "step_wall_seconds": time.perf_counter() - step_started,
            "cpu_prepare_wait_seconds": prepare_wait,
            "invocation_samples_seen": float(samples_seen),
            "step_samples": float(len(losses)),
            "input_tokens": float(input_tokens),
            "peak_cuda_bytes": float(torch.cuda.max_memory_allocated()),
        }
        if pair_weight:
            step_metrics.update(
                gold_ce=math.fsum(losses) / len(losses),
                pair_loss=math.fsum(pair_losses) / len(pair_losses),
                pair_count=float(len(pair_losses)),
                pair_loss_weight=pair_weight,
            )
        if reference_ce:
            step_metrics.update(
                gold_ce=math.fsum(reference_ce) / len(reference_ce),
                reference_kl=math.fsum(reference_kl) / len(reference_kl),
                reference_enabled_count=float(reference_enabled_count),
            )
        for modality, observed_losses in modality_losses.items():
            step_metrics[f"{modality}/samples"] = float(len(observed_losses))
            step_metrics[f"{modality}/loss"] = math.fsum(observed_losses) / len(observed_losses)
        log_training_metrics(
            telemetry,
            config,
            report,
            mean_loss=mean_loss,
            grad_norm=float(norm),
            step_metrics=step_metrics,
        )
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
            or (invocation_steps is not None and report["steps_completed"] == end_step)
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
    adapter_dir = (
        output_dir / "adapter"
        if invocation_steps is None
        else output_dir / f"adapter-invocation-step-{report['steps_completed']:04d}"
    )
    if invocation_steps is not None and adapter_dir.exists():
        raise ValueError("invocation adapter already exists; refusing to overwrite")
    model.save_pretrained(str(adapter_dir), safe_serialization=True, save_embedding_layers=False)
    adapter = adapter_dir / "adapter_model.safetensors"
    if invocation_steps is not None:
        report["adapter_dir"] = str(adapter_dir)
        report["invocation"] = {
            "steps_requested": invocation_steps,
            "start_step": start_step,
            "steps_completed": report["steps_completed"] - start_step,
            "global_steps_target": config.steps,
            "next_draw": report["steps_completed"] * config.accumulation,
            "stop_reason": (
                "global_steps_completed"
                if report["steps_completed"] == config.steps
                else "invocation_steps_limit"
            ),
        }
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
    invocation_steps: int | None = None,
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
    if not getattr(config, "reference_kl_weight", 0):
        for field in REFERENCE_FIELDS:
            report["config"].pop(field, None)
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
    reference_rows = None
    pair_rows = None
    try:
        validate_invocation_steps(invocation_steps)
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
        if isinstance(config, LoopConfig) and config.reference_kl_weight:
            reference_rows = load_reference(config, samples, examples, report, data_path)
            report["reference"] = {
                "sidecar_sha256": config.reference_sidecar_sha256,
                "draws": len(reference_rows),
                "weight": 0.5,
            }
        if isinstance(config, LoopConfig) and config.pair_loss_weight:
            pair_rows = load_claim_pairs(config, samples, examples, data_path)
            report["claim_pairs"] = {
                "pairs": len(pair_rows),
                "weight": config.pair_loss_weight,
                "sidecar_sha256": config.pair_sidecar_sha256,
                "ledger_sha256": config.pair_ledger_sha256,
                "objective": "gold CE + controlled claim softplus; not RL",
            }
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
            if invocation_steps is not None and resume_step >= config.steps:
                raise ValueError("invocation checkpoint already reached the global step target")
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
                invocation_args = (
                    {} if invocation_steps is None else {"invocation_steps": invocation_steps}
                )
                train_smoke(
                    config,
                    model_dir,
                    examples,
                    output_dir,
                    report,
                    telemetry,
                    resume_from=resume_from,
                    resume_step=resume_step,
                    reference_rows=reference_rows,
                    reference_samples=samples if reference_rows is not None else None,
                    pair_rows=pair_rows,
                    **invocation_args,
                )
            finally:
                telemetry.finish(report)
        if execute and invocation_steps is not None:
            status = "completed" if report["steps_completed"] == config.steps else "paused"
        else:
            status = "completed" if execute else "ready"
        report.update(status=status, exit_code=0)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            status="failed" if report["status"] == "running" else "blocked",
            error=f"{type(exc).__name__}: {exc}",
        )
    report["duration_seconds"] = time.monotonic() - start
    if created or accepted_resume:
        write_json(output_dir / "run.json", report)
    return report
