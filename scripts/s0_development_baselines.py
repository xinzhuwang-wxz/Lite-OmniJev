"""Run frozen development-set HF vocabulary baselines for official backbones."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from data.schema import HardTarget, InsufficientTarget, Prefix, Question, Sample, SoftTarget
from eval.metrics import evaluate
from scripts.assets import write_json
from scripts.s0_public_baselines import (
    ALLOWED_DTYPES,
    DEFAULT_DTYPE,
    MODEL_SPECS,
    code_hashes,
    configure_torch_runtime,
    model_evidence,
    now_iso,
    report_base,
    sha256_file,
    sha256_json,
    torch_dtype,
)
from train.multimodal import canonicalize_media, load_media, resolve_media
from train.readout import read_probabilities
from train.template import check_answer_tokens, render_payload

BASELINE_VERSION = "s0-development-baselines-v1"
EXPECTED_ROWS = 500
MAX_PROMPT_TOKENS = 8192
MAX_MEDIA_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_AUDIO_SECONDS = 30.0
MAX_VIDEO_SECONDS = 60.0
VIDEO_FRAMES = 16
SUPPORTED_COHORTS = ("text", "image", "screenshot", "audio", "video", "crossmodal")
MODEL_ARCHITECTURES = MODEL_SPECS
REQUIRED_CODE_FILES = (
    "scripts/s0_development_baselines.py",
    "scripts/s0_public_baselines.py",
    "train/readout.py",
    "train/template.py",
    "train/multimodal.py",
    "eval/metrics.py",
    "data/schema.py",
)


class DevelopmentBaselineError(RuntimeError):
    """Raised when development baseline evidence is not admissible."""


class UnsupportedDevelopmentBaseline(RuntimeError):
    """Raised for a declared unsupported model/cohort pair before GPU use."""


class Processor(Protocol):
    tokenizer: Any

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str | list[int]: ...

    def __call__(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PreparedSample:
    sample: Sample
    question: Question
    prompt: str
    inputs: Mapping[str, Any]
    answer_token_ids: dict[str, int]
    target_index: int
    prompt_sha256: str
    payload_sha256: str
    input_sha256: str
    prepare_wall_seconds: float
    prompt_tokens: int
    input_summary: dict[str, Any]


ForwardFn = Callable[[PreparedSample], Mapping[int, float]]
ProcessorFactory = Callable[[Path], Processor]
ForwardFactory = Callable[[Path, str, str], tuple[ForwardFn, Callable[[], None]]]


def _code_hash_subset() -> dict[str, str]:
    all_hashes = code_hashes()
    for relative in REQUIRED_CODE_FILES:
        path = Path(__file__).resolve().parents[1] / relative
        all_hashes[relative] = sha256_file(path)
    return {key: all_hashes[key] for key in REQUIRED_CODE_FILES}


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DevelopmentBaselineError(f"JSON root must be an object: {path}")
    return value


def bind_build_report(report_path: Path, dev_jsonl: Path) -> dict[str, Any]:
    report = load_json_object(report_path)
    if report.get("status") != "passed":
        raise DevelopmentBaselineError("build report must be passed")
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get("dev_jsonl_sha256"), str):
        raise DevelopmentBaselineError("build report lacks outputs.dev_jsonl_sha256")
    actual = sha256_file(dev_jsonl)
    expected = str(outputs["dev_jsonl_sha256"])
    if actual != expected:
        raise DevelopmentBaselineError("dev_jsonl_sha256 does not bind the provided dev JSONL")
    return {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "outputs_dev_jsonl_sha256": expected,
        "bound_dev_jsonl_sha256": actual,
    }


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                samples.append(Sample.model_validate_json(stripped))
            except ValueError as exc:
                raise DevelopmentBaselineError(f"invalid sample JSON at line {line_no}") from exc
    return samples


def _expected_media_types(cohort: str, sample: Sample) -> list[str]:
    if cohort == "text":
        return []
    if cohort == "screenshot":
        return ["image"]
    if cohort in {"image", "audio", "video"}:
        return [cohort]
    if cohort == "crossmodal":
        return [item.type for item in sample.prefix.media]
    raise DevelopmentBaselineError(f"unsupported cohort: {cohort}")


def validate_samples(samples: Sequence[Sample], *, cohort: str) -> dict[str, Any]:
    if cohort not in SUPPORTED_COHORTS:
        raise DevelopmentBaselineError(f"unsupported cohort: {cohort}")
    if len(samples) != EXPECTED_ROWS:
        raise DevelopmentBaselineError(
            f"expected {EXPECTED_ROWS} development rows, found {len(samples)}"
        )
    ids: set[str] = set()
    families: set[str] = set()
    question_types: dict[str, int] = {}
    option_sizes: dict[int, int] = {}
    media_counts: dict[str, int] = {"image": 0, "audio": 0, "video": 0}
    for sample in samples:
        if sample.id in ids:
            raise DevelopmentBaselineError("duplicate sample id")
        ids.add(sample.id)
        if sample.meta.pool != "dev":
            raise DevelopmentBaselineError("development baselines require pool=dev")
        if sample.meta.family_id in families:
            raise DevelopmentBaselineError("development baselines require unique item families")
        families.add(sample.meta.family_id)
        if len(sample.questions) != 1:
            raise DevelopmentBaselineError("development baseline samples must be single-question")
        question = sample.questions[0]
        if isinstance(question.target, SoftTarget):
            raise DevelopmentBaselineError("teacher soft gold targets are not supported")
        if isinstance(question.target, InsufficientTarget):
            raise DevelopmentBaselineError("insufficient targets are not supported for scoring")
        if not isinstance(question.target, HardTarget):
            raise DevelopmentBaselineError("development baseline samples require hard gold targets")
        tokens = answer_tokens_for(question)
        if len(tokens) < 2:
            raise DevelopmentBaselineError("single-option questions are refused")
        question_types[question.type] = question_types.get(question.type, 0) + 1
        option_sizes[len(tokens)] = option_sizes.get(len(tokens), 0) + 1
        expected = _expected_media_types(cohort, sample)
        got = [item.type for item in sample.prefix.media]
        if cohort == "crossmodal":
            if len(got) < 2:
                raise DevelopmentBaselineError(
                    "crossmodal samples require at least two media items"
                )
            if any(item not in {"image", "audio", "video"} for item in got):
                raise DevelopmentBaselineError("crossmodal samples contain unsupported media")
        elif got != expected:
            raise DevelopmentBaselineError("cohort and sample media do not agree")
        for media_type in got:
            media_counts[media_type] += 1
        for media_type in got:
            if media_type not in sample.meta.modality:
                raise DevelopmentBaselineError("media type missing from sample metadata modality")
    return {
        "rows": len(samples),
        "unique_ids": len(ids),
        "unique_families": len(families),
        "question_types": dict(sorted(question_types.items())),
        "option_sizes": {str(key): value for key, value in sorted(option_sizes.items())},
        "media_counts": media_counts,
    }


def answer_tokens_for(question: Question) -> tuple[str, ...]:
    from data.schema import answer_tokens

    return answer_tokens(question)


def target_index(question: Question) -> int:
    if not isinstance(question.target, HardTarget):
        raise DevelopmentBaselineError("hard target required")
    tokens = answer_tokens_for(question)
    value = question.target.value
    token = str(value) if question.type == "score" else value
    if not isinstance(token, str) or token not in tokens:
        raise DevelopmentBaselineError("hard target does not name a legal answer token")
    return tokens.index(token)


def ensure_supported_model_cohort(model_id: str, cohort: str) -> None:
    if model_id not in MODEL_ARCHITECTURES:
        raise UnsupportedDevelopmentBaseline(
            f"unsupported model for development baselines: {model_id}"
        )
    if model_id.startswith("onejev-") and cohort in {"audio", "crossmodal"}:
        raise UnsupportedDevelopmentBaseline(
            "OneJev audio and audio-bearing crossmodal baselines are unsupported: "
            "the local Qwen metadata is inconsistent for audio processor inputs"
        )


def _media_content_type(media_type: str) -> str:
    if media_type == "image":
        return "image"
    if media_type == "audio":
        return "audio"
    if media_type == "video":
        return "video"
    raise DevelopmentBaselineError(f"unsupported media type: {media_type}")


def build_messages_and_media(
    sample: Sample,
    question: Question,
    media_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    payload = render_payload(Prefix(state=sample.prefix.state), question)
    content: list[dict[str, Any]] = []
    images: list[Any] = []
    audios: list[Any] = []
    videos: list[Any] = []
    video_metadata: list[dict[str, Any]] = []
    media_summary: list[dict[str, Any]] = []
    canonical_summary: list[dict[str, Any]] = []
    total_pixels = 0
    for media in sample.prefix.media:
        path = resolve_media(media_root, media.uri, media.sha256, max_bytes=MAX_MEDIA_BYTES)
        if media.type == "image":
            image_module = importlib.import_module("PIL.Image")
            with image_module.open(path) as image:
                total_pixels += int(image.width) * int(image.height)
            if total_pixels > MAX_IMAGE_PIXELS:
                raise DevelopmentBaselineError("total image pixel budget exceeded")
        decoded = load_media(
            path,
            media.type,
            num_frames=VIDEO_FRAMES,
            max_seconds=MAX_AUDIO_SECONDS if media.type == "audio" else MAX_VIDEO_SECONDS,
        )
        content.append({"type": _media_content_type(media.type)})
        if media.type == "image":
            images.append(decoded)
        elif media.type == "audio":
            canonical = canonicalize_media("audio", decoded)
            decoded = canonical.hf_value
            audios.append(decoded)
            canonical_summary.append(canonical.summary)
        else:
            videos.append(decoded["frames"])
            video_metadata.append(decoded["metadata"])
        media_summary.append({"type": media.type, "uri": media.uri, "sha256": media.sha256})
    content.append({"type": "text", "text": payload})
    kwargs: dict[str, Any] = {
        "text": None,
        "return_tensors": "pt",
        "add_special_tokens": False,
        "truncation": False,
    }
    if images:
        kwargs["images"] = [images]
    if audios:
        kwargs["audio"] = audios
        kwargs["sampling_rate"] = 16000
    if videos:
        kwargs["videos"] = videos
        kwargs["video_metadata"] = video_metadata
        kwargs["do_sample_frames"] = False
    return (
        [{"role": "user", "content": content}],
        kwargs,
        {
            "payload": payload,
            "media": media_summary,
            "canonical_media": canonical_summary,
            "video_metadata": video_metadata,
        },
    )


def summarize_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "detach"):
            summary[key] = {"type": type(value).__name__}
            continue
        array = value.detach().cpu().numpy()
        if array.dtype.kind == "f":
            np = importlib.import_module("numpy")
            if not np.isfinite(array).all():
                raise DevelopmentBaselineError(f"processor returned nonfinite tensor: {key}")
        summary[key] = {
            "shape": list(array.shape),
            "dtype": str(value.dtype),
            "sha256": sha256_bytes(array.tobytes()),
        }
    return summary


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def prepare_sample(sample: Sample, processor: Processor, media_root: Path) -> PreparedSample:
    started = time.perf_counter()
    question = sample.questions[0]
    messages, kwargs, identity = build_messages_and_media(sample, question, media_root)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise DevelopmentBaselineError("official chat template returned an empty prompt")
    tokenizer = processor.tokenizer
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        raise DevelopmentBaselineError("official prompt encoded to no tokens")
    answer_token_ids = check_answer_tokens(tokenizer, question)
    for answer, token_id in answer_token_ids.items():
        if tokenizer.encode(prompt + answer, add_special_tokens=False) != [*prompt_ids, token_id]:
            raise DevelopmentBaselineError("answer token changes at generation boundary")
    kwargs["text"] = [prompt]
    inputs = processor(**kwargs)
    if "input_ids" not in inputs:
        raise DevelopmentBaselineError("official processor omitted input_ids")
    final_ids = inputs["input_ids"][0].tolist()
    if len(final_ids) > MAX_PROMPT_TOKENS:
        raise DevelopmentBaselineError(
            f"item_id={sample.id} prompt token budget exceeded: "
            f"{len(final_ids)} > {MAX_PROMPT_TOKENS}"
        )
    suffix_size = min(8, len(prompt_ids))
    if suffix_size == 0 or final_ids[-suffix_size:] != prompt_ids[-suffix_size:]:
        raise DevelopmentBaselineError("processor changed generation boundary suffix")
    payload = str(identity["payload"])
    prompt_sha = sha256_bytes(prompt.encode())
    payload_sha = sha256_bytes(payload.encode())
    input_identity = {
        "sample_id": sample.id,
        "family_id": sample.meta.family_id,
        "source_sha256": sample.meta.source_sha256,
        "prompt_sha256": prompt_sha,
        "media": identity["media"],
        "answer_token_ids": answer_token_ids,
    }
    return PreparedSample(
        sample=sample,
        question=question,
        prompt=prompt,
        inputs=inputs,
        answer_token_ids=answer_token_ids,
        target_index=target_index(question),
        prompt_sha256=prompt_sha,
        payload_sha256=payload_sha,
        input_sha256=sha256_json(input_identity),
        prepare_wall_seconds=round(time.perf_counter() - started, 6),
        prompt_tokens=len(final_ids),
        input_summary={
            "processor_kwargs": {
                key: ("<prompt>" if key == "text" else value)
                for key, value in kwargs.items()
                if key
                in {
                    "return_tensors",
                    "add_special_tokens",
                    "truncation",
                    "sampling_rate",
                    "do_sample_frames",
                }
            },
            "media": identity["media"],
            "canonical_media": identity["canonical_media"],
            "video_metadata": identity["video_metadata"],
            "tensors": summarize_inputs(inputs),
        },
    )


def move_inputs_to_cuda(inputs: Mapping[str, Any], torch: Any, dtype: Any) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if getattr(value, "is_floating_point", lambda: False)():
                moved[key] = value.to(device="cuda", dtype=dtype)
            else:
                moved[key] = value.to(device="cuda")
        else:
            moved[key] = value
    return moved


def default_processor_factory(directory: Path) -> Processor:
    transformers = importlib.import_module("transformers")
    return cast(
        Processor,
        transformers.AutoProcessor.from_pretrained(
            str(directory),
            local_files_only=True,
            trust_remote_code=False,
        ),
    )


def ensure_cuda_available() -> Any:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise DevelopmentBaselineError("cuda_required_before_model_load")
    return torch


def real_forward_factory(
    model_dir_path: Path,
    model_id: str,
    dtype_name: str,
) -> tuple[ForwardFn, Callable[[], None]]:
    torch = ensure_cuda_available()
    configure_torch_runtime(torch)
    dtype = torch_dtype(torch, dtype_name)
    transformers = importlib.import_module("transformers")
    class_name = MODEL_ARCHITECTURES[model_id]["transformers_class"]
    if not hasattr(transformers, class_name):
        raise DevelopmentBaselineError(f"installed Transformers lacks {class_name}")
    model_cls = getattr(transformers, class_name)
    model = model_cls.from_pretrained(
        str(model_dir_path),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.eval()
    model.to("cuda")

    def forward(prepared: PreparedSample) -> Mapping[int, float]:
        inputs = move_inputs_to_cuda(prepared.inputs, torch, dtype)
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            logits = output.logits[0, -1].float()
            if not bool(torch.isfinite(logits).all().item()):
                raise DevelopmentBaselineError(f"nonfinite logits for {prepared.sample.id}")
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            return {
                int(token_id): float(logprobs[int(token_id)].item())
                for token_id in prepared.answer_token_ids.values()
            }

    def cleanup() -> None:
        nonlocal model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return forward, cleanup


def readout_from_logprobs(
    prepared: PreparedSample, logprobs_by_id: Mapping[int, float]
) -> dict[str, Any]:
    tokens = (*answer_tokens_for(prepared.question), "U")
    by_token: dict[str, float] = {}
    for token in tokens:
        token_id = prepared.answer_token_ids[token]
        if token_id not in logprobs_by_id:
            raise DevelopmentBaselineError(f"missing logprob for {prepared.sample.id}:{token}")
        value = float(logprobs_by_id[token_id])
        if not math.isfinite(value) or value > 0.0:
            raise DevelopmentBaselineError(f"inadmissible logprob for {prepared.sample.id}:{token}")
        by_token[token] = value
    readout = read_probabilities(by_token, answer_tokens_for(prepared.question))
    return {
        "raw_logprobs": by_token,
        "joint_k_plus_u": readout.joint_probabilities,
        "conditional_without_u": readout.probabilities,
        "p_U": readout.p_insufficient,
        "prediction": readout.prediction,
    }


def make_record(prepared: PreparedSample, *, pass_index: int, forward: ForwardFn) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        raw = forward(prepared)
        forward_wall = round(time.perf_counter() - started, 6)
        readout = readout_from_logprobs(prepared, raw)
    except Exception as exc:
        raise DevelopmentBaselineError(
            f"item_id={prepared.sample.id} phase=forward pass={pass_index}: {exc}"
        ) from exc
    return {
        "pass_index": pass_index,
        "id": prepared.sample.id,
        "family_id": prepared.sample.meta.family_id,
        "question_id": prepared.question.qid,
        "question_type": prepared.question.type,
        "target_index": prepared.target_index,
        "target_source": cast(HardTarget, prepared.question.target).source,
        "modalities": prepared.sample.meta.modality,
        "prompt_tokens": prepared.prompt_tokens,
        "prepare_wall_seconds": prepared.prepare_wall_seconds,
        "forward_wall_seconds": forward_wall,
        "prompt_sha256": prepared.prompt_sha256,
        "payload_sha256": prepared.payload_sha256,
        "input_sha256": prepared.input_sha256,
        "input_summary_sha256": sha256_json(prepared.input_summary),
        "answer_token_ids": prepared.answer_token_ids,
        "input_summary": prepared.input_summary,
        **readout,
    }


def append_record_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def sidecar_evidence(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return {"path": str(path), "exists": False, "records": 0, "sha256": None}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"path": str(path), "exists": True, "records": len(lines), "sha256": sha256_file(path)}


def run_pass(
    samples: Sequence[Sample],
    *,
    processor: Processor,
    media_root: Path,
    pass_index: int,
    forward: ForwardFn,
    records_sidecar: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        try:
            item = prepare_sample(sample, processor, media_root)
        except Exception as exc:
            raise DevelopmentBaselineError(
                f"item_id={sample.id} phase=prepare pass={pass_index}: {exc}"
            ) from exc
        record = make_record(item, pass_index=pass_index, forward=forward)
        if records_sidecar is not None:
            append_record_jsonl(records_sidecar, record)
        records.append(record)
    return records


def compare_repeatability(
    a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    if len(a) != len(b):
        mismatches.append({"kind": "length", "a": len(a), "b": len(b)})
    for left, right in zip(a, b, strict=False):
        if left.get("id") != right.get("id"):
            mismatches.append({"kind": "id", "a": left.get("id"), "b": right.get("id")})
            continue
        for key in (
            "input_sha256",
            "input_summary_sha256",
            "raw_logprobs",
            "conditional_without_u",
            "p_U",
            "prediction",
        ):
            if left.get(key) != right.get(key):
                mismatches.append({"kind": key, "id": left.get("id")})
                break
    return {
        "passed": not mismatches,
        "records_compared": min(len(a), len(b)),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
    }


def score_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities = [
        [
            float(record["conditional_without_u"][token])
            for token in answer_tokens_for_record(record)
        ]
        for record in records
    ]
    targets = [int(record["target_index"]) for record in records]
    return evaluate(probabilities, targets)


def answer_tokens_for_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    mapping = record["answer_token_ids"]
    if not isinstance(mapping, Mapping):
        raise DevelopmentBaselineError("record lacks answer token mapping")
    return tuple(token for token in mapping if token != "U")


def ensure_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise DevelopmentBaselineError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def run_development_report(
    *,
    build_report: Path,
    dev_jsonl: Path,
    media_root: Path,
    cohort: str,
    asset_manifest: Path,
    model_root: Path,
    model_id: str,
    dtype: str = DEFAULT_DTYPE,
    processor_factory: ProcessorFactory = default_processor_factory,
    forward_factory: ForwardFactory | None = None,
    fake_forward: bool = False,
    records_sidecar: Path | None = None,
) -> dict[str, Any]:
    if dtype not in ALLOWED_DTYPES:
        raise DevelopmentBaselineError(f"unsupported dtype: {dtype}")
    ensure_supported_model_cohort(model_id, cohort)
    build = bind_build_report(build_report, dev_jsonl)
    samples = load_samples(dev_jsonl)
    dataset = validate_samples(samples, cohort=cohort)
    model, directory = model_evidence(asset_manifest, model_root, model_id, require_weights=True)
    processor = processor_factory(directory)
    factory = forward_factory or (
        lambda path, mid, dtype_name: real_forward_factory(path, mid, dtype_name)
    )
    forward, cleanup = factory(directory, model_id, dtype)
    try:
        records_a = run_pass(
            samples,
            processor=processor,
            media_root=media_root,
            pass_index=1,
            forward=forward,
            records_sidecar=records_sidecar,
        )
        records_b = run_pass(
            samples,
            processor=processor,
            media_root=media_root,
            pass_index=2,
            forward=forward,
            records_sidecar=records_sidecar,
        )
    finally:
        cleanup()
    repeat = compare_repeatability(records_a, records_b)
    metrics_a = score_records(records_a)
    metrics_b = score_records(records_b)
    passed = bool(repeat["passed"] and len(records_a) == EXPECTED_ROWS)
    p_u_values = [float(record["p_U"]) for record in records_a]
    return {
        "schema": "lite-omnijev.s0_development_baselines.report.v1",
        "version": BASELINE_VERSION,
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 2,
        "acceptance": {
            "real_cuda_forward": not fake_forward,
            "fake_forward_control_only": fake_forward,
            "release_or_stage_acceptance": bool(passed and not fake_forward),
            "full_denominator_500": len(records_a) == EXPECTED_ROWS,
            "two_pass_exact_raw_conditional_pu_prediction": repeat["passed"],
            "weights_verified_before_load": True,
            "trust_remote_code": False,
            "local_files_only": True,
            "use_safetensors": True,
            "official_chat_template_enable_thinking_false": True,
            "truncation": False,
        },
        "cohort": cohort,
        "build_report": build,
        "dev_jsonl": {"path": str(dev_jsonl), "sha256": build["bound_dev_jsonl_sha256"]},
        "media_root": str(media_root),
        "dataset": dataset,
        "model": model,
        "precision": {"dtype": dtype, "seed": 42, "tf32_allowed": False},
        "metrics_pass_1": metrics_a,
        "metrics_pass_2": metrics_b,
        "repeatability": repeat,
        "p_U_summary": {
            "min": min(p_u_values) if p_u_values else None,
            "max": max(p_u_values) if p_u_values else None,
            "mean": math.fsum(p_u_values) / len(p_u_values) if p_u_values else None,
        },
        "resume": {
            "supported": False,
            "records_sidecar": sidecar_evidence(records_sidecar),
            "records_saved": len(records_a),
        },
        "records": records_a,
        "records_pass_2_sha256": sha256_json(records_b),
        "records_sidecar": sidecar_evidence(records_sidecar),
    }


def command_report(args: argparse.Namespace) -> int:
    ensure_output(args.report)
    records_sidecar = args.records_sidecar or args.report.with_name(
        args.report.name + ".records.jsonl"
    )
    ensure_output(records_sidecar)
    started = time.perf_counter()
    started_at = now_iso()
    try:
        body = run_development_report(
            build_report=args.build_report,
            dev_jsonl=args.dev_jsonl,
            media_root=args.media_root,
            cohort=args.cohort,
            asset_manifest=args.asset_manifest,
            model_root=args.model_root,
            model_id=args.model_id,
            dtype=args.dtype,
            records_sidecar=records_sidecar,
        )
    except UnsupportedDevelopmentBaseline as exc:
        report = report_base("run", started_at, started)
        report.update(
            {
                "schema": "lite-omnijev.s0_development_baselines.report.v1",
                "version": BASELINE_VERSION,
                "status": "unsupported",
                "exit_code": 0,
                "cohort": args.cohort,
                "model_id": args.model_id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "acceptance": {"release_or_stage_acceptance": False},
                "records_sidecar": sidecar_evidence(records_sidecar),
            }
        )
        write_json(args.report, report)
        return 0
    except Exception as exc:
        report = report_base("run", started_at, started)
        report.update(
            {
                "schema": "lite-omnijev.s0_development_baselines.report.v1",
                "version": BASELINE_VERSION,
                "status": "blocked",
                "exit_code": 2,
                "cohort": args.cohort,
                "model_id": args.model_id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "acceptance": {"release_or_stage_acceptance": False},
                "records_sidecar": sidecar_evidence(records_sidecar),
            }
        )
        write_json(args.report, report)
        return 2
    report = report_base("run", started_at, started)
    report["source"]["code_hashes"] = _code_hash_subset()
    report.update(body)
    report["command"] = {"argv": sys.argv, "cwd": str(Path.cwd())}
    write_json(args.report, report)
    return int(report["exit_code"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--cohort", choices=SUPPORTED_COHORTS, required=True)
    parser.add_argument(
        "--asset-manifest", type=Path, default=Path("configs/project-model-assets.json")
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dtype", choices=sorted(ALLOWED_DTYPES), default=DEFAULT_DTYPE)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--records-sidecar", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return command_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
