"""Run Valen-Preview-0923 on one frozen 500-row development cohort."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from data.schema import HardTarget, Question, Sample, answer_tokens
from eval.metrics import evaluate
from scripts.assets import write_json
from scripts.s0_public_baselines import (
    SEED,
    TF32_ALLOWED,
    configure_torch_runtime,
    ensure_output,
    report_base,
    sha256_file,
    sha256_json,
)
from scripts.s0_valen_public import (
    load_processor,
    valen_code_hashes,
    verify_qwen_base_assets,
)
from train.multimodal import load_media, resolve_media
from train.valen_inference import (
    MAX_LENGTH,
    Processor,
    compile_qwen_question,
    load_valen_runtime,
    softmax,
    verify_valen_evidence,
)

EXPECTED_ROWS = 500
SUPPORTED_COHORTS = {"text", "image", "screenshot", "video"}
UNSUPPORTED_COHORTS = {"audio", "crossmodal"}
MAX_MEDIA_BYTES = 64 * 1024 * 1024
MAX_VIDEO_SECONDS = 60.0
VIDEO_FRAMES = 16
VALEN_MEDIA_KWARGS = {
    "images_kwargs": {"longest_edge": 1_048_576, "shortest_edge": 65_536},
}


class ValenDevelopmentError(RuntimeError):
    """Raised when Valen development baseline evidence is not admissible."""


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValenDevelopmentError(f"JSON root must be an object: {path}")
    return data


def bind_build_report(report_path: Path, dev_jsonl: Path) -> dict[str, Any]:
    report = load_json(report_path)
    if report.get("status") != "passed":
        raise ValenDevelopmentError("build report must be passed")
    outputs = report.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get("dev_jsonl_sha256"), str):
        raise ValenDevelopmentError("build report lacks outputs.dev_jsonl_sha256")
    actual = sha256_file(dev_jsonl)
    expected = str(outputs["dev_jsonl_sha256"])
    if actual != expected:
        raise ValenDevelopmentError("dev_jsonl_sha256 does not bind the provided dev JSONL")
    return {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "outputs_dev_jsonl_sha256": expected,
        "bound_dev_jsonl_sha256": actual,
    }


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                samples.append(Sample.model_validate_json(line))
            except ValueError as exc:
                raise ValenDevelopmentError(f"invalid sample JSON at line {line_number}") from exc
    return samples


def ensure_supported_cohort(cohort: str) -> None:
    if cohort in UNSUPPORTED_COHORTS:
        raise ValenDevelopmentError(
            f"Valen official Qwen compiler does not support development cohort: {cohort}"
        )
    if cohort not in SUPPORTED_COHORTS:
        raise ValenDevelopmentError(f"unknown development cohort: {cohort}")


def expected_media_types(cohort: str) -> list[str]:
    if cohort == "text":
        return []
    if cohort == "screenshot":
        return ["image"]
    if cohort in {"image", "video"}:
        return [cohort]
    raise ValenDevelopmentError(f"unsupported cohort: {cohort}")


def validate_samples(samples: Sequence[Sample], *, cohort: str) -> dict[str, Any]:
    ensure_supported_cohort(cohort)
    if len(samples) != EXPECTED_ROWS:
        raise ValenDevelopmentError(f"expected {EXPECTED_ROWS} rows, found {len(samples)}")
    ids: set[str] = set()
    families: set[str] = set()
    media_counts = {"image": 0, "video": 0}
    question_types: dict[str, int] = {}
    for sample in samples:
        if sample.id in ids:
            raise ValenDevelopmentError("duplicate sample id")
        ids.add(sample.id)
        if sample.meta.pool != "dev":
            raise ValenDevelopmentError("development baselines require pool=dev")
        if sample.meta.family_id in families:
            raise ValenDevelopmentError("development baselines require unique family ids")
        families.add(sample.meta.family_id)
        if len(sample.questions) != 1:
            raise ValenDevelopmentError("development rows must have one question")
        question = sample.questions[0]
        if not isinstance(question.target, HardTarget):
            raise ValenDevelopmentError("development rows require hard targets")
        got = [item.type for item in sample.prefix.media]
        if got != expected_media_types(cohort):
            raise ValenDevelopmentError("sample media does not match cohort")
        for media_type in got:
            media_counts[media_type] += 1
        question_types[question.type] = question_types.get(question.type, 0) + 1
    return {
        "rows": len(samples),
        "unique_ids": len(ids),
        "unique_families": len(families),
        "pool": "dev",
        "source_pool_hard": True,
        "question_types": dict(sorted(question_types.items())),
        "media_counts": media_counts,
    }


def target_index(question: Question) -> int:
    if not isinstance(question.target, HardTarget):
        raise ValenDevelopmentError("hard target required")
    tokens = answer_tokens(question)
    value = str(question.target.value) if question.type == "score" else question.target.value
    if not isinstance(value, str) or value not in tokens:
        raise ValenDevelopmentError("hard target does not name a legal answer")
    return tokens.index(value)


def build_state_messages(
    sample: Sample, *, cohort: str, media_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    content: list[dict[str, Any]] = []
    media_summary: list[dict[str, Any]] = []
    video_metadata: list[dict[str, Any]] = []
    for item in sample.prefix.media:
        path = resolve_media(media_root, item.uri, item.sha256, max_bytes=MAX_MEDIA_BYTES)
        decoded = load_media(
            path,
            item.type,
            num_frames=VIDEO_FRAMES,
            max_seconds=MAX_VIDEO_SECONDS,
        )
        if item.type == "image":
            content.append({"type": "image", "image": decoded})
        elif item.type == "video":
            content.append({"type": "video", "video": decoded["frames"]})
            video_metadata.append(decoded["metadata"])
        else:
            raise ValenDevelopmentError("audio media is unsupported by Valen dev runner")
        media_summary.append({"type": item.type, "uri": item.uri, "sha256": item.sha256})
    content.append({"type": "text", "text": sample.prefix.state})
    media_kwargs: dict[str, Any] = dict(VALEN_MEDIA_KWARGS)
    if video_metadata:
        media_kwargs["video_metadata"] = video_metadata
        media_kwargs["do_sample_frames"] = False
    return [{"role": "user", "content": content}], {
        "cohort": cohort,
        "media": media_summary,
        "video_metadata": video_metadata,
        "media_kwargs": media_kwargs,
    }


def tensor_summary(inputs: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "detach"):
            summary[key] = {"type": type(value).__name__}
            continue
        array = value.detach().cpu().numpy()
        if array.dtype.kind == "f":
            np = __import__("numpy")
            if not np.isfinite(array).all():
                raise ValenDevelopmentError(f"processor returned nonfinite tensor: {key}")
        summary[key] = {
            "shape": list(array.shape),
            "dtype": str(value.dtype),
            "sha256": sha256_bytes(array.tobytes()),
        }
    return summary


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def valen_probabilities(question: Question, logits: Sequence[float]) -> dict[str, Any]:
    probs = softmax(logits)
    if question.type == "noul":
        if len(probs) != 2:
            raise ValenDevelopmentError("noul expects true/false logits")
        project = {"yes": probs[0], "no": probs[1]}
        ordered = [project[token] for token in answer_tokens(question)]
    else:
        tokens = answer_tokens(question)
        if len(probs) != len(tokens):
            raise ValenDevelopmentError("logit count does not match answers")
        ordered = probs
        project = dict(zip(tokens, probs, strict=True))
    prediction = answer_tokens(question)[max(range(len(ordered)), key=lambda index: ordered[index])]
    return {
        "raw_logits": [float(value) for value in logits],
        "conditional_without_u": project,
        "probabilities_ordered": ordered,
        "prediction": prediction,
        "p_U": None,
        "u_supported": False,
    }


def position_summary(compiled: Any) -> list[dict[str, Any]]:
    return [
        {
            "keys": branch.keys,
            "candidate_positions": branch.candidate_positions,
            "decision_position": branch.decision_position,
        }
        for branch in compiled.branches
    ]


def configure_runtime() -> dict[str, Any]:
    torch = __import__("torch")
    configure_torch_runtime(torch)
    return {"seed": SEED, "tf32_allowed": TF32_ALLOWED}


def run_pass(
    *,
    samples: Sequence[Sample],
    cohort: str,
    processor: Processor,
    runtime: Any,
    media_root: Path,
    pass_index: int,
    sidecar: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        started = time.perf_counter()
        question = sample.questions[0]
        messages, media_identity = build_state_messages(
            sample, cohort=cohort, media_root=media_root
        )
        compiled = compile_qwen_question(
            processor=processor,
            state=messages,
            question=question,
            media_kwargs=cast(Mapping[str, Any], media_identity["media_kwargs"]),
            max_length=MAX_LENGTH,
        )
        input_summary = tensor_summary(compiled.branches[0].inputs)
        positions = position_summary(compiled)
        logits = runtime.logits(compiled)
        wall = round(time.perf_counter() - started, 6)
        readout = valen_probabilities(question, logits)
        record = {
            "pass_index": pass_index,
            "id": sample.id,
            "family_id": sample.meta.family_id,
            "question_id": question.qid,
            "question_type": question.type,
            "target_index": target_index(question),
            "modalities": sample.meta.modality,
            "prompt_tokens_logical": compiled.logical_tokens,
            "prompt_tokens_compute": compiled.compute_tokens,
            "branches": len(compiled.branches),
            "input_summary_sha256": sha256_json(input_summary),
            "input_summary": input_summary,
            "position_summary_sha256": sha256_json(positions),
            "position_summary": positions,
            "media": media_identity["media"],
            "video_metadata": media_identity["video_metadata"],
            "enable_thinking_false_used": compiled.enable_thinking_false_used,
            "forward_wall_seconds": wall,
            **readout,
        }
        append_record_jsonl(sidecar, record)
        records.append(record)
    return records


def append_record_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def sidecar_evidence(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"path": str(path), "records": len(lines), "sha256": sha256_file(path)}


def score_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return evaluate(
        [cast(Sequence[float], record["probabilities_ordered"]) for record in records],
        [int(record["target_index"]) for record in records],
    )


def compare_exact(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    if len(a) != len(b):
        mismatches.append({"kind": "length", "a": len(a), "b": len(b)})
    for left, right in zip(a, b, strict=False):
        if left.get("id") != right.get("id"):
            mismatches.append({"kind": "id", "a": left.get("id"), "b": right.get("id")})
            continue
        for key in (
            "raw_logits",
            "conditional_without_u",
            "input_summary_sha256",
            "position_summary_sha256",
            "prediction",
        ):
            if left.get(key) != right.get(key):
                mismatches.append({"kind": key, "id": left.get("id")})
    return {
        "passed": not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
    }


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    ensure_supported_cohort(args.cohort)
    samples = load_samples(args.dev_jsonl)
    validation = validate_samples(samples, cohort=args.cohort)
    build = bind_build_report(args.build_report, args.dev_jsonl)
    base_evidence = verify_qwen_base_assets(
        asset_manifest=args.asset_manifest,
        base_model_dir=args.base_model_dir,
        model_root=args.model_root,
    )
    valen_evidence = verify_valen_evidence(
        spec_path=args.spec,
        conversion_report_path=args.conversion_report,
        weights_path=args.weights,
    )
    runtime_config = configure_runtime()
    processor = load_processor(args.base_model_dir)
    runtime = load_valen_runtime(base_model_dir=args.base_model_dir, weights_path=args.weights)
    records_a = run_pass(
        samples=samples,
        cohort=args.cohort,
        processor=processor,
        runtime=runtime,
        media_root=args.media_root,
        pass_index=1,
        sidecar=args.records_sidecar,
    )
    records_b = run_pass(
        samples=samples,
        cohort=args.cohort,
        processor=processor,
        runtime=runtime,
        media_root=args.media_root,
        pass_index=2,
        sidecar=args.records_sidecar,
    )
    repeat = compare_exact(records_a, records_b)
    passed = repeat["passed"] and len(records_a) == EXPECTED_ROWS
    return {
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 2,
        "acceptance": {
            "real_cuda_forward": True,
            "release_or_stage_acceptance": bool(passed),
            "full_denominator": len(records_a) == EXPECTED_ROWS,
            "two_pass_exact_raw_logits_probabilities_inputs": repeat["passed"],
            "p_U": None,
            "u_supported": False,
            "no_truncation": True,
        },
        "cohort": args.cohort,
        "validation": validation,
        "build_report": build,
        "base_evidence": base_evidence,
        "valen_evidence": valen_evidence.__dict__,
        "runtime_config": runtime_config,
        "valen_code_hashes": valen_code_hashes(),
        "score_pass_1": score_records(records_a),
        "score_pass_2": score_records(records_b),
        "repeatability": repeat,
        "records": records_a,
        "records_pass_2_sha256": sha256_json(records_b),
        "records_sidecar": sidecar_evidence(args.records_sidecar),
        "cost": {"new_paid_resources": None, "gpu_started_by_this_cli": False},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--dev-jsonl", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--asset-manifest", type=Path, default=Path("configs/project-model-assets.json")
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("reports/valen-checkpoint-spec.json"))
    parser.add_argument(
        "--conversion-report",
        type=Path,
        default=Path("reports/valen-checkpoint-conversion-010.json"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--records-sidecar", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_output(args.report)
    if args.records_sidecar is None:
        args.records_sidecar = args.report.with_name(args.report.name + ".records.jsonl")
    ensure_output(args.records_sidecar)
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        body = run_report(args)
    except Exception as exc:
        report = report_base("valen-development", started_at, started)
        report.update(
            {
                "status": "blocked",
                "exit_code": 2,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "records_sidecar": {
                    "path": str(args.records_sidecar),
                    "exists": args.records_sidecar.exists(),
                },
                "acceptance": {"release_or_stage_acceptance": False},
            }
        )
        write_json(args.report, report)
        return 2
    report = report_base("valen-development", started_at, started)
    report.update(body)
    report["command"] = {"argv": sys.argv, "cwd": str(Path.cwd())}
    write_json(args.report, report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
