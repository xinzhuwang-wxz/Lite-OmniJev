"""Run Valen-Preview-0923 on JevBench231 with the safe local Qwen adapter path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from data.jevbench import (
    EXPECTED_PUBLIC_DECISIONS,
    JevBenchTask,
    build_input_question,
    deterministic_state_text,
    load_public_tasks,
    project_distribution_to_source,
    public_protocol_summary,
)
from data.schema import Question, answer_tokens
from eval.jevbench import score_predictions
from scripts.assets import write_json
from scripts.s0_public_baselines import (
    REPO_ROOT,
    ensure_output,
    environment_summary,
    load_asset_manifest,
    model_by_id,
    model_dir,
    report_base,
    sha256_file,
    sha256_json,
    sidecar_evidence,
    source_identity,
    verify_required_metadata,
    verify_required_weights,
)
from serve.gateway.app import core_question
from train.valen_inference import (
    MAX_LENGTH,
    Processor,
    ValenInferenceError,
    compile_qwen_question,
    load_valen_runtime,
    softmax,
    verify_valen_evidence,
)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValenInferenceError(f"JSON root must be an object: {path}")
    return data


def verify_qwen_base_assets(
    *, asset_manifest: Path, base_model_dir: Path, model_root: Path | None = None
) -> dict[str, Any]:
    manifest = load_asset_manifest(asset_manifest)
    models = model_by_id(manifest)
    model_id = "qwen3.5-2b-for-valen"
    if model_id not in models:
        raise ValenInferenceError("asset manifest missing qwen3.5-2b-for-valen")
    model = models[model_id]
    root = model_root if model_root is not None else base_model_dir.parents[1]
    expected_dir = model_dir(root, model)
    if base_model_dir.resolve() != expected_dir.resolve():
        raise ValenInferenceError("base model dir does not match pinned Qwen3.5-2B manifest path")
    metadata_hashes = verify_required_metadata(base_model_dir, model)
    weight_hashes = verify_required_weights(base_model_dir, model)
    config = load_json(base_model_dir / "config.json")
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != [
        "Qwen3_5ForConditionalGeneration"
    ]:
        raise ValenInferenceError("unsupported Qwen base architecture")
    if config.get("auto_map") not in (None, {}):
        raise ValenInferenceError("custom-code auto_map is not allowed for Valen base")
    if model.get("revision") != "15852e8c16360a2fea060d615a32b45270f8a8fc":
        raise ValenInferenceError("unexpected Qwen base revision")
    return {
        "id": model_id,
        "repo_id": model.get("repo_id"),
        "revision": model.get("revision"),
        "model_dir": str(base_model_dir),
        "asset_manifest": {"path": str(asset_manifest), "sha256": sha256_file(asset_manifest)},
        "metadata_hashes": metadata_hashes,
        "weight_hashes": weight_hashes,
        "architecture": {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "auto_map_absent": True,
        },
    }


def load_public_source(source_root: Path, source_config: Path) -> tuple[JevBenchTask, ...]:
    config = load_json(source_config)
    tasks = load_public_tasks(source_root, config)
    if len(tasks) != EXPECTED_PUBLIC_DECISIONS:
        raise ValenInferenceError("JevBench public denominator changed")
    return tasks


def make_public_question(task: JevBenchTask) -> Question:
    question, labels = core_question(task.id, build_input_question(task))
    if task.primitive != "noul" and tuple(labels) != tuple(answer_tokens(question)):
        raise ValenInferenceError("gateway labels changed")
    return question


def valen_probabilities(
    task: JevBenchTask, question: Question, logits: Sequence[float]
) -> dict[str, Any]:
    probs = softmax(logits)
    if question.type == "noul":
        if len(probs) != 2:
            raise ValenInferenceError("noul expects true/false logits")
        project = {"yes": probs[0], "no": probs[1]}
    else:
        tokens = answer_tokens(question)
        if len(probs) != len(tokens):
            raise ValenInferenceError("logit count does not match answer count")
        project = dict(zip(tokens, probs, strict=True))
    source = project_distribution_to_source(task, project)
    prediction = max(sorted(source), key=lambda label: source[label])
    return {
        "raw_logits": list(float(value) for value in logits),
        "conditional_without_u_project": project,
        "source_probabilities": source,
        "prediction_source": prediction,
        "p_U": None,
        "u_supported": False,
    }


def append_record_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def run_pass(
    *,
    tasks: Sequence[JevBenchTask],
    processor: Processor,
    runtime: Any,
    pass_index: int,
    sidecar: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        question = make_public_question(task)
        started = time.perf_counter()
        compiled = compile_qwen_question(
            processor=processor,
            state=deterministic_state_text(task.state),
            question=question,
            max_length=MAX_LENGTH,
        )
        logits = runtime.logits(compiled)
        wall = time.perf_counter() - started
        record = {
            "pass_index": pass_index,
            "id": task.id,
            "primitive": task.primitive,
            "source_labels": list(task.labels),
            "project_tokens": list(task.project_tokens),
            "prompt_tokens_logical": compiled.logical_tokens,
            "prompt_tokens_compute": compiled.compute_tokens,
            "branches": len(compiled.branches),
            "enable_thinking_false_used": compiled.enable_thinking_false_used,
            "forward_wall_seconds": round(wall, 6),
            **valen_probabilities(task, question, logits),
        }
        append_record_jsonl(sidecar, record)
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
        if left.get("raw_logits") != right.get("raw_logits"):
            mismatches.append({"kind": "raw_logits", "id": left.get("id")})
        if left.get("source_probabilities") != right.get("source_probabilities"):
            mismatches.append({"kind": "probabilities", "id": left.get("id")})
        if left.get("prediction_source") != right.get("prediction_source"):
            mismatches.append({"kind": "prediction", "id": left.get("id")})
    return {
        "passed": not mismatches,
        "records_compared": min(len(a), len(b)),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:10],
    }


def score_records(
    tasks: Sequence[JevBenchTask], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return score_predictions(
        tasks,
        [
            {"id": str(record["id"]), "probabilities": record["source_probabilities"]}
            for record in records
        ],
    )


def load_processor(model_dir: Path) -> Processor:
    transformers = __import__("transformers")
    return cast(
        Processor,
        transformers.AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        ),
    )


def valen_code_hashes() -> dict[str, str]:
    files = (
        "train/valen_inference.py",
        "scripts/s0_valen_public.py",
        "tests/test_valen_inference.py",
    )
    return {relative: sha256_file(REPO_ROOT / relative) for relative in files}


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_public_source(args.source_root, args.source_config)
    base_evidence = verify_qwen_base_assets(
        asset_manifest=args.asset_manifest,
        base_model_dir=args.base_model_dir,
        model_root=args.model_root,
    )
    evidence = verify_valen_evidence(
        spec_path=args.spec,
        conversion_report_path=args.conversion_report,
        weights_path=args.weights,
    )
    processor = load_processor(args.base_model_dir)
    runtime = load_valen_runtime(base_model_dir=args.base_model_dir, weights_path=args.weights)
    records_a = run_pass(
        tasks=tasks,
        processor=processor,
        runtime=runtime,
        pass_index=1,
        sidecar=args.records_sidecar,
    )
    records_b = run_pass(
        tasks=tasks,
        processor=processor,
        runtime=runtime,
        pass_index=2,
        sidecar=args.records_sidecar,
    )
    repeat = compare_repeatability(records_a, records_b)
    score_a = score_records(tasks, records_a)
    score_b = score_records(tasks, records_b)
    passed = bool(repeat["passed"] and len(records_a) == EXPECTED_PUBLIC_DECISIONS)
    return {
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 2,
        "acceptance": {
            "real_cuda_forward": True,
            "release_or_stage_acceptance": passed,
            "full_denominator": len(records_a) == EXPECTED_PUBLIC_DECISIONS,
            "two_pass_exact_probabilities_and_predictions": repeat["passed"],
            "u_supported": False,
            "local_files_only": True,
            "trust_remote_code": False,
            "raw_pickle_loaded": False,
            "no_holdout": True,
        },
        "model": {
            "id": "valen-preview-0923-qwen3.5-2b",
            "source_repo": "Valen-Team/Valen-Preview-0923",
            "source_revision": "30096aaabb2689614c83a79f64f38c865c2ce1c1",
            "base_repo": "Qwen/Qwen3.5-2B",
            "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "attention": "eager",
            "backbone_dtype": "bfloat16",
            "head_dtype": "float32",
        },
        "evidence": evidence.__dict__,
        "base_evidence": base_evidence,
        "valen_code_hashes": valen_code_hashes(),
        "jevbench": public_protocol_summary(tasks),
        "source_config": {
            "path": str(args.source_config),
            "sha256": sha256_file(args.source_config),
        },
        "source_root": str(args.source_root),
        "repeatability": repeat,
        "score_pass_1": score_a,
        "score_pass_2": score_b,
        "records": records_a,
        "records_pass_2_sha256": sha256_json(records_b),
        "records_sidecar": sidecar_evidence(args.records_sidecar),
        "cost": {"new_paid_resources": None, "gpu_started_by_this_cli": False},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-config", type=Path, default=Path("configs/jevbench-public-assets.json")
    )
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
        report = report_base("valen-public", started_at, started)
        report.update(
            {
                "status": "blocked",
                "exit_code": 2,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "environment": environment_summary(),
                "source": source_identity(),
                "records_sidecar": sidecar_evidence(args.records_sidecar),
                "acceptance": {"release_or_stage_acceptance": False},
            }
        )
        write_json(args.report, report)
        return 2
    report = report_base("valen-public", started_at, started)
    report.update(body)
    report["command"] = {"argv": sys.argv, "cwd": str(Path.cwd())}
    write_json(args.report, report)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
