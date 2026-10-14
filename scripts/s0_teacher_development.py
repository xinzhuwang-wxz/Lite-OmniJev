"""Evaluate frozen dev cohorts with the admitted Jev-Omni decision head."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from data.schema import Sample, answer_tokens
from eval.metrics import evaluate
from scripts.s0_public_baselines import sha256_file, source_identity
from train.multimodal import canonicalize_media, load_media, resolve_media
from train.teacher_inference import (
    PreparedTeacherInput,
    build_teacher_processor_inputs,
    load_teacher_forward_bundle,
    official_teacher_prompt,
)


def normalized_head(options: list[str], probabilities: dict[str, float]) -> list[float]:
    if set(options) != set(probabilities):
        raise ValueError("teacher head changed candidate labels")
    values = [float(probabilities[option]) for option in options]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("teacher head returned invalid probabilities")
    total = math.fsum(values)
    if not math.isclose(total, 1, abs_tol=0.001):
        raise ValueError("teacher head probability sum outside declared numerical tolerance")
    return [value / total for value in values]


def prepare(
    sample: Sample, root: Path, processor: Any
) -> tuple[Any, Any, list[str], dict[str, Any]]:
    question = sample.questions[0]
    tokens = list(answer_tokens(question))
    labels = question.options or question.score_descriptions or tokens
    options = [f"{token}: {label}" for token, label in zip(tokens, labels, strict=True)]
    content: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {"media_sha256": [], "video_metadata": [], "audio": []}
    for media in sample.prefix.media:
        path = resolve_media(root, media.uri, media.sha256, max_bytes=64 * 1024**2)
        value = load_media(
            path, media.type, num_frames=16, max_seconds=30 if media.type == "audio" else 60
        )
        evidence["media_sha256"].append(media.sha256)
        if media.type == "video":
            content.extend({"type": "image", "image": frame} for frame in value["frames"])
            evidence["video_metadata"].append(value["metadata"])
        elif media.type == "audio":
            canonical = canonicalize_media("audio", value)
            content.append({"type": "audio", "audio": canonical.hf_value})
            evidence["audio"].append(canonical.summary)
        else:
            content.append({"type": "image", "image": value})
    content.append(
        {
            "type": "text",
            "text": official_teacher_prompt(sample.prefix.state, question.instruction, options),
        }
    )
    prepared = PreparedTeacherInput(
        "combined"
        if len(sample.prefix.media) > 1
        else (sample.prefix.media[0].type if sample.prefix.media else "text"),
        len(options),
        [{"role": "user", "content": content}],
        None,
    )
    built = build_teacher_processor_inputs(processor, prepared)
    evidence["processor"] = built.summary
    evidence["prompt_tokens"] = int(built.inputs["input_ids"].shape[-1])
    if evidence["prompt_tokens"] > 8192:
        raise ValueError("teacher input exceeds frozen8192-token budget; no truncation")
    return prepared, built.inputs, options, evidence


def load_cohort(entry: dict[str, Any]) -> list[Sample]:
    path = Path(entry["data"])
    report = json.loads(Path(entry["report"]).read_text())
    if (
        report.get("status") != "passed"
        or report.get("exit_code") != 0
        or report.get("outputs", {}).get("dev_jsonl_sha256") != sha256_file(path)
    ):
        raise ValueError("source build report does not bind the development data")
    samples = [Sample.model_validate_json(line) for line in path.read_text().splitlines()]
    if len(samples) != 500 or len({sample.id for sample in samples}) != 500:
        raise ValueError("development cohort must retain all 500 unique items")
    for sample in samples:
        if sample.meta.pool != "dev" or len(sample.questions) != 1:
            raise ValueError("only isolated single-question development samples allowed")
        if sample.questions[0].target is None or sample.questions[0].target.kind != "hard":
            raise ValueError("development baseline requires original/program hard truth")
    return samples


def run_cohort(entry: dict[str, Any], teacher: Any, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    samples = load_cohort(entry)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with (output / "records.jsonl").open("x") as stream:
        for pass_index in (1, 2):
            for sample in samples:
                begin = time.perf_counter()
                prepared, inputs, options, evidence = prepare(
                    sample, Path(entry["media_root"]), teacher.processor
                )
                prepared_at = time.perf_counter()
                result = teacher.predict_prepared(prepared, inputs, options)
                row = {
                    "id": sample.id,
                    "pass_index": pass_index,
                    "probabilities": normalized_head(options, result.probabilities),
                    "raw_head_probabilities": [result.probabilities[o] for o in options],
                    "p_U": None,
                    "prepare_seconds": prepared_at - begin,
                    "forward_seconds": time.perf_counter() - prepared_at,
                    **evidence,
                }
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
    first, second = rows[:500], rows[500:]
    exact = all(
        a["id"] == b["id"]
        and a["probabilities"] == b["probabilities"]
        and a["raw_head_probabilities"] == b["raw_head_probabilities"]
        for a, b in zip(first, second, strict=True)
    )
    targets = []
    for sample in samples:
        question = sample.questions[0]
        assert question.target is not None and question.target.kind == "hard"
        targets.append(list(answer_tokens(question)).index(str(question.target.value)))
    return {
        "status": "passed" if exact else "failed",
        "exit_code": 0 if exact else 2,
        "cohort": entry["name"],
        "metrics": evaluate([r["probabilities"] for r in first], targets),
        "repeat_exact": exact,
        "records": len(rows),
        "source": entry,
        "dev_sha256": sha256_file(Path(entry["data"])),
        "records_sha256": sha256_file(output / "records.jsonl"),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohorts", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--protocol-source-dir", type=Path, required=True)
    p.add_argument("--head", type=Path, required=True)
    p.add_argument("--head-conversion-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        p.error("refusing to overwrite evidence")
    args.output.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "blocked",
        "exit_code": 2,
        "cohorts": [],
        "source": source_identity(),
        "runner_sha256": sha256_file(Path(__file__)),
        "config_sha256": sha256_file(args.cohorts),
        "argv": sys.argv,
        "cost_usd": None,
        "p_U_supported": False,
        "no_holdout": True,
        "scope": "real development baseline only; no trained model acceptance",
    }
    started = time.perf_counter()
    try:
        torch = __import__("torch")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["gpu"] = torch.cuda.get_device_name(0)
        report["torch"] = torch.__version__
        entries = json.loads(args.cohorts.read_text())["cohorts"]
        names = [entry["name"] for entry in entries]
        if (
            not names
            or len(names) != len(set(names))
            or not set(names) <= {"text", "image", "screenshot", "audio", "video", "crossmodal"}
        ):
            raise ValueError("invalid/duplicate cohort names")
        with load_teacher_forward_bundle(
            model_dir=args.model_dir,
            protocol_source_dir=args.protocol_source_dir,
            head_path=args.head,
            conversion_report=args.head_conversion_report,
        ) as teacher:
            for entry in entries:
                result = run_cohort(entry, teacher, args.output / entry["name"])
                report["cohorts"].append(result)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                if result["exit_code"] != 0:
                    raise ValueError("development two-pass probabilities differ")
        report.update(status="passed", exit_code=0, real_cuda_forward=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
