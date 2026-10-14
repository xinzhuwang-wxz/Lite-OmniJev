"""Measure the admitted Jev-Omni head on the fixed public JevBench231 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from data.jevbench import JevBenchTask, deterministic_state_text
from data.schema import answer_tokens
from eval.jevbench import score_predictions
from scripts.s0_public_baselines import load_public_source, make_question, source_identity
from train.teacher_inference import load_teacher_forward_bundle


def teacher_question(task: JevBenchTask) -> tuple[str, tuple[str, ...]]:
    question = make_question(task)
    tokens = answer_tokens(question)
    descriptions = question.options or question.score_descriptions or list(tokens)
    options = tuple(
        f"{task.project_to_source[token]}: {description}"
        for token, description in zip(tokens, descriptions, strict=True)
    )
    return question.instruction, options


def collect(tasks: tuple[JevBenchTask, ...], teacher: Any, stream: Any) -> list[dict[str, Any]]:
    records = []
    for pass_index in (1, 2):
        for task in tasks:
            question, options = teacher_question(task)
            started = time.perf_counter()
            result = teacher.predict(
                state=deterministic_state_text(task.state), question=question, options=options
            )
            probabilities = {
                task.project_to_source[token]: result.probabilities[option]
                for token, option in zip(task.project_tokens, options, strict=True)
            }
            record = {
                "id": task.id,
                "pass_index": pass_index,
                "probabilities": probabilities,
                "p_U": None,
                "p_U_reason": "official teacher head has only K choices; no abstention output",
                "forward_wall_seconds": time.perf_counter() - started,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-config", type=Path, default=Path("configs/jevbench-public-assets.json")
    )
    parser.add_argument("--protocol-source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--head-conversion-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    sidecar = Path(str(args.report) + ".records.jsonl")
    if any(p.exists() or p.is_symlink() for p in (args.report, sidecar)):
        parser.error("refusing to overwrite evidence")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "blocked",
        "exit_code": 2,
        "source": source_identity(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "cost_usd": None,
        "no_holdout": True,
        "teacher_195_protocol_matched": False,
        "whole_s0_complete": False,
        "p_U_supported": False,
        "scope": (
            "official teacher head, full-rubric public231, K-way probabilities; "
            "not decoder raw logprobs"
        ),
    }
    try:
        torch = __import__("torch")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["environment"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "seed": 42,
        }
        tasks = load_public_source(args.source_root, args.source_config)
        with (
            load_teacher_forward_bundle(
                protocol_source_dir=args.protocol_source_dir,
                model_dir=args.model_dir,
                head_path=args.head,
                conversion_report=args.head_conversion_report,
            ) as teacher,
            sidecar.open("x") as stream,
        ):
            records = collect(tasks, teacher, stream)
        first, second = records[:231], records[231:]
        exact = len(first) == len(second) == 231 and all(
            a["id"] == b["id"] and a["probabilities"] == b["probabilities"]
            for a, b in zip(first, second, strict=True)
        )
        report.update(
            status="passed" if exact else "failed",
            exit_code=0 if exact else 2,
            real_cuda_forward=True,
            records=len(records),
            repeat_exact=exact,
            score_pass_1=score_predictions(tasks, first),
            score_pass_2=score_predictions(tasks, second),
        )
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["records_sidecar"] = {
            "path": str(sidecar),
            "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()
            if sidecar.exists()
            else None,
            "rows": sum(1 for _ in sidecar.open()) if sidecar.exists() else 0,
        }
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
