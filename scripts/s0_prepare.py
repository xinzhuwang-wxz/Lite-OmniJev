"""Compile public development inputs with a local official processor, without weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.assets import write_json
from train.multimodal import PreparationCase, file_hash, prepare_case


def read_manifest(path: Path, *, allow_partial: bool = False) -> list[PreparationCase]:
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("manifest exceeds size limit")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw or len(raw) > 100:
        raise ValueError("manifest requires 1–100 cases")
    cases = [PreparationCase.model_validate(row) for row in raw]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    pools: dict[str, str] = {}
    for case in cases:
        if case.meta.family_id in pools and pools[case.meta.family_id] != case.meta.pool:
            raise ValueError("family crosses pools")
        pools[case.meta.family_id] = case.meta.pool
    if not allow_partial and (
        Counter(case.category for case in cases)
        != dict.fromkeys(["text", "image", "screenshot", "audio", "video"], 20)
        or {case.category for case in cases} != {"text", "image", "screenshot", "audio", "video"}
        or {case.question.type for case in cases} != {"choice", "score", "noul"}
    ):
        raise ValueError(
            "complete manifest requires 100 cases, five categories and three primitives"
        )
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    start = time.monotonic()
    report: dict[str, Any] = {
        "status": "pending",
        "exit_code": 1,
        "complete_100_cases": False,
        "pending_vllm_processor_and_transport": True,
        "scope": "CPU processor only; no weights, forward, GPU, labels or quality acceptance",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "cost": {"new_paid_resources": 0},
        "cases": [],
    }
    write_json(args.output, report)
    try:
        cases = read_manifest(args.manifest, allow_partial=args.allow_partial)
        report["manifest_sha256"] = file_hash(args.manifest)
        report["excluded_training_families"] = sorted({case.meta.family_id for case in cases})
        report["source_sha256"] = {
            str(path.relative_to(Path(__file__).parent.parent)): file_hash(path)
            for path in (Path(__file__), Path(__file__).parent.parent / "train/multimodal.py")
        }
        report["code_version"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        ).stdout.strip()
        report["model_files"] = {
            path.name: file_hash(path)
            for path in args.model_dir.iterdir()
            if path.is_file() and (path.suffix == ".json" or path.name == "chat_template.jinja")
        }
        report["phase"] = "loading_official_processor"
        write_json(args.output, report)
        transformers = importlib.import_module("transformers")
        report["environment"]["transformers"] = transformers.__version__
        report["environment"]["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "pillow", "numpy")
        }
        report.update(weights_loaded=False, gpu_used=False)
        processor = transformers.AutoProcessor.from_pretrained(
            args.model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        report["phase"] = "compiling_cases"
        write_json(args.output, report)
        for case in cases:
            report["active_case_id"] = case.id
            write_json(args.output, report)
            report["cases"].append(prepare_case(case, processor, args.asset_root).summary)
        report.pop("active_case_id", None)
        report["phase"] = "finished"
        report.update(
            status="partial" if args.allow_partial else "passed",
            exit_code=0,
            complete_100_cases=not args.allow_partial,
        )
    except Exception as exc:
        # Avoid copying source text, private paths or third-party exception payloads to evidence.
        report.update(status="blocked", error_type=type(exc).__name__)
    report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "complete_100_cases")}))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
