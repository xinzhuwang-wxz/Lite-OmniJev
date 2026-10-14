"""CPU-only JevBench public adapter validation, corpus build, and scoring CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from data.jevbench import (
    JevBenchProtocolError,
    corpus_row,
    load_public_tasks,
    public_protocol_summary,
    sha256_file,
)
from eval.jevbench import score_predictions
from scripts.assets import write_json

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCE_FILES = (
    "data/jevbench.py",
    "eval/jevbench.py",
    "scripts/s0_jevbench.py",
    "data/schema.py",
    "serve/gateway/app.py",
    "train/readout.py",
    "configs/jevbench-public-assets.json",
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def refuse_overwrite(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise JevBenchProtocolError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def git_revision() -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": run("status", "--short"),
    }


def code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise JevBenchProtocolError(f"required source file missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def run_metadata(args: argparse.Namespace, started: float, command: str) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    finished = datetime.now(UTC)
    return {
        "command": command,
        "argv": sys.argv[1:],
        "started_at": (finished - timedelta(seconds=elapsed)).isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_s": elapsed,
        "cwd": str(Path.cwd()),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "seed": SEED,
        "cost_usd": None,
        "gpu_used": False,
        "quality_gate": "not_run",
        "teacher_matched_protocol_verified": False,
        "source_root": str(args.source_root),
        "config": {
            "path": str(args.config),
            "sha256": sha256_file(args.config),
        },
        "source_identity": {
            "repo_root": str(REPO_ROOT),
            "git": git_revision(),
            "code_hashes": code_hashes(),
        },
    }


def cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    refuse_overwrite(args.output)
    config = load_json(args.config)
    tasks = load_public_tasks(args.source_root, config)
    report = {
        "schema_version": 1,
        "status": "passed",
        "exit_code": 0,
        "mode": "validate",
        "source_files": config["files"],
        "protocol": public_protocol_summary(tasks),
        "metadata": run_metadata(args, started, "validate"),
    }
    write_json(args.output, report)
    return report


def cmd_build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    refuse_overwrite(args.output_jsonl)
    refuse_overwrite(args.report)
    config = load_json(args.config)
    tasks = load_public_tasks(args.source_root, config)
    line_count = 0
    digest = hashlib.sha256()
    with args.output_jsonl.open("w", encoding="utf-8") as stream:
        for task in tasks:
            encoded = json.dumps(corpus_row(task), ensure_ascii=False, sort_keys=True) + "\n"
            stream.write(encoded)
            digest.update(encoded.encode("utf-8"))
            line_count += 1
    report = {
        "schema_version": 1,
        "status": "passed",
        "exit_code": 0,
        "mode": "build",
        "generated_corpus": {
            "path": str(args.output_jsonl),
            "usage": "research-eval",
            "training_admitted": False,
            "rows": line_count,
            "sha256": digest.hexdigest(),
        },
        "protocol": public_protocol_summary(tasks),
        "metadata": run_metadata(args, started, "build"),
    }
    write_json(args.report, report)
    return report


def cmd_score(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    refuse_overwrite(args.output)
    config = load_json(args.config)
    tasks = load_public_tasks(args.source_root, config)
    predictions = load_json(args.predictions)
    scored = score_predictions(tasks, predictions)
    report = {
        **scored,
        "exit_code": 0,
        "mode": "score",
        "prediction_file": {
            "path": str(args.predictions),
            "sha256": sha256_file(args.predictions),
        },
        "metadata": run_metadata(args, started, "score"),
    }
    write_json(args.output, report)
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/jevbench-public-assets.json",
        help="Locked public source asset receipt JSON.",
    )
    subcommands = root.add_subparsers(dest="command", required=True)
    for name in ("validate", "build", "score"):
        sub = subcommands.add_parser(name)
        sub.add_argument("--source-root", type=Path, required=True)
    validate = subcommands.choices["validate"]
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(func=cmd_validate)
    build = subcommands.choices["build"]
    build.add_argument("--output-jsonl", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.set_defaults(func=cmd_build)
    score = subcommands.choices["score"]
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(func=cmd_score)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"s0_jevbench: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
