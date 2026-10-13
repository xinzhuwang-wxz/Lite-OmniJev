"""Convert the pinned Jev-Omni decision head to audited safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.assets import write_json
from train.teacher_head import (
    SeccompError,
    convert_from_argv,
    install_network_deny_seccomp,
    isolated_convert,
    parse_conversion_args,
    probe_isolation,
)


def _conversion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--expected-spec", type=Path)
    parser.add_argument("--source-repo", default="akhilaaa3/Jev-Omni")
    parser.add_argument("--source-revision", default="5addda86ddee081a68fb067477ea100c221b8917")
    return parser


def _conversion_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--source",
        str(args.source),
        "--output",
        str(args.output),
        "--expected-sha256",
        args.expected_sha256,
        "--source-repo",
        args.source_repo,
        "--source-revision",
        args.source_revision,
    ]
    if args.expected_size is not None:
        argv.extend(["--expected-size", str(args.expected_size)])
    if args.expected_spec is not None:
        argv.extend(["--expected-spec", str(args.expected_spec)])
    return argv


def _run_convert(args: argparse.Namespace) -> dict[str, Any]:
    argv = _conversion_argv(args)
    if args.isolate:
        completed = isolated_convert(argv)
        if completed.returncode != 0:
            return {
                "schema_version": 1,
                "status": "blocked",
                "exit_code": completed.returncode,
                "blocker": completed.stderr.strip().splitlines()[-1]
                if completed.stderr.strip()
                else "isolated conversion failed",
            }
        return dict(json.loads(completed.stdout))
    kwargs = parse_conversion_args(argv)
    report = convert_from_argv(argv)
    report["isolation"] = {
        "mode": "direct",
        "network_denied": False,
        "reason": "direct conversion requested",
    }
    report["expected_spec_provided"] = kwargs["expected_spec"] is not None
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", parents=[_conversion_parser()])
    convert.add_argument("--isolate", action="store_true")
    convert.add_argument("--report", type=Path, required=True)

    sub.add_parser("_child-convert", parents=[_conversion_parser()])

    probe = sub.add_parser("probe-isolation")
    probe.add_argument("--report", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "probe-isolation":
            report = probe_isolation()
            report["exit_code"] = 0 if report["status"] == "ready" else 1
            write_json(args.report, report)
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return int(report["exit_code"])
        if args.command == "_child-convert":
            install_network_deny_seccomp()
            report = convert_from_argv(_conversion_argv(args))
            report["isolation"] = {"mode": "seccomp", "network_denied": True}
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return 0
        report = _run_convert(args)
        report.setdefault("exit_code", 0 if report.get("status") == "converted" else 1)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "blocked",
            "exit_code": 1,
            "blocker": f"{type(exc).__name__}: {exc}"
            if isinstance(exc, (SeccompError, ValueError, OSError))
            else type(exc).__name__,
        }
    if args.command == "convert":
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
