"""Convert the pinned Jev-Omni decision head to audited safetensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.assets import write_json
from train.teacher_head import (
    IsolationError,
    SeccompError,
    child_chrooted_convert,
    convert_from_argv,
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
        return isolated_convert(argv)
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

    child = sub.add_parser("_child-convert", parents=[_conversion_parser()])
    child.add_argument("--jail-root", type=Path, required=True)
    child.add_argument("--host-canary", required=True)

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
            report = child_chrooted_convert(
                _conversion_argv(args),
                jail_root=args.jail_root,
                host_canary=args.host_canary,
            )
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return 0
        report = _run_convert(args)
        report.setdefault("exit_code", 0 if report.get("status") == "converted" else 1)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "blocked",
            "exit_code": 1,
            "blocker": "isolation_unavailable"
            if isinstance(exc, (IsolationError, SeccompError))
            else "conversion_failed"
            if isinstance(exc, (ValueError, OSError, ModuleNotFoundError))
            else type(exc).__name__,
        }
    if args.command == "convert":
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
