"""Derive and convert the pinned Valen nested checkpoint safely."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.assets import write_json
from train.teacher_head import IsolationError, SeccompError
from train.valen_checkpoint import (
    CHECKPOINT_SHA256,
    child_chrooted_convert,
    derive_spec,
    isolated_convert,
)


def _conversion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--expected-spec", type=Path, required=True)
    parser.add_argument("--diagnostics-output", type=Path)
    return parser


def _conversion_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.output),
        "--expected-sha256",
        args.expected_sha256,
        "--expected-spec",
        str(args.expected_spec),
    ]
    if args.expected_size is not None:
        argv.extend(["--expected-size", str(args.expected_size)])
    if args.diagnostics_output is not None:
        argv.extend(["--diagnostics-output", str(args.diagnostics_output)])
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    spec = sub.add_parser("derive-spec")
    spec.add_argument("--checkpoint-dir", type=Path, required=True)
    spec.add_argument("--base-dir", type=Path, required=True)
    spec.add_argument("--output", type=Path, required=True)
    spec.add_argument("--report", type=Path, required=True)

    convert = sub.add_parser("convert", parents=[_conversion_parser()])
    convert.add_argument("--isolate", action="store_true")
    convert.add_argument("--report", type=Path, required=True)

    child = sub.add_parser("_child-convert", parents=[_conversion_parser()])
    child.add_argument("--jail-root", type=Path, required=True)
    child.add_argument("--host-canary", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "derive-spec":
            generated = derive_spec(args.checkpoint_dir, args.base_dir)
            write_json(args.output, generated)
            report = {
                "schema_version": 1,
                "status": "spec_derived",
                "exit_code": 0,
                "spec": str(args.output),
                "tensor_count": generated["tensor_count"],
                "tensor_payload_bytes": generated["tensor_payload_bytes"],
                "groups": generated["groups"],
            }
            write_json(args.report, report)
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == "_child-convert":
            report = child_chrooted_convert(
                _conversion_argv(args),
                jail_root=args.jail_root,
                host_canary=args.host_canary,
            )
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
            return 0
        if not args.isolate:
            report = {
                "schema_version": 1,
                "status": "blocked",
                "exit_code": 1,
                "blocker": "isolation_required",
            }
        else:
            report = isolated_convert(_conversion_argv(args))
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
