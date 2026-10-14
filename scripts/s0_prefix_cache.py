"""Validate, collect one isolated server run, or compare sequential cache evidence."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from eval.prefix_cache import CacheProtocol, CacheRun, CacheServer, assess, collect, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run", "compare"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--server-record", type=Path)
    parser.add_argument("--on", type=Path)
    parser.add_argument("--off", type=Path)
    parser.add_argument("--allow-reset", action="store_true", default=False)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("choose a new report path")
    started = time.monotonic()
    report: dict[str, Any] = {
        "action": args.action,
        "status": "blocked",
        "exit_code": 1,
        "cost_usd": None,
        "s0_acceptance": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
    }
    try:
        if args.action == "compare":
            if args.on is None or args.off is None:
                raise ValueError("two completed evidence files required")
            on = CacheRun.model_validate(json.loads(args.on.read_text())["run"])
            off = CacheRun.model_validate(json.loads(args.off.read_text())["run"])
            report.update(assess(on, off))
            report["source_reports_sha256"] = {
                "on": digest(json.loads(args.on.read_text())),
                "off": digest(json.loads(args.off.read_text())),
            }
        else:
            if args.protocol is None or args.server_record is None:
                raise ValueError("frozen protocol and server record required")
            protocol = CacheProtocol.model_validate_json(args.protocol.read_text())
            server = CacheServer.model_validate_json(args.server_record.read_text())
            report.update(
                protocol_sha256=digest(protocol.model_dump()),
                server_record_sha256=digest(server.model_dump()),
            )
            if args.action == "validate":
                report.update(status="validated-no-network", exit_code=0)
            else:
                if server.evidence_kind != "gpu":
                    raise ValueError("CLI collection requires actual GPU service attestation")
                with httpx.Client(timeout=120, follow_redirects=False, trust_env=False) as client:
                    run = collect(
                        protocol, server, client, allow_reset=args.allow_reset, clock=time.monotonic
                    )
                report.update(status="collected", exit_code=0, run=run.model_dump())
    except (Exception, KeyboardInterrupt) as exc:
        # Exception strings can contain private payloads/URLs or validation input; omit them.
        report.update(status="blocked", exit_code=1, error_type=type(exc).__name__)
    report["duration_seconds"] = time.monotonic() - started
    with args.report.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("action", "status", "exit_code")}))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
