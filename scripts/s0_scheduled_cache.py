"""Validate, collect, or compare bounded scheduled native cache evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from eval.prefix_cache import CacheProtocol, CacheServer, digest
from eval.scheduled_cache import ScheduledCacheRun, assess_scheduled, collect_scheduled


def _code_hash(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


async def _run_collect(
    protocol: CacheProtocol,
    server: CacheServer,
    *,
    allow_reset: bool,
    concurrency: int,
) -> ScheduledCacheRun:
    async with httpx.AsyncClient(
        timeout=120,
        follow_redirects=False,
        trust_env=False,
    ) as client:
        return await collect_scheduled(
            protocol,
            server,
            client,
            allow_reset=allow_reset,
            concurrency_limit=concurrency,
            clock=time.monotonic,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run", "compare"))
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--server-record", type=Path)
    parser.add_argument("--on", type=Path)
    parser.add_argument("--off", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--allow-reset", action="store_true", default=False)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("choose a new report path")
    repo = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    report: dict[str, Any] = {
        "action": args.action,
        "status": "blocked",
        "exit_code": 1,
        "cost_usd": None,
        "s0_acceptance": False,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mode": "bounded-scheduled-v1",
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
        ).strip(),
        "code_sha256": _code_hash(
            [
                repo / "eval" / "scheduled_cache.py",
                repo / "scripts" / "s0_scheduled_cache.py",
            ]
        ),
    }
    try:
        if args.action == "compare":
            if args.on is None or args.off is None:
                raise ValueError("two completed scheduled evidence files required")
            on_data = json.loads(args.on.read_text())
            off_data = json.loads(args.off.read_text())
            on = ScheduledCacheRun.model_validate(on_data["run"])
            off = ScheduledCacheRun.model_validate(off_data["run"])
            report.update(assess_scheduled(on, off))
            report["source_reports_sha256"] = {
                "on": digest(on_data),
                "off": digest(off_data),
            }
            if args.reference is not None:
                report["reference_report_sha256"] = digest(json.loads(args.reference.read_text()))
        else:
            if args.protocol is None or args.server_record is None:
                raise ValueError("frozen protocol and server record required")
            protocol = CacheProtocol.model_validate_json(args.protocol.read_text())
            server = CacheServer.model_validate_json(args.server_record.read_text())
            report.update(
                protocol_sha256=digest(protocol.model_dump()),
                server_record_sha256=digest(server.model_dump()),
                concurrency_limit=args.concurrency,
            )
            if args.action == "validate":
                report.update(status="validated-no-network", exit_code=0)
            else:
                if server.evidence_kind != "gpu":
                    raise ValueError("CLI collection requires actual GPU service attestation")
                run = asyncio.run(
                    _run_collect(
                        protocol,
                        server,
                        allow_reset=args.allow_reset,
                        concurrency=args.concurrency,
                    )
                )
                report.update(status="collected", exit_code=0, run=run.model_dump())
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="blocked", exit_code=1, error_type=type(exc).__name__)
    report["duration_seconds"] = time.monotonic() - started
    with args.report.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("action", "status", "exit_code")}))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
