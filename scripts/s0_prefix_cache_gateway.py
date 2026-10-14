"""Measure the real in-process gateway with captured sequential native upstream evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from eval.prefix_cache import CacheProtocol, CacheServer, digest
from eval.prefix_cache_gateway import (
    CaptureTransport,
    GatewayProtocol,
    GatewayRun,
    assess_gateway,
    collect_gateway,
)


async def run_gateway(
    protocol: GatewayProtocol,
    server: CacheServer,
    model_dir: Path,
    media_root: Path,
) -> GatewayRun:
    # No model weights or forward: the gateway only needs the official CPU processor.
    processor = importlib.import_module("transformers").AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    gateway_module = importlib.import_module("serve.gateway.app")
    capture = CaptureTransport(
        httpx.AsyncHTTPTransport(trust_env=False),
        protocol.native,
        server,
        time.monotonic,
    )
    app = gateway_module.create_app(
        gateway_module.Settings(
            upstream_url=server.base_url,
            upstream_model=server.model,
            model_version=server.model_revision,
            model_dir=model_dir,
            raw_logprobs_confirmed=True,
            media_root=media_root,
        ),
        processor=processor,
        transport=capture,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://127.0.0.1",
            timeout=120,
        ) as gateway:
            async with httpx.AsyncClient(
                timeout=120,
                follow_redirects=False,
                trust_env=False,
            ) as control:
                return await collect_gateway(
                    protocol,
                    server,
                    gateway,
                    control,
                    capture,
                    allow_reset=True,
                    clock=time.monotonic,
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run", "compare"))
    parser.add_argument("--native-protocol", type=Path)
    parser.add_argument("--gateway-request", type=Path)
    parser.add_argument("--server-record", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--on", type=Path)
    parser.add_argument("--off", type=Path)
    parser.add_argument(
        "--assessment-policy",
        choices=("strict-sequential-v1", "original-product-cold-v2"),
        default="strict-sequential-v1",
    )
    parser.add_argument("--allow-reset", action="store_true", default=False)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("choose a new report")
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "blocked",
        "exit_code": 1,
        "action": args.action,
        "weights_loaded": False,
        "s0_acceptance": False,
        "cost_usd": None,
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
                raise ValueError("two complete gateway run reports required")
            on = GatewayRun.model_validate(json.loads(args.on.read_text())["run"])
            off = GatewayRun.model_validate(json.loads(args.off.read_text())["run"])
            report.update(assess_gateway(on, off, policy=args.assessment_policy))
        else:
            if any(
                value is None
                for value in (
                    args.native_protocol,
                    args.gateway_request,
                    args.server_record,
                    args.model_dir,
                    args.media_root,
                )
            ):
                raise ValueError("frozen native/public inputs, server and local processor required")
            native = CacheProtocol.model_validate_json(args.native_protocol.read_text())
            server = CacheServer.model_validate_json(args.server_record.read_text())
            gateway_source = Path(__file__).resolve().parents[1] / "serve/gateway/app.py"
            configuration = {
                "gateway_source_sha256": hashlib.sha256(gateway_source.read_bytes()).hexdigest(),
                "model_metadata": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in args.model_dir.iterdir()
                    if path.is_file() and path.suffix in {".json", ".jinja"}
                },
                "media_root_identity_sha256": digest(str(args.media_root.resolve())),
                "transport": "in-process-ASGI-canonical-chat",
            }
            protocol = GatewayProtocol(
                native=native,
                request=json.loads(args.gateway_request.read_text()),
                gateway_configuration_sha256=digest(configuration),
            )
            report.update(
                gateway_configuration=configuration,
                protocol_sha256=digest(native.model_dump()),
                gateway_request_sha256=digest(protocol.request),
                server_record_sha256=digest(server.model_dump()),
            )
            if args.action == "validate":
                report.update(status="validated-no-network", exit_code=0)
            else:
                if not args.allow_reset or not server.exclusive_local_test:
                    raise ValueError("explicit reset permission and exclusive test server required")
                if server.evidence_kind != "gpu":
                    raise ValueError("actual service GPU evidence record required for CLI run")
                result = asyncio.run(run_gateway(protocol, server, args.model_dir, args.media_root))
                report.update(status="collected", exit_code=0, run=result.model_dump())
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="blocked", exit_code=1, error_type=type(exc).__name__)
    report["duration_seconds"] = time.monotonic() - started
    with args.report.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "exit_code")}))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
