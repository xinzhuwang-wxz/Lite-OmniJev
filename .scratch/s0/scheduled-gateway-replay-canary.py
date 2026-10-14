"""Official CPU processor and actual ASGI gateway; explicitly replayed model replies."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import subprocess
import time
import traceback
from pathlib import Path

import httpx

from eval.prefix_cache import CacheProtocol, CacheRun, CacheServer, digest
from eval.scheduled_cache_gateway import (
    ScheduledCaptureTransport,
    ScheduledGatewayProtocol,
    ScheduledGatewayRun,
    collect_scheduled_gateway,
)
from serve.gateway.app import Settings, create_app


async def run(args):
    from transformers import AutoProcessor

    native = CacheProtocol.model_validate_json(args.native_protocol.read_text())
    prior = CacheRun.model_validate(json.loads(args.reference.read_text())["run"])
    references = {row.case_index: row for m in prior.measurements for row in m.observations}
    assert prior.protocol_sha256 == digest(native.model_dump())
    server = CacheServer.model_validate_json(args.server_record.read_text()).model_copy(
        update={"evidence_kind": "mock", "run_id": "official-cpu-recorded-response-replay"}
    )
    request_body = json.loads(args.gateway_request.read_text())
    protocol = ScheduledGatewayProtocol(
        native=native,
        request=request_body,
        gateway_configuration_sha256=digest(
            {"mode": "cpu-replay-only", "model_dir": str(args.model_dir)}
        ),
    )
    seen = False
    calls = 0

    async def replay(request):
        nonlocal seen, calls
        if request.url.path == "/reset_prefix_cache":
            seen = False
            return httpx.Response(200, json={"success": True})
        if request.url.path in {"/reset_mm_cache", "/reset_encoder_cache"}:
            return httpx.Response(200)
        payload = json.loads(request.content)
        matches = [
            i
            for i in range(10)
            if payload["messages"] == native.payload(server.model, i)["messages"]
        ]
        assert len(matches) == 1
        index = matches[0]
        cached = 288 if seen else 0  # Synthetic cache bookkeeping, never GPU cache proof.
        seen = True
        calls += 1
        row = references[index]
        assert row.input_sha256 == digest(native.payload(server.model, index))
        values = row.raw_logprobs
        sampled = max(values, key=values.get)
        await asyncio.sleep(0.05)  # Exercise overlap; not a latency measurement.
        prompt = native.cases[index].prompt_token_ids
        return httpx.Response(
            200,
            json={
                "model": server.model,
                "prompt_token_ids": prompt,
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": 1,
                    "total_tokens": len(prompt) + 1,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
                "metrics": {
                    "time_to_first_token_ms": 0.0,
                    "generation_time_ms": 0.0,
                    "queue_time_ms": 0.0,
                },
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "logprobs": {
                            "content": [
                                {
                                    "token": f"token_id:{native.answer_token_ids[sampled]}",
                                    "logprob": values[sampled],
                                    "top_logprobs": [
                                        {
                                            "token": f"token_id:{native.answer_token_ids[label]}",
                                            "logprob": value,
                                        }
                                        for label, value in values.items()
                                    ],
                                }
                            ]
                        },
                    }
                ],
            },
        )

    processor = AutoProcessor.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    capture = ScheduledCaptureTransport(httpx.MockTransport(replay), native, server, time.monotonic)
    app = create_app(
        Settings(
            upstream_url=server.base_url,
            upstream_model=server.model,
            model_version=server.model_revision,
            model_dir=args.model_dir,
            media_root=args.media_root,
            raw_logprobs_confirmed=True,
            question_concurrency=10,
            request_image_reuse=True,
        ),
        processor=processor,
        transport=capture,
    )
    async with app.router.lifespan_context(app):
        async with (
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as gateway,
            httpx.AsyncClient(transport=httpx.MockTransport(replay)) as control,
        ):
            result = await collect_scheduled_gateway(
                protocol, server, gateway, control, capture, allow_reset=True, clock=time.monotonic
            )
    reloaded = ScheduledGatewayRun.model_validate_json(result.model_dump_json())
    assert reloaded.gateway_question_ids == list(request_body["questions"])
    assert reloaded.api_answer_digests == result.api_answer_digests
    return {
        "measurements": len(result.measurements),
        "captured_calls_including_warmup": calls,
        "measured_questions": sum(m.size for m in result.measurements),
        "maximum_replay_http_overlap": max(m.actual_max_inflight for m in result.measurements),
        "gateway_question_ids": result.gateway_question_ids,
        "run_json_sha256": hashlib.sha256(result.model_dump_json().encode()).hexdigest(),
        "api_answer_digests": [item.model_dump() for item in result.api_answer_digests],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "native-protocol",
        "gateway-request",
        "server-record",
        "reference",
        "model-dir",
        "media-root",
        "report",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("choose a new report path")
    started = time.monotonic()
    report = {
        "status": "failed",
        "exit_code": 1,
        "weights_loaded": False,
        "gpu_forward": False,
        "s0_acceptance": False,
        "performance_evidence": False,
        "model_transport": "synthetic protocol envelopes replaying prior GPU raw values",
        "cache_metrics": "synthetic; not GPU prefix evidence",
        "cost_usd": None,
        "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": {
            name: importlib.metadata.version(name) for name in ("transformers", "torch", "httpx")
        },
        "inputs_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                args.native_protocol,
                args.gateway_request,
                args.server_record,
                args.reference,
            )
        },
    }
    try:
        report.update(
            asyncio.run(run(args)), status="passed_official_cpu_asgi_replay_only", exit_code=0
        )
    except Exception as exc:
        report.update(
            error_type=type(exc).__name__,
            stack=[
                {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                for f in traceback.extract_tb(exc.__traceback__)
            ],
        )
    report["duration_seconds"] = time.monotonic() - started
    with args.report.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "duration_seconds")}))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
