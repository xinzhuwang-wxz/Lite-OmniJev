"""Gateway adapter timer/capture tests; synthetic transport, never real GPU speed."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from test_prefix_cache import Backend, FakeClock, protocol, server

from eval.prefix_cache_gateway import (
    CaptureTransport,
    GatewayProtocol,
    assess_gateway,
    collect_gateway,
    parse_server_timing,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def gateway_protocol() -> GatewayProtocol:
    return GatewayProtocol(
        native=protocol(),
        gateway_configuration_sha256="c" * 64,
        request={
            "state": "public image protocol",
            "media": [{"type": "image", "uri": "frame.png", "sha256": "d" * 64}],
            "questions": {
                f"q{i}": {
                    "type": "choice",
                    "instructions": f"tail{i}",
                    "criteria": {"A": "first", "B": "second"},
                }
                for i in range(10)
            },
        },
    )


async def collect(enabled: bool, defect: str | None = None) -> Any:
    frozen = gateway_protocol()
    record = server(enabled)
    clock = FakeClock()
    backend = Backend(enabled, clock)
    capture = CaptureTransport(httpx.MockTransport(backend), frozen.native, record, clock)

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        clock.now += 0.001
        before = len(capture.observations)
        answers = {}
        async with httpx.AsyncClient(transport=capture) as client:
            for key in body["questions"]:
                index = int(key[1:])
                response = await client.post(
                    record.base_url + "/v1/chat/completions",
                    json=frozen.native.payload(record.model, index),
                )
                response.raise_for_status()
                answers[key] = {"probabilities": {"A": 0.5, "B": 0.5}}
        rows = capture.observations[before:]
        upstream = sum(row.http_wall_seconds for row in rows) * 1000
        timing = f"preprocess;dur=1,upstream;dur={upstream},queue;dur=0,total;dur={upstream + 1}"
        if defect == "timing":
            timing = "preprocess;dur=1,total;dur=1"
        if defect == "usage":
            answers.clear()
        return httpx.Response(
            200,
            headers={"Server-Timing": timing},
            json={
                "answers": answers,
                "backend_usage": {"input_tokens": 3 * len(rows), "output_tokens": len(rows)},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1"
    ) as gateway:
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as control:
            return await collect_gateway(
                frozen, record, gateway, control, capture, allow_reset=True, clock=clock
            )


@pytest.mark.anyio
async def test_gateway_sequential_capture_and_actual_timer_fields() -> None:
    on, off = await collect(True), await collect(False)
    result = assess_gateway(on, off)
    assert result["status"] == "mock-validated"
    assert result["s0_acceptance"] is False
    assert len(result["timing_rows"]) == 16
    assert all(row["server_timing_ms"]["preprocess"] == 1 for row in result["timing_rows"])


@pytest.mark.anyio
@pytest.mark.parametrize("defect", ["timing", "usage"])
async def test_gateway_missing_measured_evidence_fails(defect: str) -> None:
    with pytest.raises(ValueError):
        await collect(True, defect)


@pytest.mark.parametrize(
    "header",
    [
        "",
        "preprocess;dur=nan,upstream;dur=1,queue;dur=0,total;dur=2",
        "preprocess;dur=1,upstream;dur=2,queue;dur=0,total;dur=1",
        "preprocess;dur=1,preprocess;dur=1,queue;dur=0,total;dur=2",
    ],
)
def test_timer_parser_refuses_missing_nonfinite_or_overlapping_intervals(header: str) -> None:
    with pytest.raises(ValueError):
        parse_server_timing(header)


@pytest.mark.anyio
async def test_capture_rejects_input_drift_and_concurrent_native_calls() -> None:
    frozen, record, clock = protocol(), server(True), FakeClock()
    capture = CaptureTransport(httpx.MockTransport(Backend(True, clock)), frozen, record, clock)
    payload = frozen.payload(record.model, 0)
    payload["top_k"] = 4
    async with httpx.AsyncClient(transport=capture) as client:
        with pytest.raises(ValueError, match="exclusive server record"):
            await client.post("http://example.com/v1/chat/completions", json=payload)
        with pytest.raises(ValueError, match="sampling/token"):
            await client.post(record.base_url + "/v1/chat/completions", json=payload)
        capture.inflight = True
        with pytest.raises(ValueError, match="concurrent upstream"):
            await client.post(
                record.base_url + "/v1/chat/completions", json=frozen.payload(record.model, 0)
            )


@pytest.mark.anyio
async def test_actual_official_cpu_processor_and_project_gateway(
    tmp_path: Path,
) -> None:
    """Opt-in actual CPU preparation/ASGI, with simulated backend probabilities and KV."""
    model_path = os.getenv("S0_LOCAL_MODEL_DIR")
    media_path = os.getenv("S0_CANONICAL_MEDIA_ROOT")
    if not model_path or not media_path:
        pytest.skip("explicit pinned official processor and public media required")
    started = time.monotonic()
    from transformers import AutoProcessor

    from eval.prefix_cache import digest
    from scripts.freeze_prefix_cache import freeze
    from scripts.s0_prepare import read_manifest
    from serve.gateway.app import Settings, create_app

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    native, public, summary = freeze(
        read_manifest(Path("tests/fixtures/s0-prepare-public-001/manifest.json")),
        processor,
        Path(media_path),
        repeats=2,
        warmup_requests=1,
    )
    frozen = GatewayProtocol(native=native, request=public, gateway_configuration_sha256="e" * 64)
    runs = []
    for enabled in (True, False):
        record = server(enabled)
        cached = False

        def backend(
            request: httpx.Request,
            record: Any = record,
            enabled: bool = enabled,
        ) -> httpx.Response:
            nonlocal cached
            if request.url.path == "/reset_prefix_cache":
                cached = False
                return httpx.Response(200, json={"success": True})
            if request.url.path in {"/reset_mm_cache", "/reset_encoder_cache"}:
                return httpx.Response(200)
            payload = json.loads(request.content)
            index = next(
                i
                for i in range(10)
                if payload["messages"] == native.payload(record.model, i)["messages"]
            )
            prompt_ids = native.cases[index].prompt_token_ids
            hit = len(native.shared_prompt_token_ids) if cached and enabled else 0
            cached = True
            tokens = list(native.answer_token_ids.values())
            return httpx.Response(
                200,
                json={
                    "model": record.model,
                    "prompt_token_ids": prompt_ids,
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": 1,
                        "total_tokens": len(prompt_ids) + 1,
                        "prompt_tokens_details": {"cached_tokens": hit},
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
                                        "token": f"token_id:{tokens[0]}",
                                        "logprob": -2.0,
                                        "top_logprobs": [
                                            {"token": f"token_id:{token}", "logprob": -float(i + 2)}
                                            for i, token in enumerate(tokens)
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                },
            )

        capture = CaptureTransport(httpx.MockTransport(backend), native, record, time.monotonic)
        app = create_app(
            Settings(
                upstream_url=record.base_url,
                upstream_model=record.model,
                model_version="actual-CPU-mock-backend",
                media_root=Path(media_path),
                raw_logprobs_confirmed=True,
            ),
            processor=processor,
            transport=capture,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
            ) as gateway:
                async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as control:
                    run = await collect_gateway(
                        frozen,
                        record,
                        gateway,
                        control,
                        capture,
                        allow_reset=True,
                        clock=time.monotonic,
                    )
                    runs.append(run)
        assert all(timing.preprocess > 0 for timing in run.timings)
        assert all(timing.upstream > 0 for timing in run.timings)
        assert len(run.measurements) == 8
    result = assess_gateway(runs[0], runs[1])
    assert result["evidence_kind"] == "mock" and result["s0_acceptance"] is False
    assert set(result["failures"]) <= {"ten_to_single_ratio_exceeds_two"}
    report_path = os.getenv("S0_CACHE_GATEWAY_CPU_REPORT")
    if report_path:
        with Path(report_path).open("x") as handle:
            handle.write(
                json.dumps(
                    {
                        "status": "cpu-integration-passed",
                        "exit_code": 0,
                        "scope": (
                            "actual official CPU processor and actual gateway ASGI; "
                            "mock native KV/logits"
                        ),
                        "gpu_requests": 0,
                        "weights_loaded": False,
                        "s0_acceptance": False,
                        "cost_usd": 0,
                        "environment": {
                            "python": platform.python_version(),
                            "platform": platform.platform(),
                        },
                        "code_revision": subprocess.check_output(
                            ["git", "rev-parse", "HEAD"],
                            cwd=Path(__file__).resolve().parents[1],
                            text=True,
                        ).strip(),
                        "implementation_sha256": {
                            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                            for name in (
                                "tests/test_prefix_cache_gateway.py",
                                "eval/prefix_cache_gateway.py",
                                "serve/gateway/app.py",
                                "train/multimodal.py",
                                "serve/vllm_chat.py",
                            )
                        },
                        "duration_seconds": time.monotonic() - started,
                        "protocol_sha256": digest(native.model_dump()),
                        "freeze_summary": summary,
                        "repeats": 2,
                        "measured_requests": 16,
                        "timing_rows": result["timing_rows"],
                        "latency_gate_evaluated": False,
                        "mock_timing_assessment": {
                            key: result[key] for key in ("status", "groups", "failures")
                        },
                        "note": "CPU mock timing ratios do not measure GPU cache performance",
                    },
                    indent=2,
                )
                + "\n"
            )
