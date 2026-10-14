"""Scheduled gateway adapter tests use synthetic ASGI/upstream evidence only."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from eval.prefix_cache import CacheCase, CacheProtocol, CacheServer, digest
from eval.scheduled_cache_gateway import (
    ScheduledCaptureTransport,
    ScheduledGatewayProtocol,
    ScheduledGatewayRun,
    assess_scheduled_gateway,
    collect_scheduled_gateway,
    expected_gateway_answer,
    parse_scheduled_server_timing,
)
from serve.gateway.app import InputQuestion


def protocol() -> CacheProtocol:
    png = b"\x89PNG\r\n\x1a\nSCHEDULED_GATEWAY"
    return CacheProtocol(
        image_png_base64=base64.b64encode(png).decode(),
        image_sha256=hashlib.sha256(png).hexdigest(),
        shared_text="gateway shared:",
        shared_prompt_token_ids=[10, 20],
        media_prefix_end=2,
        answer_token_ids={"A": 1, "B": 2, "U": 3},
        cases=[CacheCase(tail=f" q{i}", prompt_token_ids=[10, 20, 100 + i]) for i in range(10)],
        repeats=2,
        warmup_requests=1,
        seed=42,
        output_tolerance=0.0,
    )


def server(enabled: bool, *, max_num_seqs: int = 10) -> CacheServer:
    return CacheServer(
        base_url="http://127.0.0.1:8002",
        run_id="gateway-on" if enabled else "gateway-off",
        model="pinned",
        model_revision="fixed",
        weights_manifest_sha256="a" * 64,
        dtype="bfloat16",
        gpu_identity="synthetic GPU",
        vllm_version="0.31.0",
        common_engine_config_sha256="b" * 64,
        launch_command=[
            "vllm",
            "serve",
            "--enable-prefix-caching" if enabled else "--no-enable-prefix-caching",
            "--max-num-seqs",
            str(max_num_seqs),
            "--logprobs-mode",
            "raw_logprobs",
            "--enable-prompt-tokens-details",
            "--enable-per-request-metrics",
        ],
        cache_enabled=enabled,
        exclusive_local_test=True,
        raw_logprobs_confirmed=True,
        evidence_kind="mock",
    )


def gateway_protocol() -> ScheduledGatewayProtocol:
    return ScheduledGatewayProtocol(
        native=protocol(),
        gateway_configuration_sha256="c" * 64,
        request={
            "state": "public image protocol",
            "media": [{"type": "image", "uri": "frame.png", "sha256": "d" * 64}],
            "questions": {
                f"q{i}": {
                    "type": "choice",
                    "instructions": f" q{i}",
                    "criteria": {"A": "first", "B": "second"},
                }
                for i in range(10)
            },
        },
    )


class Backend:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.cached = False
        self.concurrent = 0
        self.maximum = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reset_prefix_cache":
            self.cached = False
            return httpx.Response(200, json={"success": True})
        if request.url.path in {"/reset_mm_cache", "/reset_encoder_cache"}:
            return httpx.Response(200)
        payload = json.loads(request.content)
        index = int(payload["messages"][0]["content"][-1]["text"].split(" q")[1])
        hit = 2 if self.cached and self.enabled else 0
        self.cached = True
        self.concurrent += 1
        self.maximum = max(self.maximum, self.concurrent)
        try:
            await asyncio.sleep(0.02 if index == 0 else 0.001)
        finally:
            self.concurrent -= 1
        return httpx.Response(
            200,
            json={
                "model": "pinned",
                "prompt_token_ids": [10, 20, 100 + index],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": hit},
                },
                "metrics": {
                    "time_to_first_token_ms": 2.0,
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
                                    "token": "token_id:1",
                                    "logprob": -2.0,
                                    "top_logprobs": [
                                        {"token": f"token_id:{token}", "logprob": -float(token + 1)}
                                        for token in (1, 2, 3)
                                    ],
                                }
                            ]
                        },
                    }
                ],
            },
        )


async def collect(
    enabled: bool, defect: str | None = None, *, frozen_qids: bool = False
) -> tuple[Any, Backend]:
    frozen = gateway_protocol()
    if frozen_qids:
        frozen = ScheduledGatewayProtocol.model_validate(
            {
                **frozen.model_dump(),
                "request": {
                    **frozen.request,
                    "questions": {
                        f"cache-probe-{index}": item
                        for index, item in enumerate(frozen.request["questions"].values())
                    },
                },
            }
        )
    record = server(enabled)
    backend = Backend(enabled)
    capture = ScheduledCaptureTransport(
        httpx.MockTransport(backend),
        frozen.native,
        record,
        asyncio.get_running_loop().time,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        before = len(capture.observations)
        answers: dict[str, Any] = {}
        async with httpx.AsyncClient(transport=capture) as client:
            items = list(body["questions"].items())

            async def one(qid: str) -> None:
                index = list(frozen.request["questions"]).index(qid)
                response = await client.post(
                    record.base_url + "/v1/chat/completions",
                    json=frozen.native.payload(record.model, index),
                )
                response.raise_for_status()
                row = capture.observations[-1]
                item = InputQuestion.model_validate(body["questions"][qid])
                answers[qid] = expected_gateway_answer(item, qid, row.raw_logprobs)
                if defect == "wrong_prob" and qid == "q1":
                    answers[qid]["probabilities"]["A"] = 0.123
                if defect == "wrong_mapping" and qid == "q1":
                    answers[qid]["choice"] = "B" if answers[qid]["choice"] == "A" else "A"

            await one(items[0][0])
            await asyncio.gather(*(one(qid) for qid, _ in items[1:]))
        rows = capture.observations[before:]
        upstream = sum(row.http_wall_seconds for row in rows) * 1000
        timing = f"preprocess_work;dur=3,upstream_work;dur={upstream},total;dur=4,queue_work;dur=1"
        if defect == "headers":
            headers = {"Server-Timing": timing, "X-Lite-Question-Concurrency": "10"}
        else:
            headers = {
                "Server-Timing": timing,
                "X-Lite-Execution-Mode": "bounded-scheduled-v1",
                "X-Lite-Question-Concurrency": "10",
            }
        answers = {qid: answers[qid] for qid in body["questions"] if qid in answers}
        if defect == "missing":
            answers.pop(next(iter(answers)))
        if defect == "reordered":
            answers = dict(reversed(list(answers.items())))
        if defect == "duplicate":
            extra = frozen.native.payload(record.model, 0)
            async with httpx.AsyncClient(transport=capture) as client:
                await client.post(record.base_url + "/v1/chat/completions", json=extra)
        return httpx.Response(
            200,
            headers=headers,
            json={
                "answers": answers,
                "backend_usage": {"input_tokens": 3 * len(rows), "output_tokens": len(rows)},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://127.0.0.1"
    ) as gateway:
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as control:
            run = await collect_scheduled_gateway(
                frozen,
                record,
                gateway,
                control,
                capture,
                allow_reset=True,
                clock=asyncio.get_running_loop().time,
            )
    return run, backend


def test_scheduled_gateway_collects_real_overlap_and_reuses_assessment() -> None:
    on, backend = asyncio.run(collect(True))
    off, _ = asyncio.run(collect(False))
    result = assess_scheduled_gateway(on, off)
    assert result["status"] == "mock-validated"
    assert result["assessment_policy"] == "bounded-scheduled-v1"
    assert result["s0_acceptance"] is False
    ten = next(m for m in on.measurements if (m.condition, m.size, m.repeat) == ("cold", 10, 0))
    assert ten.batch_started_at <= ten.observations[0].request_started_at
    assert ten.batch_finished_at >= max(row.parsed_finished_at for row in ten.observations)
    assert ten.batch_wall_seconds < ten.workload_http_wall_seconds
    assert ten.actual_max_inflight == backend.maximum == 9
    assert [row.case_index for row in ten.observations] == list(range(10))
    assert ten.api_answers_sha256
    assert list(ten.api_answer_sha256_by_qid) == [f"q{i}" for i in range(10)]
    assert result["timing_rows"][0]["server_timing_ms"]["preprocess_work"] == 3.0
    assert result["timing_rows"][0]["api_answers_sha256"]


def test_gateway_run_json_roundtrip_preserves_and_validates_api_answer_digests() -> None:
    run, _ = asyncio.run(collect(True))
    reloaded = ScheduledGatewayRun.model_validate_json(run.model_dump_json())
    assert reloaded.api_answer_digests == run.api_answer_digests
    assert reloaded.api_answer_digests[1].api_answers_sha256

    data = json.loads(run.model_dump_json())
    data["api_answer_digests"][1]["api_answers_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="API answer digest mismatch"):
        ScheduledGatewayRun.model_validate(data)


def test_real_frozen_question_ids_survive_collection_and_roundtrip() -> None:
    run, _ = asyncio.run(collect(True, frozen_qids=True))
    reloaded = ScheduledGatewayRun.model_validate_json(run.model_dump_json())
    assert reloaded.gateway_question_ids == [f"cache-probe-{i}" for i in range(10)]
    assert list(reloaded.api_answer_digests[1].api_answer_sha256_by_qid) == (
        reloaded.gateway_question_ids
    )
    data = json.loads(run.model_dump_json())
    data["gateway_question_ids"] = list(reversed(data["gateway_question_ids"]))
    with pytest.raises(ValueError, match="qid order"):
        ScheduledGatewayRun.model_validate(data)

    data = json.loads(run.model_dump_json())
    data["api_answer_digests"][1]["api_answer_sha256_by_qid"] = {"q9": "0" * 64}
    data["api_answer_digests"][1]["api_answers_sha256"] = digest({"q9": "0" * 64})
    with pytest.raises(ValueError, match="qid order"):
        ScheduledGatewayRun.model_validate(data)


@pytest.mark.parametrize(
    "defect", ["headers", "missing", "reordered", "duplicate", "wrong_prob", "wrong_mapping"]
)
def test_scheduled_gateway_rejects_bad_headers_or_outputs(defect: str) -> None:
    with pytest.raises(ValueError):
        asyncio.run(collect(True, defect))


def test_capture_rejects_input_drift() -> None:
    frozen = gateway_protocol()
    record = server(True)
    capture = ScheduledCaptureTransport(
        httpx.MockTransport(Backend(True)),
        frozen.native,
        record,
        asyncio.get_event_loop_policy().new_event_loop().time,
    )
    payload = frozen.native.payload(record.model, 0)
    payload["top_k"] = 4

    async def main() -> None:
        async with httpx.AsyncClient(transport=capture) as client:
            with pytest.raises(ValueError, match="exclusive server record"):
                await client.post("http://example.com/v1/chat/completions", json=payload)
            with pytest.raises(ValueError, match="sampling/token"):
                await client.post(record.base_url + "/v1/chat/completions", json=payload)

    asyncio.run(main())


@pytest.mark.parametrize(
    "header",
    [
        "",
        "preprocess;dur=1,upstream_work;dur=1,queue_work;dur=0,total;dur=2",
        "preprocess_work;dur=nan,upstream_work;dur=1,queue_work;dur=0,total;dur=2",
        "preprocess_work;dur=1,preprocess_work;dur=1,queue_work;dur=0,total;dur=2",
    ],
)
def test_scheduled_timer_parser_refuses_bad_headers(header: str) -> None:
    with pytest.raises(ValueError):
        parse_scheduled_server_timing(header)


def test_cli_validate_is_network_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.s0_scheduled_cache_gateway import main

    native_path = tmp_path / "native.json"
    request_path = tmp_path / "gateway.json"
    server_path = tmp_path / "server.json"
    model_dir = tmp_path / "model"
    media_root = tmp_path / "media"
    report_path = tmp_path / "report.json"
    model_dir.mkdir()
    media_root.mkdir()
    (model_dir / "tokenizer_config.json").write_text("{}")
    native_path.write_text(protocol().model_dump_json())
    request_path.write_text(json.dumps(gateway_protocol().request))
    server_path.write_text(server(True).model_dump_json())

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("validate must not create HTTP transports")

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", forbidden)
    monkeypatch.setattr(
        "sys.argv",
        [
            "s0-scheduled-cache-gateway",
            "validate",
            "--native-protocol",
            str(native_path),
            "--gateway-request",
            str(request_path),
            "--server-record",
            str(server_path),
            "--model-dir",
            str(model_dir),
            "--media-root",
            str(media_root),
            "--report",
            str(report_path),
        ],
    )
    assert main() == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "validated-no-network"
    assert report["mode"] == "bounded-scheduled-v1"
    assert report["s0_acceptance"] is False
    assert "request_media_cache_source_sha256" in report["gateway_configuration"]
