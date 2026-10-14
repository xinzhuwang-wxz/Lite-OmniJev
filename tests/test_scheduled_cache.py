"""Scheduled native cache tests use MockTransport only; they are not performance evidence."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
from typing import Any

import httpx
import pytest

from eval.prefix_cache import CacheCase, CacheProtocol, CacheServer, digest
from eval.scheduled_cache import ScheduledCacheRun, assess_scheduled, collect_scheduled


def protocol() -> CacheProtocol:
    png = b"\x89PNG\r\n\x1a\nSCHEDULED_SYNTHETIC"
    return CacheProtocol(
        image_png_base64=base64.b64encode(png).decode(),
        image_sha256=hashlib.sha256(png).hexdigest(),
        shared_text="shared media prompt:",
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
        run_id="scheduled-on" if enabled else "scheduled-off",
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


class Backend:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.cached = False
        self.resets = 0
        self.sequence: list[int] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self.failure: str | None = None
        self.cancelled = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/reset_prefix_cache":
            assert request.url.params["reset_running_requests"] == "false"
            assert request.url.params["reset_external"] == "false"
            self.resets += 1
            self.cached = False
            return httpx.Response(200, json={"success": self.failure != "reset"})
        if request.url.path in {"/reset_mm_cache", "/reset_encoder_cache"}:
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert payload["top_k"] == -1
        assert payload["top_logprobs"] == 0
        assert payload["return_prompt_text"] is False
        assert payload["logprob_token_ids"] == [1, 2, 3]
        index = int(payload["messages"][0]["content"][-1]["text"].split(" q")[1])
        self.sequence.append(index)
        if self.failure == "case1_500" and index == 1:
            return httpx.Response(500, json={"error": "synthetic"})
        hit = 2 if self.cached and self.enabled else 0
        self.cached = True
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.failure == "case1_500":
                await asyncio.sleep(0.05)
            else:
                await asyncio.sleep(0.02 if index == 0 else 0.001)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.concurrent -= 1
        body: dict[str, Any] = {
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
        }
        if self.failure == "missing_u":
            body["choices"][0]["logprobs"]["content"][0]["top_logprobs"].pop()
        if self.failure == "bad_prompt":
            body["prompt_token_ids"] = [10, 999, 100 + index]
        if self.failure == "bad_cache":
            body["usage"]["prompt_tokens_details"]["cached_tokens"] = 99
        return httpx.Response(200, json=body)


async def run(enabled: bool, *, failure: str | None = None) -> tuple[ScheduledCacheRun, Backend]:
    backend = Backend(enabled)
    backend.failure = failure
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
        result = await collect_scheduled(
            protocol(),
            server(enabled),
            client,
            allow_reset=True,
            concurrency_limit=3,
            clock=asyncio.get_running_loop().time,
        )
    return result, backend


def test_bounded_scheduled_collects_after_first_and_returns_by_index() -> None:
    on, backend = asyncio.run(run(True))
    off, _ = asyncio.run(run(False))

    result = assess_scheduled(on, off)
    assert result["status"] == "mock-validated"
    assert result["s0_acceptance"] is False
    assert result["assessment_policy"] == "bounded-scheduled-v1"
    assert backend.resets == 8
    assert backend.max_concurrent == 3
    assert max(m.actual_max_inflight for m in on.measurements) == 3
    assert backend.sequence == [0] + ([0] + list(range(10)) + [0, 0] + [0] + list(range(10))) * 2
    ten = next(m for m in on.measurements if (m.condition, m.size, m.repeat) == ("cold", 10, 0))
    assert [row.case_index for row in ten.observations] == list(range(10))
    assert ten.batch_wall_seconds < ten.workload_http_wall_seconds
    assert ten.batch_started_at <= ten.observations[0].request_started_at
    assert ten.batch_finished_at >= max(row.parsed_finished_at for row in ten.observations)
    assert ten.batch_wall_seconds == pytest.approx(ten.batch_finished_at - ten.batch_started_at)
    assert ten.actual_max_inflight <= ten.concurrency_limit
    first_done = ten.observations[0].parsed_finished_at
    assert all(row.request_started_at >= first_done for row in ten.observations[1:])


def test_no_reset_or_network_without_permission() -> None:
    calls: list[httpx.Request] = []

    def record_call(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(record_call)) as client:
            with pytest.raises(ValueError, match="explicit reset permission"):
                await collect_scheduled(
                    protocol(),
                    server(True),
                    client,
                    clock=asyncio.get_running_loop().time,
                )

    asyncio.run(main())
    assert calls == []


def test_max_num_seqs_must_cover_concurrency() -> None:
    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(Backend(True))) as client:
            with pytest.raises(ValueError, match="max-num-seqs"):
                await collect_scheduled(
                    protocol(),
                    server(True, max_num_seqs=2),
                    client,
                    allow_reset=True,
                    concurrency_limit=3,
                    clock=asyncio.get_running_loop().time,
                )

    asyncio.run(main())


def test_bad_concurrency_is_rejected_before_network() -> None:
    calls: list[httpx.Request] = []

    def record_call(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(record_call)) as client:
            with pytest.raises(ValueError, match="integer between 2 and 10"):
                await collect_scheduled(
                    protocol(),
                    server(True),
                    client,
                    allow_reset=True,
                    concurrency_limit=True,
                    clock=asyncio.get_running_loop().time,
                )

    asyncio.run(main())
    assert calls == []


@pytest.mark.parametrize("failure", ["reset", "missing_u", "bad_prompt", "bad_cache"])
def test_bad_native_evidence_is_rejected(failure: str) -> None:
    with pytest.raises(ValueError):
        asyncio.run(run(True, failure=failure))


def test_failed_fanout_cancels_and_drains_sibling_requests() -> None:
    backend = Backend(True)
    backend.failure = "case1_500"

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await collect_scheduled(
                    protocol(),
                    server(True),
                    client,
                    allow_reset=True,
                    concurrency_limit=3,
                    clock=asyncio.get_running_loop().time,
                )

    asyncio.run(main())
    assert backend.max_concurrent == 3
    assert backend.cancelled > 0
    assert backend.concurrent == 0


def test_compare_rejects_missing_measurement_and_cache_misreports() -> None:
    on, _ = asyncio.run(run(True))
    off, _ = asyncio.run(run(False))
    on.measurements.pop()
    with pytest.raises(ValueError, match="missing or duplicate"):
        assess_scheduled(on, off)

    on, _ = asyncio.run(run(True))
    off, _ = asyncio.run(run(False))
    on.measurements[1].observations[1].cached_tokens = 0
    off.measurements[1].observations[0].cached_tokens = 2
    result = assess_scheduled(on, off)
    assert "missing_prefix_hit" in result["failures"]
    assert "cache_off_hit" in result["failures"]


def test_schema_binds_run_concurrency_and_launch_record() -> None:
    on, _ = asyncio.run(run(True))
    data = on.model_dump()
    data["measurements"][0]["concurrency_limit"] = 2
    with pytest.raises(ValueError, match="measurement concurrency"):
        ScheduledCacheRun.model_validate(data)

    data = on.model_dump()
    data["launch_max_num_seqs"] = 9
    with pytest.raises(ValueError, match="launch command"):
        ScheduledCacheRun.model_validate(data)


def test_compare_uses_probability_distributions_and_cold_ratio_only() -> None:
    on, _ = asyncio.run(run(True))
    off, _ = asyncio.run(run(False))
    warm_single = next(
        m for m in on.measurements if (m.condition, m.size, m.repeat) == ("warm", 1, 0)
    )
    warm_ten = next(
        m for m in on.measurements if (m.condition, m.size, m.repeat) == ("warm", 10, 0)
    )
    on.measurements[1].observations[0].raw_logprobs["A"] = math.log(
        math.exp(on.measurements[1].observations[0].raw_logprobs["A"]) + 0.0011
    )
    on.measurements[1].observations[0].raw_logprobs_sha256 = "0" * 64

    with pytest.raises(ValueError, match="raw logprobs digest mismatch"):
        ScheduledCacheRun.model_validate(on.model_dump())

    on.measurements[1].observations[0].raw_logprobs_sha256 = "0" * 64
    data = on.model_dump()
    data["measurements"][1]["observations"][0]["raw_logprobs_sha256"] = digest(
        data["measurements"][1]["observations"][0]["raw_logprobs"]
    )
    on = ScheduledCacheRun.model_validate(data)
    warm_ten = next(
        m for m in on.measurements if (m.condition, m.size, m.repeat) == ("warm", 10, 0)
    )
    warm_ten.batch_wall_seconds = warm_single.batch_wall_seconds * 3
    result = assess_scheduled(on, off)
    assert result["status"] == "failed"
    assert "probability_distribution_changed" in result["failures"]
    warm_group = next(
        group
        for group in result["groups"]
        if group["cache_enabled"] and group["condition"] == "warm"
    )
    assert warm_group["maximum_ratio"] > 2
    assert warm_group["acceptance_condition"] is False


def test_cli_validate_is_network_free_and_failure_safe(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.s0_scheduled_cache import main

    protocol_path = tmp_path / "protocol.json"
    server_path = tmp_path / "server.json"
    report_path = tmp_path / "report.json"
    protocol_path.write_text(protocol().model_dump_json())
    server_path.write_text(server(True).model_dump_json())

    def forbidden_client(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("validation must not create an HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_client)
    monkeypatch.setattr(
        "sys.argv",
        [
            "s0-scheduled-cache",
            "validate",
            "--protocol",
            str(protocol_path),
            "--server-record",
            str(server_path),
            "--report",
            str(report_path),
        ],
    )
    assert main() == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "validated-no-network"
    assert report["mode"] == "bounded-scheduled-v1"
    assert report["s0_acceptance"] is False
    assert "shared media prompt" not in report_path.read_text()

    protocol_path.write_text('{"image_png_base64":"PRIVATE-SENTINEL"}')
    report_path.unlink()
    assert main() == 1
    assert "PRIVATE-SENTINEL" not in report_path.read_text()
    assert json.loads(report_path.read_text())["error_type"] == "ValidationError"
