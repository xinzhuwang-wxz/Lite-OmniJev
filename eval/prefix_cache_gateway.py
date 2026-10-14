"""Measure the actual sequential gateway and its upstream with independent timers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal, Self

import httpx
from pydantic import Field, model_validator

from data.schema import Contract, Prefix, Sha256
from eval.prefix_cache import (
    CacheProtocol,
    CacheRun,
    CacheServer,
    Measurement,
    Observation,
    assess,
    digest,
)
from serve.gateway.app import InputQuestion
from serve.vllm_chat import parse_chat_logprobs


class GatewayProtocol(Contract):
    native: CacheProtocol
    request: dict[str, Any]
    gateway_configuration_sha256: Sha256

    @model_validator(mode="after")
    def canonical_request(self) -> Self:
        request = self.request
        prefix = Prefix.model_validate({"state": request["state"], "media": request["media"]})
        if len(prefix.media) != 1 or prefix.media[0].type != "image":
            raise ValueError("one shared image required")
        questions = request["questions"]
        if not isinstance(questions, dict) or len(questions) != 10:
            raise ValueError("ten ordered gateway questions required")
        parsed = [InputQuestion.model_validate(question) for question in questions.values()]
        if any((q.type, q.criteria) != (parsed[0].type, parsed[0].criteria) for q in parsed):
            raise ValueError("gateway legal answer sets must match")
        if self.native.seed != 42:
            raise ValueError("gateway native client has a fixed seed of 42")
        return self


class GatewayTiming(Contract):
    preprocess: float = Field(ge=0)
    upstream: float = Field(ge=0)
    queue: float = Field(ge=0)
    total: float = Field(gt=0)

    @model_validator(mode="after")
    def separate_intervals(self) -> Self:
        if self.preprocess + self.upstream + self.queue > self.total + 0.01:
            raise ValueError("gateway timing intervals exceed total wall")
        return self


def parse_server_timing(header: str) -> GatewayTiming:
    """Only the gateway's fixed dur fields; never infer preprocessing by subtraction."""
    values = {}
    for entry in header.split(","):
        fields = [part.strip() for part in entry.split(";")]
        if len(fields) != 2 or fields[0] in values or not fields[1].startswith("dur="):
            raise ValueError("invalid fixed Server-Timing protocol")
        values[fields[0]] = float(fields[1][4:])
    return GatewayTiming.model_validate(values)


class GatewayRun(CacheRun):
    gateway_request_sha256: Sha256
    gateway_configuration_sha256: Sha256
    timings: list[GatewayTiming]


class CaptureTransport(httpx.AsyncBaseTransport):
    """Wrap the gateway's real upstream transport; fail if native calls overlap or differ."""

    def __init__(
        self,
        inner: httpx.AsyncBaseTransport,
        protocol: CacheProtocol,
        server: CacheServer,
        clock: Callable[[], float],
    ) -> None:
        self.inner = inner
        self.protocol = protocol
        self.server = server
        self.clock = clock
        self.observations: list[Observation] = []
        self.inflight = False

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.inflight:
            raise ValueError("concurrent upstream calls cannot prove sequential latency")
        expected_url = self.server.base_url.rstrip("/") + "/v1/chat/completions"
        if request.url != httpx.URL(expected_url):
            raise ValueError("upstream endpoint differs from the exclusive server record")
        payload = json.loads(request.content)
        matching = [
            index
            for index in range(10)
            if payload.get("messages")
            == self.protocol.payload(self.server.model, index)["messages"]
        ]
        if len(matching) != 1:
            raise ValueError("gateway upstream input differs from frozen canonical input")
        index = matching[0]
        expected = self.protocol.payload(self.server.model, index)
        if any(
            payload.get(key, False if key == "return_prompt_text" else None) != value
            for key, value in expected.items()
        ):
            raise ValueError("gateway upstream sampling/token protocol differs")
        if set(payload) - set(expected):
            raise ValueError("unexpected upstream generation modifiers")
        self.inflight = True
        try:
            started = self.clock()
            response = await self.inner.handle_async_request(request)
            response.request = request
            await response.aread()
            elapsed = self.clock() - started
            response.raise_for_status()
            body = response.json()
            parsed = parse_chat_logprobs(
                body,
                model=self.server.model,
                answer_token_ids=self.protocol.answer_token_ids,
                expected_prompt_ids=self.protocol.cases[index].prompt_token_ids,
            )
            self.observations.append(
                Observation(
                    case_index=index,
                    input_sha256=digest(expected),
                    http_wall_seconds=elapsed,
                    payload_preparation_seconds=0.0,
                    cached_tokens=body["usage"]["prompt_tokens_details"]["cached_tokens"],
                    prompt_tokens=len(parsed.prompt_token_ids),
                    raw_logprobs=parsed.raw_logprobs,
                    metrics={
                        key: body["metrics"][key]
                        for key in (
                            "time_to_first_token_ms",
                            "generation_time_ms",
                            "queue_time_ms",
                        )
                    },
                )
            )
            return response
        finally:
            self.inflight = False


async def collect_gateway(
    protocol: GatewayProtocol,
    server: CacheServer,
    gateway: httpx.AsyncClient,
    control: httpx.AsyncClient,
    capture: CaptureTransport,
    *,
    allow_reset: bool = False,
    clock: Callable[[], float],
) -> GatewayRun:
    """The supplied gateway client is an in-process ASGI client, not a remote gateway."""
    if not allow_reset or not server.exclusive_local_test:
        raise ValueError("exclusive test attestation and explicit reset permission required")
    if capture.protocol != protocol.native or capture.server != server:
        raise ValueError("upstream capture must bind the same protocol and server")
    base = server.base_url.rstrip("/")

    async def reset() -> None:
        result = await control.post(
            base + "/reset_prefix_cache",
            params={
                "reset_running_requests": "false",
                "reset_external": "false",
            },
        )
        result.raise_for_status()
        if result.json() != {"success": True}:
            raise ValueError("native prefix reset failed")
        for endpoint in ("reset_mm_cache", "reset_encoder_cache"):
            result = await control.post(base + "/" + endpoint)
            result.raise_for_status()

    async def request(size: Literal[1, 10]) -> tuple[Measurement, GatewayTiming]:
        payload = dict(protocol.request)
        payload["questions"] = dict(list(protocol.request["questions"].items())[:size])
        before = len(capture.observations)
        started = clock()
        response = await gateway.post("/v1/systemone", json=payload)
        wall = clock() - started
        response.raise_for_status()
        timing = parse_server_timing(response.headers.get("Server-Timing", ""))
        if timing.total > wall * 1000 + 0.01:
            raise ValueError("gateway reported total exceeds client wall")
        rows = capture.observations[before:]
        if [row.case_index for row in rows] != list(range(size)):
            raise ValueError("gateway did not complete the exact ordered native calls")
        body = response.json()
        if set(body["answers"]) != set(payload["questions"]) or (
            body["backend_usage"]["input_tokens"] != sum(row.prompt_tokens for row in rows)
            or body["backend_usage"]["output_tokens"] != size
        ):
            raise ValueError("gateway completion/usage does not match native evidence")
        if sum(row.http_wall_seconds for row in rows) * 1000 > timing.upstream + 0.01:
            raise ValueError("native durations exceed measured gateway upstream interval")
        return Measurement(
            repeat=0, condition="cold", size=size, wall_seconds=wall, observations=rows
        ), timing

    for _ in range(protocol.native.warmup_requests):
        await request(1)
    measurements = []
    timings = []
    for repeat in range(protocol.native.repeats):
        for condition in ("cold", "warm"):
            for size in (1, 10):
                await reset()
                if condition == "warm":
                    await request(1)
                measured, timing = await request(size)
                measured.repeat = repeat
                measured.condition = condition
                measurements.append(measured)
                timings.append(timing)
    return GatewayRun(
        protocol_sha256=digest(protocol.native.model_dump()),
        server=server,
        repeats=protocol.native.repeats,
        output_tolerance=protocol.native.output_tolerance,
        media_prefix_end=protocol.native.media_prefix_end,
        measurements=measurements,
        gateway_request_sha256=digest(protocol.request),
        gateway_configuration_sha256=protocol.gateway_configuration_sha256,
        timings=timings,
    )


def assess_gateway(on: GatewayRun, off: GatewayRun) -> dict[str, Any]:
    if (
        on.gateway_configuration_sha256 != off.gateway_configuration_sha256
        or on.gateway_request_sha256 != off.gateway_request_sha256
    ):
        raise ValueError("gateway configuration or frozen public request changed")
    result = assess(on, off)
    timing_rows = []
    for run in (on, off):
        if len(run.timings) != len(run.measurements):
            raise ValueError("missing actual gateway timing headers")
        for measurement, timing in zip(run.measurements, run.timings, strict=True):
            timing_rows.append(
                {
                    "cache_enabled": run.server.cache_enabled,
                    "repeat": measurement.repeat,
                    "condition": measurement.condition,
                    "size": measurement.size,
                    "gateway_client_wall_seconds": measurement.wall_seconds,
                    "server_timing_ms": timing.model_dump(),
                    "native_http_sum_seconds": sum(
                        row.http_wall_seconds for row in measurement.observations
                    ),
                }
            )
    result.update(
        timing_rows=timing_rows,
        timing_boundary="actual in-process gateway ASGI wall; measured Server-Timing intervals",
        gateway_configuration_sha256=on.gateway_configuration_sha256,
        gateway_request_sha256=on.gateway_request_sha256,
        transport_scope="in-process real gateway; networked upstream; no external gateway network",
    )
    return result
