"""Measure the real ASGI gateway in bounded scheduled mode with upstream evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any, Literal, Self

import httpx
from pydantic import Field, model_validator

from data.schema import Contract, Prefix, Sha256, answer_tokens
from eval.prefix_cache import CacheProtocol, CacheServer, digest
from eval.scheduled_cache import (
    ScheduledCacheRun,
    ScheduledMeasurement,
    ScheduledObservation,
    assess_scheduled,
    validate_scheduled_launch,
)
from serve.gateway.app import InputQuestion, core_question
from serve.vllm_chat import parse_chat_logprobs
from train.readout import read_probabilities


class ScheduledGatewayProtocol(Contract):
    native: CacheProtocol
    request: dict[str, Any]
    gateway_configuration_sha256: Sha256
    question_concurrency: int = Field(default=10, ge=2, le=10)
    request_image_reuse: bool = True

    @model_validator(mode="after")
    def canonical_request(self) -> Self:
        if self.question_concurrency != 10 or self.request_image_reuse is not True:
            raise ValueError("scheduled gateway candidate requires concurrency=10 and image reuse")
        request = self.request
        prefix = Prefix.model_validate({"state": request["state"], "media": request["media"]})
        if len(prefix.media) != 1 or prefix.media[0].type != "image":
            raise ValueError("one shared image required")
        questions = request["questions"]
        if not isinstance(questions, dict) or len(questions) != 10:
            raise ValueError("ten ordered gateway questions required")
        parsed = [InputQuestion.model_validate(question) for question in questions.values()]
        if any(
            (question.type, question.criteria) != (parsed[0].type, parsed[0].criteria)
            for question in parsed
        ):
            raise ValueError("gateway legal answer sets must match")
        if self.native.seed != 42:
            raise ValueError("gateway native client has a fixed seed of 42")
        return self


class ScheduledGatewayTiming(Contract):
    preprocess_work: float = Field(ge=0)
    upstream_work: float = Field(ge=0)
    queue_work: float = Field(ge=0)
    total: float = Field(gt=0)


class ScheduledGatewayAnswerDigest(Contract):
    api_answer_sha256_by_qid: dict[str, Sha256]
    api_answers_sha256: Sha256

    @model_validator(mode="after")
    def digest_matches_qids(self) -> Self:
        if not self.api_answer_sha256_by_qid:
            raise ValueError("API answer digests are required")
        if self.api_answers_sha256 != digest(self.api_answer_sha256_by_qid):
            raise ValueError("API answer digest mismatch")
        return self


class ScheduledGatewayMeasurement(ScheduledMeasurement):
    question_ids: list[str]
    api_answer_sha256_by_qid: dict[str, Sha256]
    api_answers_sha256: Sha256

    @model_validator(mode="after")
    def answer_digest_matches_rows(self) -> Self:
        expected_qids = self.question_ids
        if len(expected_qids) != len(self.observations) or len(set(expected_qids)) != len(
            expected_qids
        ):
            raise ValueError("measured question IDs must be unique and complete")
        if list(self.api_answer_sha256_by_qid) != expected_qids:
            raise ValueError("API answer digests must follow measured qid order")
        if self.api_answers_sha256 != digest(self.api_answer_sha256_by_qid):
            raise ValueError("API answer digest mismatch")
        return self


def expected_gateway_answer(
    item: InputQuestion, qid: str, raw_logprobs: dict[str, float]
) -> dict[str, Any]:
    question, labels = core_question(qid, item)
    readout = read_probabilities(raw_logprobs, answer_tokens(question))
    values = list(readout.probabilities.values())
    probabilities = dict(zip(labels, values, strict=True))
    result: dict[str, Any] = {
        "type": item.type,
        "probabilities": probabilities,
        "confidence": readout.confidence,
        "p_insufficient": readout.p_insufficient,
        "calibration": "unfitted",
        "confidence_semantics": "normalized_peak_non_U",
    }
    if item.type == "choice":
        result["choice"] = labels[values.index(max(values))]
    elif item.type == "noul":
        result["noul"] = probabilities["yes"]
    else:
        assert isinstance(item.criteria, list)
        result["score"] = sum(index * probability for index, probability in enumerate(values))
        result["legend"] = {str(i): description for i, description in enumerate(item.criteria)}
    return result


def parse_scheduled_server_timing(header: str) -> ScheduledGatewayTiming:
    values: dict[str, float] = {}
    for entry in header.split(","):
        fields = [part.strip() for part in entry.split(";")]
        if len(fields) != 2 or fields[0] in values or not fields[1].startswith("dur="):
            raise ValueError("invalid scheduled Server-Timing protocol")
        value = float(fields[1][4:])
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid scheduled Server-Timing duration")
        values[fields[0]] = value
    return ScheduledGatewayTiming.model_validate(values)


class ScheduledGatewayRun(ScheduledCacheRun):
    gateway_question_ids: list[str] = Field(min_length=10, max_length=10)
    gateway_request_sha256: Sha256
    gateway_configuration_sha256: Sha256
    scheduled_gateway_timings: list[ScheduledGatewayTiming]
    api_answer_digests: list[ScheduledGatewayAnswerDigest]
    transport_scope: str = "in-process real gateway ASGI; networked upstream capture"

    @model_validator(mode="after")
    def timing_count_matches_measurements(self) -> Self:
        if len(set(self.gateway_question_ids)) != 10:
            raise ValueError("gateway question IDs must be unique")
        if len(self.scheduled_gateway_timings) != len(self.measurements):
            raise ValueError("missing scheduled gateway timing rows")
        if len(self.api_answer_digests) != len(self.measurements):
            raise ValueError("missing scheduled gateway API answer digests")
        for measurement, answer_digest in zip(
            self.measurements, self.api_answer_digests, strict=True
        ):
            expected_qids = [
                self.gateway_question_ids[row.case_index] for row in measurement.observations
            ]
            if list(answer_digest.api_answer_sha256_by_qid) != expected_qids:
                raise ValueError("API answer digests must follow measured qid order")
        return self


class ScheduledCaptureTransport(httpx.AsyncBaseTransport):
    """Capture concurrent gateway upstream requests without serializing the gateway."""

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
        self.observations: list[ScheduledObservation] = []
        self.current = 0
        self.maximum = 0

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
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
        started = self.clock()
        self.current += 1
        self.maximum = max(self.maximum, self.current)
        try:
            response = await self.inner.handle_async_request(request)
            response.request = request
            await response.aread()
        finally:
            self.current -= 1
        response_finished = self.clock()
        response.raise_for_status()
        body = response.json()
        parsed = parse_chat_logprobs(
            body,
            model=self.server.model,
            answer_token_ids=self.protocol.answer_token_ids,
            expected_prompt_ids=self.protocol.cases[index].prompt_token_ids,
        )
        cached = body["usage"]["prompt_tokens_details"]["cached_tokens"]
        if type(cached) is not int or not 0 <= cached <= len(parsed.prompt_token_ids):
            raise ValueError("invalid native cached_tokens")
        if not self.server.cache_enabled and cached != 0:
            raise ValueError("cache-off response claims a prefix hit")
        metrics = {
            key: body["metrics"][key]
            for key in (
                "time_to_first_token_ms",
                "generation_time_ms",
                "queue_time_ms",
            )
        }
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in metrics.values()
        ):
            raise ValueError("native timing metrics missing or invalid")
        parsed_finished = self.clock()
        self.observations.append(
            ScheduledObservation(
                case_index=index,
                input_sha256=digest(expected),
                http_wall_seconds=response_finished - started,
                payload_preparation_seconds=0.0,
                cached_tokens=cached,
                prompt_tokens=len(parsed.prompt_token_ids),
                raw_logprobs=parsed.raw_logprobs,
                metrics=metrics,
                request_started_at=started,
                response_finished_at=response_finished,
                parsed_finished_at=parsed_finished,
                raw_logprobs_sha256=digest(parsed.raw_logprobs),
                native_metrics=metrics,
            )
        )
        return response


async def collect_scheduled_gateway(
    protocol: ScheduledGatewayProtocol,
    server: CacheServer,
    gateway: httpx.AsyncClient,
    control: httpx.AsyncClient,
    capture: ScheduledCaptureTransport,
    *,
    allow_reset: bool = False,
    clock: Callable[[], float],
) -> ScheduledGatewayRun:
    """Collect scheduled evidence through the real in-process gateway ASGI app."""
    max_num_seqs = validate_scheduled_launch(server, protocol.question_concurrency)
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

    async def request(
        size: Literal[1, 10],
    ) -> tuple[ScheduledGatewayMeasurement, ScheduledGatewayTiming]:
        payload = dict(protocol.request)
        payload["questions"] = dict(list(protocol.request["questions"].items())[:size])
        before = len(capture.observations)
        batch_started_at = clock()
        response = await gateway.post("/v1/systemone", json=payload)
        response.raise_for_status()
        body = response.json()
        timing = parse_scheduled_server_timing(response.headers.get("Server-Timing", ""))
        if response.headers.get("X-Lite-Execution-Mode") != "bounded-scheduled-v1":
            raise ValueError("gateway did not report bounded scheduled execution")
        if response.headers.get("X-Lite-Question-Concurrency") != str(
            protocol.question_concurrency
        ):
            raise ValueError("gateway concurrency header mismatch")
        captured = capture.observations[before:]
        if len(captured) != size:
            raise ValueError("gateway upstream capture count mismatch")
        case_indices = [row.case_index for row in captured]
        if len(set(case_indices)) != size:
            raise ValueError("gateway upstream capture repeated a case index")
        rows_by_case = {row.case_index: row for row in captured}
        rows = (
            [rows_by_case[index] for index in range(size)]
            if set(rows_by_case) == set(range(size))
            else []
        )
        if len(rows) != size:
            raise ValueError("gateway did not complete the exact scheduled native calls")
        expected_qids = list(payload["questions"])
        if list(body["answers"]) != expected_qids:
            raise ValueError("gateway answers are missing, reordered or mis-keyed")
        api_answer_sha256_by_qid: dict[str, Sha256] = {}
        for qid, row in zip(expected_qids, rows, strict=True):
            item = InputQuestion.model_validate(payload["questions"][qid])
            expected_answer = expected_gateway_answer(item, qid, row.raw_logprobs)
            if body["answers"][qid] != expected_answer:
                raise ValueError("gateway API answer does not match captured raw readout")
            api_answer_sha256_by_qid[qid] = digest(body["answers"][qid])
        if (
            body["backend_usage"]["input_tokens"] != sum(row.prompt_tokens for row in rows)
            or body["backend_usage"]["output_tokens"] != size
        ):
            raise ValueError("gateway completion/usage does not match native evidence")
        if timing.total > (clock() - batch_started_at) * 1000 + 0.01:
            raise ValueError("gateway reported total exceeds client wall")
        batch_finished_at = clock()
        return (
            ScheduledGatewayMeasurement(
                repeat=0,
                condition="cold",
                size=size,
                concurrency_limit=protocol.question_concurrency,
                batch_started_at=batch_started_at,
                batch_finished_at=batch_finished_at,
                batch_wall_seconds=batch_finished_at - batch_started_at,
                workload_http_wall_seconds=math.fsum(row.http_wall_seconds for row in rows),
                local_parse_seconds=math.fsum(
                    row.parsed_finished_at - row.response_finished_at for row in rows
                ),
                actual_max_inflight=_measurement_max_inflight(rows),
                observations=rows,
                question_ids=expected_qids,
                api_answer_sha256_by_qid=api_answer_sha256_by_qid,
                api_answers_sha256=digest(api_answer_sha256_by_qid),
            ),
            timing,
        )

    for _ in range(protocol.native.warmup_requests):
        await request(1)
    measurements: list[ScheduledMeasurement] = []
    timings: list[ScheduledGatewayTiming] = []
    api_answer_digests: list[ScheduledGatewayAnswerDigest] = []
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
                api_answer_digests.append(
                    ScheduledGatewayAnswerDigest(
                        api_answer_sha256_by_qid=measured.api_answer_sha256_by_qid,
                        api_answers_sha256=measured.api_answers_sha256,
                    )
                )
    return ScheduledGatewayRun(
        protocol_sha256=digest(protocol.native.model_dump()),
        server=server,
        repeats=protocol.native.repeats,
        output_tolerance=protocol.native.output_tolerance,
        media_prefix_end=protocol.native.media_prefix_end,
        concurrency_limit=protocol.question_concurrency,
        launch_max_num_seqs=max_num_seqs,
        measurements=measurements,
        gateway_request_sha256=digest(protocol.request),
        gateway_configuration_sha256=protocol.gateway_configuration_sha256,
        scheduled_gateway_timings=timings,
        api_answer_digests=api_answer_digests,
        gateway_question_ids=list(protocol.request["questions"]),
    )


def _measurement_max_inflight(rows: list[ScheduledObservation]) -> int:
    events: list[tuple[float, int]] = []
    for row in rows:
        events.append((row.request_started_at, 1))
        events.append((row.response_finished_at, -1))
    current = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def assess_scheduled_gateway(
    on: ScheduledGatewayRun,
    off: ScheduledGatewayRun,
) -> dict[str, Any]:
    if (
        on.gateway_configuration_sha256 != off.gateway_configuration_sha256
        or on.gateway_question_ids != off.gateway_question_ids
        or on.gateway_request_sha256 != off.gateway_request_sha256
        or on.transport_scope != off.transport_scope
    ):
        raise ValueError("gateway configuration or frozen public request changed")
    result = assess_scheduled(on, off)
    timing_rows = []
    for run in (on, off):
        for measurement, timing, answer_digest in zip(
            run.measurements,
            run.scheduled_gateway_timings,
            run.api_answer_digests,
            strict=True,
        ):
            timing_rows.append(
                {
                    "cache_enabled": run.server.cache_enabled,
                    "repeat": measurement.repeat,
                    "condition": measurement.condition,
                    "size": measurement.size,
                    "gateway_client_wall_seconds": measurement.batch_wall_seconds,
                    "server_timing_ms": timing.model_dump(),
                    "native_http_sum_seconds": measurement.workload_http_wall_seconds,
                    "actual_max_inflight": measurement.actual_max_inflight,
                    "api_answers_sha256": answer_digest.api_answers_sha256,
                }
            )
    result.update(
        timing_rows=timing_rows,
        timing_boundary=(
            "actual in-process gateway ASGI scheduled wall; work intervals are diagnostic"
        ),
        gateway_configuration_sha256=on.gateway_configuration_sha256,
        gateway_request_sha256=on.gateway_request_sha256,
        transport_scope=on.transport_scope,
    )
    return result
