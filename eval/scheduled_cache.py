"""Bounded native prefix-cache scheduling measurements; never a strict sequential claim."""

from __future__ import annotations

import asyncio
import json
import math
import statistics
from collections.abc import Callable
from typing import Any, Literal, Self

import httpx
from pydantic import Field, model_validator

from data.schema import Contract, Sha256
from eval.prefix_cache import (
    PROBABILITY_THRESHOLD_EXCLUSIVE,
    CacheProtocol,
    CacheServer,
    Observation,
    _distribution,
    _update_distribution_deltas,
    digest,
)
from serve.vllm_chat import parse_chat_logprobs

ScheduledCondition = Literal["cold", "warm"]
ScheduledSize = Literal[1, 10]
SCHEDULED_MODE: Literal["bounded-scheduled-v1"] = "bounded-scheduled-v1"


def _launch_arg_int(command: list[str], flag: str) -> int | None:
    prefix = flag + "="
    for index, arg in enumerate(command):
        if arg == flag and index + 1 < len(command):
            try:
                return int(command[index + 1])
            except ValueError:
                return None
        if arg.startswith(prefix):
            try:
                return int(arg.removeprefix(prefix))
            except ValueError:
                return None
    return None


class ScheduledObservation(Observation):
    request_started_at: float = Field(ge=0)
    response_finished_at: float = Field(gt=0)
    parsed_finished_at: float = Field(gt=0)
    raw_logprobs_sha256: Sha256
    native_metrics: dict[str, float]

    @model_validator(mode="after")
    def ordered_events(self) -> Self:
        if not (
            self.request_started_at < self.response_finished_at <= self.parsed_finished_at
            and math.isclose(
                self.http_wall_seconds,
                self.response_finished_at - self.request_started_at,
                rel_tol=0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("request event times must describe the HTTP wall")
        if self.native_metrics != self.metrics:
            raise ValueError("native metrics must match the observation metrics")
        if self.raw_logprobs_sha256 != digest(self.raw_logprobs):
            raise ValueError("raw logprobs digest mismatch")
        return self


def _event_sweep_max_inflight(rows: list[ScheduledObservation]) -> int:
    events: list[tuple[float, int]] = []
    for row in rows:
        events.append((row.request_started_at, 1))
        events.append((row.response_finished_at, -1))
    inflight = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        inflight += delta
        if inflight < 0:
            raise ValueError("request event sweep became negative")
        maximum = max(maximum, inflight)
    if inflight != 0:
        raise ValueError("request event sweep did not close")
    return maximum


class ScheduledMeasurement(Contract):
    repeat: int = Field(ge=0)
    condition: ScheduledCondition
    size: ScheduledSize
    concurrency_limit: int = Field(ge=2, le=10)
    batch_started_at: float = Field(ge=0)
    batch_finished_at: float = Field(gt=0)
    batch_wall_seconds: float = Field(gt=0)
    workload_http_wall_seconds: float = Field(gt=0)
    local_parse_seconds: float = Field(ge=0)
    actual_max_inflight: int = Field(ge=1, le=10)
    observations: list[ScheduledObservation]

    @model_validator(mode="after")
    def complete_bounded_measurement(self) -> Self:
        if [row.case_index for row in self.observations] != list(range(self.size)):
            raise ValueError("scheduled measurements must be returned in case-index order")
        if len(self.observations) != self.size:
            raise ValueError("scheduled measurement size mismatch")
        if self.actual_max_inflight > min(self.concurrency_limit, self.size):
            raise ValueError("actual inflight exceeded configured bound")
        if self.actual_max_inflight != _event_sweep_max_inflight(self.observations):
            raise ValueError("actual inflight does not match request event sweep")
        if self.size == 10:
            first_done = self.observations[0].parsed_finished_at
            if any(row.request_started_at < first_done for row in self.observations[1:]):
                raise ValueError("case 0 must complete before scheduled fanout starts")
        if any(
            row.request_started_at < self.batch_started_at
            or row.parsed_finished_at > self.batch_finished_at
            for row in self.observations
        ):
            raise ValueError("batch wall must wrap all request events and parsing")
        if not math.isclose(
            self.batch_wall_seconds,
            self.batch_finished_at - self.batch_started_at,
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError("batch wall must use the outer measurement clock")
        if not math.isclose(
            self.workload_http_wall_seconds,
            math.fsum(row.http_wall_seconds for row in self.observations),
            rel_tol=0,
            abs_tol=1e-6,
        ):
            raise ValueError("workload wall must remain separate from batch wall")
        return self


class ScheduledCacheRun(Contract):
    protocol_sha256: Sha256
    server: CacheServer
    repeats: int = Field(ge=2, le=100)
    output_tolerance: float = Field(ge=0, le=1e-3)
    media_prefix_end: int = Field(gt=0)
    concurrency_limit: int = Field(ge=2, le=10)
    launch_max_num_seqs: int = Field(ge=2)
    measurements: list[ScheduledMeasurement]
    mode: Literal["bounded-scheduled-v1"] = SCHEDULED_MODE
    scope: Literal["scheduled-prefix-cache-evidence-only"] = "scheduled-prefix-cache-evidence-only"

    @model_validator(mode="after")
    def launch_supports_bound(self) -> Self:
        if self.launch_max_num_seqs < self.concurrency_limit:
            raise ValueError("server --max-num-seqs must cover the configured concurrency")
        command_max = _launch_arg_int(self.server.launch_command, "--max-num-seqs")
        if command_max != self.launch_max_num_seqs:
            raise ValueError("launch_max_num_seqs must match the server launch command")
        for measurement in self.measurements:
            if measurement.concurrency_limit != self.concurrency_limit:
                raise ValueError("measurement concurrency must match the run concurrency")
        return self


class _InflightCounter:
    def __init__(self) -> None:
        self.current = 0
        self.maximum = 0
        self.lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self.lock:
            self.current += 1
            self.maximum = max(self.maximum, self.current)

    async def exit(self) -> None:
        async with self.lock:
            self.current -= 1


async def collect_scheduled(
    protocol: CacheProtocol,
    server: CacheServer,
    client: httpx.AsyncClient,
    *,
    allow_reset: bool = False,
    concurrency_limit: int = 10,
    clock: Callable[[], float],
) -> ScheduledCacheRun:
    """Collect one bounded native run against an already-started exclusive GPU server."""
    if type(concurrency_limit) is not int or not 2 <= concurrency_limit <= 10:
        raise ValueError("scheduled concurrency must be an integer between 2 and 10")
    if not allow_reset or not server.exclusive_local_test:
        raise ValueError("explicit reset permission and exclusive test attestation required")
    max_num_seqs = _launch_arg_int(server.launch_command, "--max-num-seqs")
    if max_num_seqs is None or max_num_seqs < concurrency_limit:
        raise ValueError("server --max-num-seqs must cover the configured concurrency")
    base = server.base_url.rstrip("/")

    async def reset() -> None:
        response = await client.post(
            base + "/reset_prefix_cache",
            params={
                "reset_running_requests": "false",
                "reset_external": "false",
            },
        )
        response.raise_for_status()
        if response.json() != {"success": True}:
            raise ValueError("prefix reset did not succeed")
        for endpoint in ("reset_mm_cache", "reset_encoder_cache"):
            response = await client.post(base + "/" + endpoint)
            response.raise_for_status()

    async def request(index: int, counter: _InflightCounter) -> ScheduledObservation:
        prep_start = clock()
        payload = protocol.payload(server.model, index)
        encoded = json.dumps(payload, allow_nan=False).encode()
        prep_seconds = clock() - prep_start
        started = clock()
        await counter.enter()
        try:
            response = await client.post(
                base + "/v1/chat/completions",
                content=encoded,
                headers={"Content-Type": "application/json"},
            )
        finally:
            await counter.exit()
        response_finished = clock()
        response.raise_for_status()
        body = response.json()
        readout = parse_chat_logprobs(
            body,
            model=server.model,
            answer_token_ids=protocol.answer_token_ids,
            expected_prompt_ids=protocol.cases[index].prompt_token_ids,
        )
        cached = body["usage"]["prompt_tokens_details"]["cached_tokens"]
        if type(cached) is not int or not 0 <= cached <= len(readout.prompt_token_ids):
            raise ValueError("invalid native cached_tokens")
        if not server.cache_enabled and cached != 0:
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
        parsed_finished = clock()
        return ScheduledObservation(
            case_index=index,
            input_sha256=digest(payload),
            http_wall_seconds=response_finished - started,
            payload_preparation_seconds=prep_seconds,
            cached_tokens=cached,
            prompt_tokens=len(readout.prompt_token_ids),
            raw_logprobs=readout.raw_logprobs,
            metrics=metrics,
            request_started_at=started,
            response_finished_at=response_finished,
            parsed_finished_at=parsed_finished,
            raw_logprobs_sha256=digest(readout.raw_logprobs),
            native_metrics=metrics,
        )

    async def measure(
        repeat: int, condition: ScheduledCondition, size: ScheduledSize
    ) -> ScheduledMeasurement:
        await reset()
        if condition == "warm":
            await request(0, _InflightCounter())
        counter = _InflightCounter()
        batch_started_at = clock()
        try:
            if size == 1:
                rows = [await request(0, counter)]
            else:
                first = await request(0, counter)
                semaphore = asyncio.Semaphore(concurrency_limit)

                async def bounded(index: int) -> ScheduledObservation:
                    async with semaphore:
                        return await request(index, counter)

                tasks = [asyncio.create_task(bounded(index)) for index in range(1, 10)]
                try:
                    rest = await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
                rows = [first, *rest]
        finally:
            batch_finished_at = clock()
        if condition == "cold" and rows[0].cached_tokens != 0:
            raise ValueError("cold first request unexpectedly reused KV")
        return ScheduledMeasurement(
            repeat=repeat,
            condition=condition,
            size=size,
            concurrency_limit=concurrency_limit,
            batch_started_at=batch_started_at,
            batch_finished_at=batch_finished_at,
            batch_wall_seconds=batch_finished_at - batch_started_at,
            workload_http_wall_seconds=math.fsum(row.http_wall_seconds for row in rows),
            local_parse_seconds=math.fsum(
                row.parsed_finished_at - row.response_finished_at for row in rows
            ),
            actual_max_inflight=counter.maximum,
            observations=rows,
        )

    for _ in range(protocol.warmup_requests):
        await request(0, _InflightCounter())
    measurements: list[ScheduledMeasurement] = []
    for repeat in range(protocol.repeats):
        for condition in ("cold", "warm"):
            for size in (1, 10):
                measurements.append(await measure(repeat, condition, size))
    return ScheduledCacheRun(
        protocol_sha256=digest(protocol.model_dump()),
        server=server,
        repeats=protocol.repeats,
        output_tolerance=protocol.output_tolerance,
        media_prefix_end=protocol.media_prefix_end,
        concurrency_limit=concurrency_limit,
        launch_max_num_seqs=max_num_seqs,
        measurements=measurements,
    )


def assess_scheduled(on: ScheduledCacheRun, off: ScheduledCacheRun) -> dict[str, Any]:
    """Assess only the explicit bounded scheduled policy."""
    if not on.server.cache_enabled or off.server.cache_enabled:
        raise ValueError("one scheduled on and one scheduled off run required")
    if (
        on.server.run_id == off.server.run_id
        or on.server.common_identity() != off.server.common_identity()
        or on.protocol_sha256 != off.protocol_sha256
        or on.repeats != off.repeats
        or on.output_tolerance != off.output_tolerance
        or on.media_prefix_end != off.media_prefix_end
        or on.concurrency_limit != off.concurrency_limit
        or on.launch_max_num_seqs != off.launch_max_num_seqs
    ):
        raise ValueError("separate matching scheduled server starts and protocol required")
    reference: dict[int, ScheduledObservation] = {}
    failures: list[str] = []
    groups: list[dict[str, Any]] = []
    max_differences = dict.fromkeys(("raw_probabilities", "joint", "conditional", "p_U"), 0.0)
    for run in (on, off):
        measurements = {(m.condition, m.size, m.repeat): m for m in run.measurements}
        expected = {
            (condition, size, repeat)
            for condition in ("cold", "warm")
            for size in (1, 10)
            for repeat in range(run.repeats)
        }
        if set(measurements) != expected or len(run.measurements) != len(expected):
            raise ValueError("missing or duplicate scheduled measurement")
        for condition in ("cold", "warm"):
            ratios = []
            for repeat in range(run.repeats):
                single = measurements[condition, 1, repeat]
                ten = measurements[condition, 10, repeat]
                ratios.append(ten.batch_wall_seconds / single.batch_wall_seconds)
                for measurement in (single, ten):
                    if measurement.actual_max_inflight > run.concurrency_limit:
                        failures.append("inflight_exceeded_limit")
                    if measurement.size == 10:
                        first_done = measurement.observations[0].parsed_finished_at
                        if any(
                            row.request_started_at < first_done
                            for row in measurement.observations[1:]
                        ):
                            failures.append("fanout_started_before_primer_finished")
                    rows = measurement.observations
                    if condition == "cold" and rows[0].cached_tokens != 0:
                        failures.append("cold_prefix_not_empty")
                    if not run.server.cache_enabled and any(row.cached_tokens for row in rows):
                        failures.append("cache_off_hit")
                    if run.server.cache_enabled:
                        required = rows if condition == "warm" else rows[1:]
                        if any(row.cached_tokens < run.media_prefix_end for row in required):
                            failures.append("missing_prefix_hit")
                    for row in rows:
                        _distribution(row)
                        ref = reference.setdefault(row.case_index, row)
                        if row.input_sha256 != ref.input_sha256:
                            raise ValueError("case input binding mismatch")
                        if row.raw_logprobs.keys() != ref.raw_logprobs.keys():
                            failures.append("output_changed")
                        else:
                            _update_distribution_deltas(max_differences, ref, row)
            if (
                run.server.cache_enabled
                and condition == "cold"
                and any(ratio > 2 for ratio in ratios)
            ):
                failures.append("ten_to_single_ratio_exceeds_two")
            groups.append(
                {
                    "cache_enabled": run.server.cache_enabled,
                    "condition": condition,
                    "ratios": ratios,
                    "median_ratio": statistics.median(ratios),
                    "maximum_ratio": max(ratios),
                    "acceptance_condition": run.server.cache_enabled and condition == "cold",
                }
            )
    if any(value >= PROBABILITY_THRESHOLD_EXCLUSIVE for value in max_differences.values()):
        failures.append("probability_distribution_changed")
    mock = on.server.evidence_kind == "mock"
    return {
        "status": "failed" if failures else ("mock-validated" if mock else "measured"),
        "exit_code": 1 if failures else 0,
        "assessment_policy": SCHEDULED_MODE,
        "acceptance_timing_condition": "cache-on cold scheduled T10/T1 <= 2",
        "ratio_threshold_inclusive": 2,
        "probability_threshold_exclusive": PROBABILITY_THRESHOLD_EXCLUSIVE,
        "max_distribution_differences": max_differences,
        "statistics": "all cache-on cold paired repeats must pass; warm ratios are diagnostic",
        "groups": groups,
        "failures": sorted(set(failures)),
        "protocol_sha256": on.protocol_sha256,
        "evidence_kind": on.server.evidence_kind,
        "mode": SCHEDULED_MODE,
        "concurrency_limit": on.concurrency_limit,
        "launch_max_num_seqs": on.launch_max_num_seqs,
        "s0_acceptance": False,
        "gateway_wall_seconds": None,
        "media_processor_seconds": None,
        "timing_boundary": (
            "direct native HTTP scheduled batch wall; sum HTTP work and local parsing separate"
        ),
        "scope": "scheduled latency evidence only; strict sequential acceptance unchanged",
    }
