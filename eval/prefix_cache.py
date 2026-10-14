"""Frozen sequential prefix-cache measurements; mock evidence never passes S0."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import statistics
from collections.abc import Callable
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import httpx
from pydantic import Field, model_validator

from data.schema import Contract, Nonempty, Sha256
from serve.vllm_chat import parse_chat_logprobs


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


class CacheCase(Contract):
    tail: str = Field(min_length=1)
    prompt_token_ids: list[int] = Field(min_length=2)


class CacheProtocol(Contract):
    """Sensitive local input. Reports retain hashes, never image bytes or text."""

    image_png_base64: Nonempty
    image_sha256: Sha256
    shared_text: str = Field(min_length=1)
    shared_prompt_token_ids: list[int] = Field(min_length=1)
    media_prefix_end: int = Field(gt=0)
    answer_token_ids: dict[str, int]
    cases: list[CacheCase] = Field(min_length=10, max_length=10)
    repeats: int = Field(ge=2, le=100)
    warmup_requests: int = Field(ge=1, le=20)
    seed: int = Field(ge=0)
    output_tolerance: float = Field(ge=0, le=1e-3)

    @model_validator(mode="after")
    def freeze_inputs(self) -> Self:
        try:
            image = base64.b64decode(self.image_png_base64, validate=True)
        except ValueError:
            raise ValueError("invalid canonical PNG encoding") from None
        if not image.startswith(b"\x89PNG\r\n\x1a\n") or (
            hashlib.sha256(image).hexdigest() != self.image_sha256
        ):
            raise ValueError("canonical PNG hash mismatch")
        ids = self.answer_token_ids
        if (
            "U" not in ids
            or not 3 <= len(ids) <= 21
            or len(set(ids.values())) != len(ids)
            or any(type(token) is not int or token < 0 for token in ids.values())
        ):
            raise ValueError("complete unique answer IDs including U required")
        prefix = self.shared_prompt_token_ids
        if self.media_prefix_end > len(prefix):
            raise ValueError("shared prefix must include the complete HF media placeholder range")
        if len({case.tail for case in self.cases}) != 10:
            raise ValueError("ten distinct question tails required")
        for case in self.cases:
            if (
                any(token < 0 for token in [*prefix, *case.prompt_token_ids])
                or case.prompt_token_ids[: len(prefix)] != prefix
                or len(case.prompt_token_ids) <= len(prefix)
            ):
                raise ValueError("all expanded HF inputs must share the frozen prefix")
        if len({tuple(case.prompt_token_ids) for case in self.cases}) != 10:
            raise ValueError("ten distinct expanded HF prompts required")
        return self

    def payload(self, model: str, index: int) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + self.image_png_base64,
                            },
                        },
                        {"type": "text", "text": self.shared_text + self.cases[index].tail},
                    ],
                }
            ],
            "max_completion_tokens": 1,
            "stream": False,
            "n": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": self.seed,
            "logprobs": True,
            "top_logprobs": 0,
            "logprob_token_ids": list(self.answer_token_ids.values()),
            "return_tokens_as_token_ids": True,
            "return_token_ids": True,
            "return_prompt_text": False,
            "add_generation_prompt": True,
            "add_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }


class CacheServer(Contract):
    base_url: Nonempty
    run_id: Nonempty
    model: Nonempty
    model_revision: Nonempty
    weights_manifest_sha256: Sha256
    dtype: Literal["float32", "float16", "bfloat16"]
    gpu_identity: Nonempty
    vllm_version: Literal["0.26.0", "0.31.0"]
    common_engine_config_sha256: Sha256
    launch_command: list[Nonempty] = Field(min_length=2)
    cache_enabled: bool
    exclusive_local_test: bool
    raw_logprobs_confirmed: bool
    evidence_kind: Literal["mock", "gpu"]

    @model_validator(mode="after")
    def attest_launch(self) -> Self:
        url = urlsplit(self.base_url)
        if (
            url.scheme != "http"
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in {"", "/"}
        ):
            raise ValueError("only explicit loopback test endpoints accepted")
        command = self.launch_command
        if any(arg.split("=", 1)[0] in {"--api-key", "--hf-token", "--token"} for arg in command):
            raise ValueError("launch records must not contain credentials")
        expected = "--enable-prefix-caching" if self.cache_enabled else "--no-enable-prefix-caching"
        opposite = "--no-enable-prefix-caching" if self.cache_enabled else "--enable-prefix-caching"
        if expected not in command or opposite in command or not self.raw_logprobs_confirmed:
            raise ValueError("cache/raw mode launch attestation mismatch")
        for flag in ("--enable-prompt-tokens-details", "--enable-per-request-metrics"):
            if flag not in command:
                raise ValueError("native cache and timing metrics must be enabled")
        if "--disable-log-stats" in command:
            raise ValueError("request metrics require statistics")
        if not any(
            command[i : i + 2] == ["--logprobs-mode", "raw_logprobs"]
            for i in range(len(command) - 1)
        ):
            raise ValueError("raw_logprobs launch flag required")
        return self

    def common_identity(self) -> dict[str, Any]:
        return self.model_dump(
            exclude={
                "base_url",
                "run_id",
                "launch_command",
                "cache_enabled",
                "exclusive_local_test",
            }
        )


class Observation(Contract):
    case_index: int = Field(ge=0, le=9)
    input_sha256: Sha256
    http_wall_seconds: float = Field(gt=0)
    payload_preparation_seconds: float = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    prompt_tokens: int = Field(gt=0)
    raw_logprobs: dict[str, float]
    metrics: dict[str, float]

    @model_validator(mode="after")
    def complete_evidence(self) -> Self:
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached tokens cannot exceed prompt length")
        if (
            "U" not in self.raw_logprobs
            or not 3 <= len(self.raw_logprobs) <= 21
            or any(not -9999 < value <= 0 for value in self.raw_logprobs.values())
            or math.fsum(math.exp(value) for value in self.raw_logprobs.values()) > 1 + 1e-6
        ):
            raise ValueError("complete finite full-vocabulary answer readout required")
        if set(self.metrics) != {
            "time_to_first_token_ms",
            "generation_time_ms",
            "queue_time_ms",
        } or any(value < 0 for value in self.metrics.values()):
            raise ValueError("complete nonnegative native timing fields required")
        return self


class Measurement(Contract):
    repeat: int = Field(ge=0)
    condition: Literal["cold", "warm"]
    size: Literal[1, 10]
    wall_seconds: float = Field(gt=0)
    observations: list[Observation]


class CacheRun(Contract):
    protocol_sha256: Sha256
    server: CacheServer
    repeats: int = Field(ge=2, le=100)
    output_tolerance: float = Field(ge=0, le=1e-3)
    media_prefix_end: int = Field(gt=0)
    measurements: list[Measurement]
    scope: Literal["prefix-cache-evidence-only"] = "prefix-cache-evidence-only"


def collect(
    protocol: CacheProtocol,
    server: CacheServer,
    client: httpx.Client,
    *,
    allow_reset: bool = False,
    clock: Callable[[], float],
) -> CacheRun:
    """Only an attested dedicated server may be reset; no launch or concurrent HTTP."""
    if not allow_reset or not server.exclusive_local_test:
        raise ValueError("explicit reset permission and exclusive test attestation required")
    base = server.base_url.rstrip("/")

    def reset() -> None:
        response = client.post(
            base + "/reset_prefix_cache",
            params={
                "reset_running_requests": "false",
                "reset_external": "false",
            },
        )
        response.raise_for_status()
        if response.json() != {"success": True}:
            raise ValueError("prefix reset did not succeed")
        # Native API only acknowledges these resets; record that limitation in runbook.
        for endpoint in ("reset_mm_cache", "reset_encoder_cache"):
            response = client.post(base + "/" + endpoint)
            response.raise_for_status()

    def request(index: int) -> Observation:
        prep_start = clock()
        payload = protocol.payload(server.model, index)
        encoded = json.dumps(payload, allow_nan=False).encode()
        prep_seconds = clock() - prep_start
        started = clock()
        response = client.post(
            base + "/v1/chat/completions",
            content=encoded,
            headers={"Content-Type": "application/json"},
        )
        wall = clock() - started
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
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in metrics.values()
        ):
            raise ValueError("native timing metrics missing or invalid")
        return Observation(
            case_index=index,
            input_sha256=digest(payload),
            http_wall_seconds=wall,
            payload_preparation_seconds=prep_seconds,
            cached_tokens=cached,
            prompt_tokens=len(readout.prompt_token_ids),
            raw_logprobs=readout.raw_logprobs,
            metrics=metrics,
        )

    # Model/HTTP warmup stays outside measurements; reset removes its cached prefixes.
    for _ in range(protocol.warmup_requests):
        request(0)
    measurements = []
    for repeat in range(protocol.repeats):
        for condition in ("cold", "warm"):
            for size in (1, 10):
                reset()
                if condition == "warm":
                    request(0)
                started = clock()
                rows = [request(index) for index in range(size)]
                wall = clock() - started
                if condition == "cold" and rows[0].cached_tokens != 0:
                    raise ValueError("cold first request unexpectedly reused KV")
                measurements.append(
                    Measurement(
                        repeat=repeat,
                        condition=condition,
                        size=size,
                        wall_seconds=wall,
                        observations=rows,
                    )
                )
    return CacheRun(
        protocol_sha256=digest(protocol.model_dump()),
        server=server,
        repeats=protocol.repeats,
        output_tolerance=protocol.output_tolerance,
        media_prefix_end=protocol.media_prefix_end,
        measurements=measurements,
    )


def assess(on: CacheRun, off: CacheRun) -> dict[str, Any]:
    """Every paired repetition is <=2; raw output consistency and KV hits are required."""
    if not on.server.cache_enabled or off.server.cache_enabled:
        raise ValueError("one on and one off run required")
    if (
        on.server.run_id == off.server.run_id
        or on.server.common_identity() != off.server.common_identity()
        or on.protocol_sha256 != off.protocol_sha256
        or on.repeats != off.repeats
        or on.output_tolerance != off.output_tolerance
        or on.media_prefix_end != off.media_prefix_end
    ):
        raise ValueError("separate matching server starts and frozen protocol required")
    reference: dict[int, Observation] = {}
    groups = []
    failures = []
    for run in (on, off):
        measurements = {(m.condition, m.size, m.repeat): m for m in run.measurements}
        expected = {
            (c, n, r) for c in ("cold", "warm") for n in (1, 10) for r in range(run.repeats)
        }
        if set(measurements) != expected or len(run.measurements) != len(expected):
            raise ValueError("missing or duplicate frozen measurement")
        for condition in ("cold", "warm"):
            ratios = []
            for repeat in range(run.repeats):
                single = measurements[condition, 1, repeat]
                ten = measurements[condition, 10, repeat]
                ratios.append(ten.wall_seconds / single.wall_seconds)
                for measurement in (single, ten):
                    rows = measurement.observations
                    if [r.case_index for r in rows] != list(range(measurement.size)):
                        raise ValueError("measurements must contain sequential complete cases")
                    if measurement.wall_seconds < math.fsum(r.http_wall_seconds for r in rows):
                        raise ValueError("total wall shorter than sequential HTTP requests")
                    if condition == "cold" and rows[0].cached_tokens != 0:
                        failures.append("cold_prefix_not_empty")
                    if not run.server.cache_enabled and any(r.cached_tokens for r in rows):
                        failures.append("cache_off_hit")
                    if run.server.cache_enabled:
                        required = rows if condition == "warm" else rows[1:]
                        if any(r.cached_tokens < run.media_prefix_end for r in required):
                            failures.append("missing_prefix_hit")
                    for row in rows:
                        ref = reference.setdefault(row.case_index, row)
                        if row.input_sha256 != ref.input_sha256:
                            raise ValueError("case input binding mismatch")
                        if row.raw_logprobs.keys() != ref.raw_logprobs.keys() or any(
                            abs(row.raw_logprobs[k] - ref.raw_logprobs[k]) > run.output_tolerance
                            for k in row.raw_logprobs
                        ):
                            failures.append("output_changed")
            # on is the acceptance target; off remains a fully reported control.
            if run.server.cache_enabled and any(ratio > 2 for ratio in ratios):
                failures.append("ten_to_single_ratio_exceeds_two")
            groups.append(
                {
                    "cache_enabled": run.server.cache_enabled,
                    "condition": condition,
                    "ratios": ratios,
                    "median_ratio": statistics.median(ratios),
                    "maximum_ratio": max(ratios),
                }
            )
    mock = on.server.evidence_kind == "mock"
    return {
        "status": "failed" if failures else ("mock-validated" if mock else "measured"),
        "exit_code": 1 if failures else 0,
        "ratio_threshold_inclusive": 2,
        "statistics": "all paired repeats must pass",
        "groups": groups,
        "failures": sorted(set(failures)),
        "protocol_sha256": on.protocol_sha256,
        "evidence_kind": on.server.evidence_kind,
        "s0_acceptance": False,
        "gateway_wall_seconds": None,
        "media_processor_seconds": None,
        "timing_boundary": (
            "direct native HTTP wall; payload preparation and engine metrics separate"
        ),
        "scope": "latency evidence only; stage dependencies and provenance require review",
    }
