"""Synthetic MockTransport and monotonic-clock evidence; never a cache latency claim."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from typing import Any

import httpx
import pytest

from eval.prefix_cache import CacheProtocol, CacheRun, CacheServer, assess, collect


def protocol() -> CacheProtocol:
    png = b"\x89PNG\r\n\x1a\nSYNTHETIC_NOT_A_DECODE_TEST"
    return CacheProtocol(
        image_png_base64=base64.b64encode(png).decode(),
        image_sha256=hashlib.sha256(png).hexdigest(),
        shared_text="private prefix:",
        shared_prompt_token_ids=[10, 20],
        media_prefix_end=2,
        answer_token_ids={"A": 1, "B": 2, "U": 3},
        cases=[{"tail": f"tail{i}", "prompt_token_ids": [10, 20, 100 + i]} for i in range(10)],
        repeats=2,
        warmup_requests=1,
        seed=42,
        output_tolerance=0.0,
    )


def server(enabled: bool) -> CacheServer:
    return CacheServer(
        base_url="http://127.0.0.1:8002",
        run_id="on" if enabled else "off",
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


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.000001
        return self.now


class Backend:
    def __init__(self, enabled: bool, clock: FakeClock) -> None:
        self.enabled = enabled
        self.clock = clock
        self.resets = 0
        self.cached = False
        self.paths: list[str] = []
        self.sequence: list[int] = []
        self.failure: str | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/reset_prefix_cache":
            assert request.url.params["reset_running_requests"] == "false"
            assert request.url.params["reset_external"] == "false"
            self.resets += 1
            self.cached = False
            return httpx.Response(200, json={"success": self.failure != "reset"})
        if request.url.path in {"/reset_mm_cache", "/reset_encoder_cache"}:
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert payload["top_k"] == -1 and payload["top_logprobs"] == 0
        assert payload["return_prompt_text"] is False
        assert payload["logprob_token_ids"] == [1, 2, 3]
        index = int(payload["messages"][0]["content"][-1]["text"].split("tail")[1])
        self.sequence.append(index)
        hit = 2 if self.cached and self.enabled else 0
        self.cached = True
        # Intentionally engineered mock timings to test arithmetic, not engine speed.
        self.clock.now += 0.1 if self.resets % 2 else 0.01
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
                                    {"token": f"token_id:{i}", "logprob": -float(i + 1)}
                                    for i in (1, 2, 3)
                                ],
                            }
                        ]
                    },
                }
            ],
        }
        if self.failure == "missing_u":
            body["choices"][0]["logprobs"]["content"][0]["top_logprobs"].pop()
        if self.failure == "missing_metrics":
            body["metrics"]["time_to_first_token_ms"] = None
        if self.failure == "bad_prefix":
            body["prompt_token_ids"] = [999, 20, 100 + index]
        return httpx.Response(200, json=body)


def run(enabled: bool, failure: str | None = None) -> tuple[CacheRun, Backend]:
    clock = FakeClock()
    backend = Backend(enabled, clock)
    backend.failure = failure
    with httpx.Client(transport=httpx.MockTransport(backend)) as client:
        result = collect(protocol(), server(enabled), client, allow_reset=True, clock=clock)
    return result, backend


def test_sequential_collection_and_mock_scope() -> None:
    on, backend = run(True)
    off, _ = run(False)
    result = assess(on, off)
    assert result["status"] == "mock-validated"
    assert result["s0_acceptance"] is False
    assert backend.resets == 8
    assert backend.sequence == [0] + ([0] + list(range(10)) + [0, 0] + [0] + list(range(10))) * 2
    assert len(on.measurements) == 8


def test_no_reset_or_network_without_permission() -> None:
    calls: list[httpx.Request] = []
    with httpx.Client(transport=httpx.MockTransport(lambda r: calls.append(r))) as client:
        with pytest.raises(ValueError, match="explicit reset permission"):
            collect(protocol(), server(True), client, clock=FakeClock())
    assert calls == []


@pytest.mark.parametrize("failure", ["reset", "missing_u", "missing_metrics", "bad_prefix"])
def test_protocol_failures_are_not_measurements(failure: str) -> None:
    with pytest.raises(ValueError):
        run(True, failure)


def test_output_drift_and_ratio_failure() -> None:
    on, _ = run(True)
    off, _ = run(False)
    on.measurements[1].wall_seconds = 10.0
    on.measurements[1].observations[0].raw_logprobs["U"] = -4.2
    result = assess(on, off)
    assert "ten_to_single_ratio_exceeds_two" in result["failures"]
    assert "output_changed" in result["failures"]


def test_original_product_cold_policy_ignores_warm_ratio_and_uses_probability_deltas() -> None:
    on, _ = run(True)
    off, _ = run(False)
    warm_single = next(
        m for m in on.measurements if (m.condition, m.size, m.repeat) == ("warm", 1, 0)
    )
    warm_ten = next(
        m for m in on.measurements if (m.condition, m.size, m.repeat) == ("warm", 10, 0)
    )
    warm_ten.wall_seconds = warm_single.wall_seconds * 3
    on.measurements[1].observations[0].raw_logprobs["A"] -= 0.0001

    strict = assess(on, off)
    assert strict["status"] == "failed"
    assert strict["assessment_policy"] == "strict-sequential-v1"
    assert "ten_to_single_ratio_exceeds_two" in strict["failures"]
    assert "output_changed" in strict["failures"]

    original_product = assess(on, off, policy="original-product-cold-v2")
    assert original_product["status"] == "mock-validated"
    assert original_product["assessment_policy"] == "original-product-cold-v2"
    assert original_product["acceptance_timing_condition"].startswith("cache-on cold")
    assert original_product["probability_threshold_exclusive"] == 0.001
    assert original_product["max_distribution_differences"]["raw_probabilities"] < 0.001
    warm_group = next(
        group
        for group in original_product["groups"]
        if group["cache_enabled"] and group["condition"] == "warm"
    )
    assert warm_group["maximum_ratio"] > 2
    assert warm_group["acceptance_condition"] is False


def test_original_product_probability_compare_uses_stable_label_order() -> None:
    on, _ = run(True)
    off, _ = run(False)
    row = on.measurements[1].observations[0]
    row.raw_logprobs = {
        "U": row.raw_logprobs["U"],
        "B": row.raw_logprobs["B"],
        "A": row.raw_logprobs["A"],
    }

    result = assess(on, off, policy="original-product-cold-v2")
    assert result["status"] == "mock-validated"
    assert result["max_distribution_differences"] == {
        "raw_probabilities": 0.0,
        "joint": 0.0,
        "conditional": 0.0,
        "p_U": 0.0,
    }


def test_original_product_cold_policy_still_fails_cold_ratio() -> None:
    on, _ = run(True)
    off, _ = run(False)
    cold_single = next(
        m for m in on.measurements if (m.condition, m.size, m.repeat) == ("cold", 1, 0)
    )
    cold_ten = next(
        m for m in on.measurements if (m.condition, m.size, m.repeat) == ("cold", 10, 0)
    )
    cold_ten.wall_seconds = cold_single.wall_seconds * 3

    assert assess(on, off)["status"] == "failed"
    original_product = assess(on, off, policy="original-product-cold-v2")
    assert original_product["status"] == "failed"
    assert "ten_to_single_ratio_exceeds_two" in original_product["failures"]


@pytest.mark.parametrize("field", ["raw_probabilities", "joint", "conditional", "p_U"])
def test_original_product_probability_drift_is_strictly_exclusive(field: str) -> None:
    on, _ = run(True)
    off, _ = run(False)
    row = on.measurements[1].observations[0]
    if field == "p_U":
        row.raw_logprobs["U"] = math.log(math.exp(row.raw_logprobs["U"]) + 0.0011)
    elif field == "conditional":
        row.raw_logprobs["A"] += 0.02
    elif field == "joint":
        row.raw_logprobs["B"] += 0.02
    else:
        row.raw_logprobs["A"] = math.log(math.exp(row.raw_logprobs["A"]) + 0.0011)

    result = assess(on, off, policy="original-product-cold-v2")
    assert result["status"] == "failed"
    assert "probability_distribution_changed" in result["failures"]
    assert result["max_distribution_differences"][field] >= 0.001


@pytest.mark.parametrize("bad", [0.1, -10000.0, float("nan")])
def test_original_product_rejects_invalid_raw_values_even_when_all_rows_match(bad: float) -> None:
    on, _ = run(True)
    off, _ = run(False)
    for run_result in (on, off):
        for measurement in run_result.measurements:
            for row in measurement.observations:
                row.raw_logprobs["A"] = bad

    with pytest.raises(ValueError, match="raw logprobs"):
        assess(on, off, policy="original-product-cold-v2")


def test_original_product_rejects_raw_probability_mass_even_when_all_rows_match() -> None:
    on, _ = run(True)
    off, _ = run(False)
    for run_result in (on, off):
        for measurement in run_result.measurements:
            for row in measurement.observations:
                row.raw_logprobs = {"A": -0.1, "B": -0.1, "U": -0.1}

    with pytest.raises(ValueError, match="full vocabulary probability mass"):
        assess(on, off, policy="original-product-cold-v2")


def test_missing_repetition_cannot_change_denominator() -> None:
    on, _ = run(True)
    off, _ = run(False)
    on.measurements.pop()
    with pytest.raises(ValueError, match="missing or duplicate"):
        assess(on, off)


def test_no_hits_and_false_cold_cache_fail() -> None:
    on, _ = run(True)
    off, _ = run(False)
    on.measurements[1].observations[1].cached_tokens = 0
    on.measurements[0].observations[0].cached_tokens = 2
    result = assess(on, off)
    assert "missing_prefix_hit" in result["failures"]
    assert "cold_prefix_not_empty" in result["failures"]


def test_identity_and_endpoint_guards() -> None:
    record = server(True).model_dump()
    record["base_url"] = "http://example.com"
    with pytest.raises(ValueError, match="loopback"):
        CacheServer.model_validate(record)
    on, _ = run(True)
    off, _ = run(False)
    off.server.dtype = "float32"
    with pytest.raises(ValueError, match="matching server starts"):
        assess(on, off)


def test_input_hash_and_common_prefix_required() -> None:
    data = protocol().model_dump()
    data["cases"][2]["prompt_token_ids"] = [10, 999, 102]
    with pytest.raises(ValueError, match="frozen prefix"):
        CacheProtocol.model_validate(data)
    data = protocol().model_dump()
    data["image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="PNG hash mismatch"):
        CacheProtocol.model_validate(data)


def test_threshold_is_inclusive_and_each_repeat_is_required() -> None:
    on, _ = run(True)
    off, _ = run(False)
    for condition in ("cold", "warm"):
        for repeat in range(on.repeats):
            single = next(
                m
                for m in on.measurements
                if (m.condition, m.size, m.repeat) == (condition, 1, repeat)
            )
            ten = next(
                m
                for m in on.measurements
                if (m.condition, m.size, m.repeat) == (condition, 10, repeat)
            )
            ten.wall_seconds = single.wall_seconds * 2
    assert assess(on, off)["status"] == "mock-validated"
    on.measurements[-1].wall_seconds += 0.000001
    assert assess(on, off)["status"] == "failed"


def test_cli_compare_accepts_original_product_policy(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.s0_prefix_cache import main

    on, _ = run(True)
    off, _ = run(False)
    warm_single = next(
        m for m in on.measurements if (m.condition, m.size, m.repeat) == ("warm", 1, 0)
    )
    warm_ten = next(
        m for m in on.measurements if (m.condition, m.size, m.repeat) == ("warm", 10, 0)
    )
    warm_ten.wall_seconds = warm_single.wall_seconds * 3
    on_path = tmp_path / "on.json"
    off_path = tmp_path / "off.json"
    report_path = tmp_path / "compare.json"
    on_path.write_text(json.dumps({"run": on.model_dump()}, allow_nan=False))
    off_path.write_text(json.dumps({"run": off.model_dump()}, allow_nan=False))
    monkeypatch.setattr(
        "sys.argv",
        [
            "s0-prefix-cache",
            "compare",
            "--on",
            str(on_path),
            "--off",
            str(off_path),
            "--assessment-policy",
            "original-product-cold-v2",
            "--report",
            str(report_path),
        ],
    )

    assert main() == 0
    report = json.loads(report_path.read_text())
    assert report["assessment_policy"] == "original-product-cold-v2"
    assert report["status"] == "mock-validated"


def test_cli_validation_is_network_free_and_failure_keeps_safe_evidence(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.s0_prefix_cache import main

    protocol_path = tmp_path / "protocol.json"
    server_path = tmp_path / "server.json"
    report_path = tmp_path / "report.json"
    protocol_path.write_text(protocol().model_dump_json())
    server_path.write_text(server(True).model_dump_json())

    def forbidden_client(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("validation must not create an HTTP client")

    monkeypatch.setattr(httpx, "Client", forbidden_client)
    monkeypatch.setattr(
        "sys.argv",
        [
            "s0-prefix-cache",
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
    assert "private prefix" not in report_path.read_text()
    assert report["s0_acceptance"] is False
    protocol_path.write_text('{"image_png_base64":"PRIVATE-SENTINEL"}')
    report_path.unlink()
    assert main() == 1
    assert "PRIVATE-SENTINEL" not in report_path.read_text()
    assert json.loads(report_path.read_text())["error_type"] == "ValidationError"
