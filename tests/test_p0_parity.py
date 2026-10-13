"""CPU protocol/strict gate tests; no test here establishes real GPU parity."""

import copy
import json
import math
from pathlib import Path

import httpx
import pytest

from scripts import p0_parity as parity


def examples():
    return [{"id": str(i), "input_ids": [1, i + 2], "token_ids": [10, 11, 12]} for i in range(100)]


def rows():
    return [parity.row(example, [-2.0, -3.0, -4.0]) for example in examples()]


def test_reference_requires_exact_identity_and_provenance():
    provenance = {"mode": "base", "data": "hash", "adapter_hashes": None}
    reference = {
        "status": "reference-ready",
        "action": "hf-reference",
        "provenance": provenance,
        "rows": rows(),
    }
    assert len(parity.validate_reference(reference, provenance, examples())) == 100
    for key, value in (
        ("mode", "adapter"),
        ("data", "stale"),
        ("adapter_hashes", {"weights": "changed"}),
    ):
        stale = {**provenance, key: value}
        with pytest.raises(ValueError, match="provenance"):
            parity.validate_reference(reference, stale, examples())
    reference["rows"][0]["input_sha256"] = "changed"
    with pytest.raises(ValueError, match="row identity"):
        parity.validate_reference(reference, provenance, examples())


def test_missing_rows_and_tokens_fail_closed():
    with pytest.raises(ValueError, match="100"):
        parity.compare_rows(rows()[:-1], rows())
    with pytest.raises(ValueError, match="complete"):
        parity.row(examples()[0], [-2.0, -3.0])
    for value in (math.nan, math.inf, -10000.0, 0.1, True):
        with pytest.raises(ValueError):
            parity.row(examples()[0], [-2.0, -3.0, value])
    changed = rows()
    changed[0]["token_ids"] = [10, 11, 13]
    with pytest.raises(ValueError, match="token mismatch"):
        parity.compare_rows(rows(), changed)


def test_strict_threshold_and_u_are_separate():
    baseline = rows()
    assert parity.compare_rows(baseline, baseline)["status"] == "passed"
    for field in ("conditional", "p_U"):
        changed = copy.deepcopy(baseline)
        if field == "conditional":
            changed[0][field][0] += 0.002
        else:
            changed[0][field] += 0.002
        assert parity.compare_rows(baseline, changed)["status"] == "failed"
    # Use zero to make the exact boundary representable as the same float constant.
    baseline[0]["p_U"] = 0.0
    changed = copy.deepcopy(baseline)
    changed[0]["p_U"] = parity.THRESHOLD
    assert parity.compare_rows(baseline, changed)["status"] == "failed"


def test_vllm_requests_whole_legal_distribution_by_token_id():
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        body = json.loads(request.content)
        assert body["logprob_token_ids"] == [10, 11, 12]
        assert body["prompt"] == [1, 2]
        assert body["temperature"] == body["top_p"] == 1
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 2},
                "choices": [
                    {
                        "logprobs": {
                            "top_logprobs": [
                                {"token_id:10": -2.0, "token_id:11": -3.0, "token_id:12": -4.0}
                            ]
                        }
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = parity.vllm_rows(client, "http://localhost", "base", examples()[:1])
    assert result == rows()[:1]
    assert "input_ids" not in result[0]


def test_vllm_truncated_top_k_is_rejected():
    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "base"}]})
        return httpx.Response(
            200,
            json={
                "usage": {"prompt_tokens": 2},
                "choices": [{"logprobs": {"top_logprobs": [{"token_id:10": -2.0}]}}],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(KeyError):
            parity.vllm_rows(client, "http://localhost", "base", examples()[:1])


def test_hf_refuses_runtime_before_model_loading(monkeypatch):
    def blocked():
        raise ValueError("CUDA required")

    monkeypatch.setattr(parity, "runtime_requirements", blocked)
    with pytest.raises(ValueError, match="CUDA"):
        parity.hf_reference(parity.SmokeConfig(), Path("missing"), None, examples(), {})


def test_compare_cli_rejects_stale_server_without_network(tmp_path, monkeypatch):
    provenance = {"mode": "base", "adapter_hashes": None}
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "status": "reference-ready",
                "action": "hf-reference",
                "provenance": provenance,
                "rows": rows(),
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_text("{}")
    server = tmp_path / "server.json"
    server.write_text(
        json.dumps(
            {
                "provenance": {"mode": "adapter", "adapter_hashes": {"weights": "wrong"}},
                "logprobs_mode": "raw_logprobs",
                "dtype": "bfloat16",
                "served_model": "decision",
                "launch_command": "vllm serve ...",
            }
        )
    )
    monkeypatch.setattr(parity, "prepare", lambda *args: (provenance, examples()))

    def forbidden(*args, **kwargs):
        pytest.fail("stale provenance must not make HTTP calls")

    monkeypatch.setattr(parity.httpx, "Client", forbidden)
    report = tmp_path / "report.json"
    result = parity.main(
        [
            "compare-vllm",
            "--config",
            str(config),
            "--model-dir",
            str(tmp_path),
            "--asset-manifest",
            "unused",
            "--data",
            "unused",
            "--reference",
            str(reference),
            "--server-record",
            str(server),
            "--report",
            str(report),
        ]
    )
    assert result == 1
    saved = json.loads(report.read_text())
    assert saved["status"] == "blocked"
    assert saved["error_type"] == "ValueError"
    assert "rows" not in saved
