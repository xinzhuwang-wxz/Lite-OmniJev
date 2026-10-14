from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from eval.multimodal_parity import compare_rows
from scripts import s0_parity


class _TensorLike:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _InputIds:
    def __init__(self, values):
        self._row = _TensorLike(values)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return self._row


def _case(i: int):
    return SimpleNamespace(id=f"case-{i:03d}")


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _prepared(i: int):
    messages = [{"role": "user", "content": [{"type": "text", "text": f"case {i}"}]}]
    media_kwargs = {}
    answer_token_ids = {"A": 11, "B": 22, "U": 33}
    return SimpleNamespace(
        answer_token_ids=answer_token_ids,
        inputs={"input_ids": _InputIds([1000 + i, 2000 + i])},
        http_messages=messages,
        http_media_io_kwargs=media_kwargs,
        summary={
            "input_profile": "canonical-http-v1",
            "prompt_sha256": f"prompt-{i}",
            "http_messages_sha256": _sha(messages),
            "http_media_io_kwargs_sha256": _sha(media_kwargs),
            "input_tensors": {
                "input_ids": {"sha256": f"ids-{i}", "shape": [1, 2], "dtype": "int64"}
            },
            "answer_token_mapping_sha256": _sha(answer_token_ids),
            "media_sha256": f"media-{i}",
            "canonical_media": [],
        },
    )


def _response_for(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content.decode("utf-8"))
    text = payload["messages"][0]["content"][0]["text"]
    index = int(text.rsplit(" ", 1)[1])
    prompt_ids = [1000 + index, 2000 + index]
    requested_ids = payload["logprob_token_ids"]
    return httpx.Response(
        200,
        json={
            "id": "fake",
            "object": "chat.completion",
            "created": 1,
            "model": payload["model"],
            "prompt_token_ids": prompt_ids,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": 1,
                "total_tokens": len(prompt_ids) + 1,
            },
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "A"},
                    "logprobs": {
                        "content": [
                            {
                                "token": f"token_id:{requested_ids[0]}",
                                "logprob": -2.0,
                                "top_logprobs": [
                                    {"token": f"token_id:{requested_ids[0]}", "logprob": -2.0},
                                    {"token": f"token_id:{requested_ids[1]}", "logprob": -3.0},
                                    {"token": f"token_id:{requested_ids[2]}", "logprob": -4.0},
                                ],
                            }
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
        },
    )


def test_row_from_prepared_binds_input_media_and_k_plus_u_probabilities():
    case = _case(1)
    prepared = _prepared(1)
    row = s0_parity.row_from_prepared(case, prepared, {11: -2.0, 22: -3.0, 33: -4.0})

    assert row.token_ids == [11, 22, 33]
    assert len(row.joint) == 3
    assert sum(row.joint) == pytest.approx(1.0)
    assert row.p_U == row.joint[2]
    assert row.input_sha256 == s0_parity.prepared_input_hash(prepared)
    assert row.media_sha256 == s0_parity.prepared_media_hash(prepared)


def test_row_json_round_trip_preserves_media_list():
    row = s0_parity.row_from_prepared(_case(2), _prepared(2), {11: -2.0, 22: -3.0, 33: -4.0})
    restored = s0_parity.row_from_json(s0_parity.row_to_json(row))

    assert restored == row
    assert isinstance(restored.media_sha256, list)


def test_validate_reference_rows_rejects_current_input_mismatch():
    cases = [_case(i) for i in range(100)]
    rows = [
        s0_parity.row_from_prepared(case, _prepared(i), {11: -2.0, 22: -3.0, 33: -4.0})
        for i, case in enumerate(cases)
    ]

    def changed_prepare(case, processor, asset_root):
        index = int(case.id.rsplit("-", 1)[1])
        prepared = _prepared(index)
        prepared.summary["prompt_sha256"] = "tampered"
        return prepared

    with pytest.raises(s0_parity.S0ParityError, match="canonical_http prepared inputs"):
        s0_parity.validate_prepared_rows(
            rows,
            cases=cases,
            processor=object(),
            asset_root=Path("/tmp/assets"),
            prepare=changed_prepare,
        )


def test_validate_server_record_requires_operator_attested_and_matching_provenance():
    provenance = {"version": "x"}
    with pytest.raises(s0_parity.S0ParityError, match="operator_attested"):
        s0_parity.validate_server_record({"provenance": provenance}, provenance=provenance)

    valid = s0_parity.validate_server_record(
        {
            "operator_attested": True,
            "raw_logprobs_confirmed": True,
            "served_model": "local-model",
            "provenance": provenance,
        },
        provenance=provenance,
    )
    assert valid["remote_attestation"] is False
    assert valid["server_startup"] == "not_performed_by_this_cli"


def test_run_vllm_pass_uses_canonical_prepared_inputs(monkeypatch):
    cases = [_case(i) for i in range(100)]

    def fake_prepare(case, processor, asset_root, *, profile):
        assert profile == s0_parity.CANONICAL_HTTP_PROFILE
        index = int(case.id.rsplit("-", 1)[1])
        return _prepared(index)

    monkeypatch.setattr(s0_parity, "prepare_case", fake_prepare)
    rows, summaries = asyncio.run(
        s0_parity.run_vllm_pass(
            cases=cases,
            processor=object(),
            asset_root=Path("/tmp/assets"),
            served_model="local-model",
            upstream_url="http://fake.local",
            transport=httpx.MockTransport(_response_for),
        )
    )

    assert len(rows) == 100
    assert len(summaries) == 100
    assert compare_rows(rows, rows, exact=True)["status"] == "passed"


def test_hf_pass_refuses_no_cuda_before_model_load(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    fake_torch = SimpleNamespace(cuda=FakeCuda())

    def fake_import(name):
        if name == "torch":
            return fake_torch
        if name == "transformers":
            raise AssertionError("transformers must not be imported after CUDA refusal")
        return __import__(name)

    monkeypatch.setattr(s0_parity.importlib, "import_module", fake_import)
    with pytest.raises(s0_parity.S0ParityError, match="cuda_required_before_model_load"):
        s0_parity.run_hf_pass(
            cases=[],
            processor=object(),
            paths=SimpleNamespace(
                model_dir=Path("/weights"), adapter_dir=None, asset_root=Path("/assets")
            ),
            dtype_name="bfloat16",
        )


def test_synthetic_control_refuses_overwrite(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}")
    assert s0_parity.main(["synthetic-control", "--report", str(report)]) == 2
