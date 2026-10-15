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


def _mapping_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _prepared(i: int):
    messages = [{"role": "user", "content": [{"type": "text", "text": f"case {i}"}]}]
    media_kwargs = {}
    answer_token_ids = {"A": 11, "B": 22, "U": 33}
    return SimpleNamespace(
        answer_token_ids=answer_token_ids,
        inputs={"input_ids": _InputIds([1000 + i, 2000 + i])},
        http_messages=messages,
        http_media_io_kwargs=media_kwargs,
        http_image_max_soft_tokens=None,
        summary={
            "input_profile": "canonical-http-v1",
            "prompt_sha256": f"prompt-{i}",
            "http_messages_sha256": _sha(messages),
            "http_media_io_kwargs_sha256": _sha(media_kwargs),
            "tensors": {"input_ids": {"sha256": f"ids-{i}", "shape": [1, 2], "dtype": "int64"}},
            "answer_token_mapping_sha256": _mapping_sha(answer_token_ids),
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


def test_row_from_prepared_binds_real_tensors_key_and_probabilities():
    case = _case(1)
    prepared = _prepared(1)
    row = s0_parity.row_from_prepared(case, prepared, {11: -2.0, 22: -3.0, 33: -4.0})

    original_hash = row.input_sha256
    prepared.summary["tensors"]["input_ids"]["sha256"] = "changed"

    assert row.token_ids == [11, 22, 33]
    assert len(row.joint) == 3
    assert sum(row.joint) == pytest.approx(1.0)
    assert row.p_U == row.joint[2]
    assert original_hash != s0_parity.prepared_input_hash(prepared)
    assert row.media_sha256 == s0_parity.prepared_media_hash(prepared)


def test_row_json_round_trip_strictly_validates_saved_derived_fields():
    row = s0_parity.row_from_prepared(_case(2), _prepared(2), {11: -2.0, 22: -3.0, 33: -4.0})
    saved = s0_parity.row_to_json(row)
    restored = s0_parity.row_from_json(saved)

    assert restored == row
    assert isinstance(restored.media_sha256, list)

    tampered = {**saved, "raw_probabilities": [0.0, *saved["raw_probabilities"][1:]]}
    with pytest.raises(s0_parity.S0ParityError, match="strict derived"):
        s0_parity.row_from_json(tampered)

    with_extra = {**saved, "extra": "not allowed"}
    with pytest.raises(s0_parity.S0ParityError, match="strict parity schema"):
        s0_parity.row_from_json(with_extra)

    bool_coerce = {**saved, "raw_logprobs": [True, *saved["raw_logprobs"][1:]]}
    with pytest.raises(s0_parity.S0ParityError, match="strict list of floats"):
        s0_parity.row_from_json(bool_coerce)

    int_coerce = {**saved, "raw_logprobs": [-2, *saved["raw_logprobs"][1:]]}
    with pytest.raises(s0_parity.S0ParityError, match="strict list of floats"):
        s0_parity.row_from_json(int_coerce)


def test_validate_reference_rows_rejects_current_input_mismatch():
    cases = [_case(i) for i in range(100)]
    rows = [
        s0_parity.row_from_prepared(case, _prepared(i), {11: -2.0, 22: -3.0, 33: -4.0})
        for i, case in enumerate(cases)
    ]

    def changed_prepare(case, processor, asset_root):
        index = int(case.id.rsplit("-", 1)[1])
        prepared = _prepared(index)
        prepared.summary["tensors"]["input_ids"]["sha256"] = "tampered"
        return prepared

    with pytest.raises(s0_parity.S0ParityError, match="canonical_http prepared inputs"):
        s0_parity.validate_prepared_rows(
            rows,
            cases=cases,
            processor=object(),
            asset_root=Path("/tmp/assets"),
            prepare=changed_prepare,
        )


def test_validate_server_record_requires_launch_dtype_and_logprobs_mode():
    provenance = {"version": "x", "dtype": "bfloat16"}
    with pytest.raises(s0_parity.S0ParityError, match="operator_attested"):
        s0_parity.validate_server_record({"provenance": provenance}, provenance=provenance)

    base = {
        "operator_attested": True,
        "raw_logprobs_confirmed": True,
        "served_model": "local-model",
        "provenance": provenance,
        "dtype": "bfloat16",
        "logprobs_mode": "raw_logprobs",
        "launch_command": "vllm serve local-model --dtype bfloat16",
    }
    valid = s0_parity.validate_server_record(base, provenance=provenance)
    assert valid["remote_attestation"] is False
    assert valid["server_startup"] == "not_performed_by_this_cli"
    assert valid["launch_command_present"] is True

    bad = {**base, "logprobs_mode": "top_logprobs"}
    with pytest.raises(s0_parity.S0ParityError, match="logprobs_mode"):
        s0_parity.validate_server_record(bad, provenance=provenance)

    missing_launch = dict(base)
    missing_launch.pop("launch_command")
    with pytest.raises(s0_parity.S0ParityError, match="launch_command"):
        s0_parity.validate_server_record(missing_launch, provenance=provenance)


def test_run_vllm_pass_uses_canonical_prepared_inputs(monkeypatch):
    cases = [_case(i) for i in range(100)]

    def fake_prepare(case, processor, asset_root, *, profile, image_max_soft_tokens):
        assert image_max_soft_tokens is None
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


@pytest.mark.parametrize("dtype_name", sorted(s0_parity.ALLOWED_DTYPES))
def test_every_admitted_dtype_reaches_official_loader(monkeypatch, tmp_path, dtype_name):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    received = {}

    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            received.update(kwargs)
            raise RuntimeError("official loader boundary reached")

    original_import = s0_parity.importlib.import_module
    monkeypatch.setattr(s0_parity, "ensure_cuda_available", lambda: torch)
    monkeypatch.setattr(s0_parity, "configure_torch_runtime", lambda _torch: None)
    monkeypatch.setattr(s0_parity, "runtime_requirements", lambda: {})
    monkeypatch.setattr(
        s0_parity.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(Gemma4ForConditionalGeneration=Loader)
            if name == "transformers"
            else original_import(name)
        ),
    )
    paths = s0_parity.ParityPaths(tmp_path, tmp_path, tmp_path, tmp_path, None)
    with pytest.raises(RuntimeError, match="official loader boundary reached"):
        s0_parity.run_hf_pass(cases=[], processor=None, paths=paths, dtype_name=dtype_name)
    assert received["torch_dtype"] == getattr(torch, dtype_name)
    assert received["local_files_only"] and received["use_safetensors"]
    assert received["trust_remote_code"] is False


def test_hf_reference_writes_blocked_report_for_cpu_gate(monkeypatch, tmp_path):
    report = tmp_path / "hf-blocked.json"
    monkeypatch.setattr(
        s0_parity, "build_provenance", lambda paths, dtype, **kwargs: {"dtype": dtype}
    )
    monkeypatch.setattr(s0_parity, "load_cases", lambda manifest: [])
    monkeypatch.setattr(s0_parity, "load_official_processor", lambda model_dir: object())

    def blocked(**kwargs):
        raise s0_parity.S0ParityError("cuda_required_before_model_load")

    monkeypatch.setattr(s0_parity, "run_hf_pass", blocked)
    args = SimpleNamespace(
        manifest=Path("manifest.json"),
        asset_root=Path("assets"),
        model_dir=Path("model"),
        asset_manifest=Path("manifest-assets.json"),
        adapter_dir=None,
        dtype="bfloat16",
        report=report,
    )

    assert s0_parity.run_hf_reference(args) == 2
    data = json.loads(report.read_text())
    assert data["status"] == "blocked"
    assert data["exit_code"] == 2
    assert data["error"]["type"] == "S0ParityError"
    assert data["environment"]["packages"]
    assert data["source"]["git"]["commit"]
    assert data["cost"] == {"new_paid_resources": 0, "gpu_started_by_this_cli": False}


def test_synthetic_control_refuses_overwrite(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}")
    assert s0_parity.main(["synthetic-control", "--report", str(report)]) == 2


@pytest.mark.parametrize("budget", [None, 70, 140, 280, 560, 1120])
def test_budget_binds_provenance_and_native_request(monkeypatch, tmp_path, budget):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(s0_parity, "verified_assets", lambda *args: {})
    monkeypatch.setattr(s0_parity, "source_identity", lambda: {})
    paths = s0_parity.ParityPaths(manifest, tmp_path, tmp_path, manifest, None)
    provenance = s0_parity.build_provenance(paths, dtype="float32", image_max_soft_tokens=budget)
    assert provenance.get("image_max_soft_tokens") == budget
    if budget is None:
        assert "image_max_soft_tokens" not in provenance

    prepared = _prepared(0)
    prepared.http_image_max_soft_tokens = budget
    if budget is not None:
        prepared.summary["http_mm_processor_kwargs_sha256"] = _mapping_sha(
            {"images_kwargs": {"max_soft_tokens": budget}}
        )

    def prepare(*args, profile, image_max_soft_tokens):
        assert profile == "canonical_http"
        assert image_max_soft_tokens == budget
        return prepared

    def respond(request):
        payload = json.loads(request.content)
        assert payload.get("mm_processor_kwargs") == (
            {"images_kwargs": {"max_soft_tokens": budget}} if budget is not None else None
        )
        return _response_for(request)

    monkeypatch.setattr(s0_parity, "prepare_case", prepare)
    rows, _ = asyncio.run(
        s0_parity.run_vllm_pass(
            cases=[_case(0)],
            processor=object(),
            asset_root=tmp_path,
            served_model="local",
            upstream_url="http://fake.local",
            transport=httpx.MockTransport(respond),
            image_max_soft_tokens=budget,
        )
    )
    assert len(rows) == 1


@pytest.mark.parametrize("budget", [None, 1120])
def test_reference_validation_reprepares_requested_budget(monkeypatch, tmp_path, budget):
    cases = [_case(i) for i in range(100)]
    prepared = [_prepared(i) for i in range(100)]
    for item in prepared:
        item.http_image_max_soft_tokens = budget
    provenance = {} if budget is None else {"image_max_soft_tokens": budget}
    rows = [
        s0_parity.row_from_prepared(case, item, {11: -2.0, 22: -3.0, 33: -4.0})
        for case, item in zip(cases, prepared, strict=True)
    ]
    reference = {
        "action": "hf-reference",
        "status": "reference-ready",
        "provenance": provenance,
        "rows": s0_parity.rows_to_json(rows),
        "hf_repeatability": {"passed": True},
    }

    def prepare(case, *args, profile, image_max_soft_tokens):
        assert image_max_soft_tokens == budget
        return prepared[int(case.id.rsplit("-", 1)[1])]

    monkeypatch.setattr(s0_parity, "prepare_case", prepare)
    _, result = s0_parity.validate_reference_report(
        reference, provenance=provenance, cases=cases, processor=object(), asset_root=tmp_path
    )
    assert result["passed"]


@pytest.mark.parametrize(
    "reference_budget,current_budget", [(None, 1120), (1120, None), (280, 1120)]
)
def test_reference_budget_mismatch_rejected_before_preparation(
    monkeypatch, tmp_path, reference_budget, current_budget
):
    def forbidden(*args, **kwargs):
        pytest.fail("mismatched reference must fail before preparation or inference")

    monkeypatch.setattr(s0_parity, "prepare_case", forbidden)
    reference = {
        "action": "hf-reference",
        "status": "reference-ready",
        "provenance": {}
        if reference_budget is None
        else {"image_max_soft_tokens": reference_budget},
    }
    provenance = {} if current_budget is None else {"image_max_soft_tokens": current_budget}
    with pytest.raises(s0_parity.S0ParityError, match="image_max_soft_tokens"):
        s0_parity.validate_reference_report(
            reference, provenance=provenance, cases=[], processor=object(), asset_root=tmp_path
        )


@pytest.mark.parametrize("budget", [0, 281, True, "1120"])
def test_invalid_budget_rejected_before_hf_cuda(monkeypatch, tmp_path, budget):
    def forbidden():
        pytest.fail("invalid budget must not enter CUDA")

    monkeypatch.setattr(s0_parity, "ensure_cuda_available", forbidden)
    with pytest.raises(ValueError, match="image_max_soft_tokens"):
        s0_parity.run_hf_pass(
            cases=[], processor=None, paths=None, dtype_name="float32", image_max_soft_tokens=budget
        )


def test_budget_changes_input_identity_without_changing_tensors():
    prepared = _prepared(0)
    original = s0_parity.prepared_input_hash(prepared)
    prepared.http_image_max_soft_tokens = 1120
    assert s0_parity.prepared_input_hash(prepared) != original


def test_hf_forward_receives_budget_prepared_inputs(monkeypatch, tmp_path):
    from contextlib import nullcontext

    prepared = _prepared(0)
    prepared.inputs = {"pixel_values": object()}
    seen = []

    class Logits:
        def __getitem__(self, key):
            return self

        def float(self):
            return self

        def item(self):
            return -2.0

    class Model:
        def requires_grad_(self, value):
            pass

        def eval(self):
            pass

        def to(self, value):
            pass

        def __call__(self, **kwargs):
            assert kwargs["pixel_values"] is prepared.inputs["pixel_values"]
            return SimpleNamespace(logits=Logits())

    torch = SimpleNamespace(
        float32="float32",
        inference_mode=nullcontext,
        nn=SimpleNamespace(functional=SimpleNamespace(log_softmax=lambda x, **kw: x)),
        cuda=SimpleNamespace(empty_cache=lambda: None, is_available=lambda: False),
    )
    monkeypatch.setattr(s0_parity, "ensure_cuda_available", lambda: torch)
    monkeypatch.setattr(s0_parity, "configure_torch_runtime", lambda _: None)
    monkeypatch.setattr(s0_parity, "runtime_requirements", lambda: None)
    monkeypatch.setattr(
        s0_parity.importlib,
        "import_module",
        lambda _: SimpleNamespace(
            Gemma4ForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **kw: Model())
        ),
    )

    def prepare(*args, profile, image_max_soft_tokens):
        seen.append(image_max_soft_tokens)
        return prepared

    monkeypatch.setattr(s0_parity, "prepare_case", prepare)
    paths = s0_parity.ParityPaths(tmp_path, tmp_path, tmp_path, tmp_path, None)
    rows = s0_parity.run_hf_pass(
        cases=[_case(0)],
        processor=None,
        paths=paths,
        dtype_name="float32",
        image_max_soft_tokens=1120,
    )
    assert seen == [1120]
    assert len(rows) == 1


@pytest.mark.parametrize("action", ["hf-reference", "compare-vllm"])
def test_cli_budget_choices_and_default(action):
    argv = [
        action,
        "--manifest",
        "m",
        "--asset-root",
        "a",
        "--model-dir",
        "d",
        "--asset-manifest",
        "am",
        "--report",
        "r",
    ]
    if action == "compare-vllm":
        argv += ["--reference", "ref", "--server-record", "server"]
    parser = s0_parity.build_parser()
    assert parser.parse_args(argv).image_max_soft_tokens is None
    assert (
        parser.parse_args([*argv, "--image-max-soft-tokens", "1120"]).image_max_soft_tokens == 1120
    )
    with pytest.raises(SystemExit):
        parser.parse_args([*argv, "--image-max-soft-tokens", "281"])


def test_server_cannot_reuse_different_budget_identity():
    with pytest.raises(s0_parity.S0ParityError, match="server record provenance"):
        s0_parity.validate_server_record(
            {"operator_attested": True, "provenance": {"image_max_soft_tokens": 280}},
            provenance={"image_max_soft_tokens": 1120},
        )
