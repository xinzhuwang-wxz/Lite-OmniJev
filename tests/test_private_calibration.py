"""Synthetic CPU contracts only; no private rows, model or host ACL are accessed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scripts import private_calibration as job
from scripts import private_gold_job as broker
from train.calibration import LogitRow, TemperatureArtifact


def specification() -> dict[str, Any]:
    artifact = {"path": "/root/input.json", "sha256": "a" * 64}
    datasets = []
    for cohort, count in [
        ("text", 3000),
        ("image", 3000),
        ("screenshot", 3000),
        ("audio", 1600),
        ("audio", 1400),
        ("video", 3000),
        ("crossmodal", 3000),
    ]:
        datasets.append(
            dict(
                mode="S5",
                cohort=cohort,
                data=artifact,
                pool="calibration",
                media_root="/root/media",
                model_dir="/root/model",
                base_revision=job.stage.FROZEN_REVISION,
                model_files={"model.safetensors": "b" * 64},
                adapter_dir="/root/adapter",
                adapter_files={"adapter_model.safetensors": "c" * 64},
                admission=artifact,
                selected_manifest=artifact,
                data_admission=artifact,
                evidence_root="/root/evidence",
                expected_rows=count,
                output_root="/root/output",
                timeout_seconds=3600,
            )
        )
    return dict(
        datasets=datasets,
        producer_revision="d" * 40,
        runtime_python=artifact,
        inventory=artifact,
        inventory_receipt=artifact,
    )


def test_seven_real_groups_and_eleven_absent() -> None:
    spec = job.JobSpec.model_validate(specification())
    assert sum(s.expected_rows for s in spec.datasets) == 18000
    assert len(job.ALL_GROUPS - job.GROUP_COUNTS.keys()) == 11
    assert not spec.root_traverse_acl


@pytest.mark.parametrize("change", ["pool", "rows", "adapter", "unknown"])
def test_scope_is_frozen(change: str) -> None:
    value = specification()
    if change == "pool":
        value["datasets"][0]["pool"] = "holdout"
    elif change == "rows":
        value["datasets"][3]["expected_rows"] = 3000
    elif change == "adapter":
        value["datasets"][0]["adapter_files"] = {"adapter_model.safetensors": "e" * 64}
    else:
        value["command"] = ["arbitrary"]
    with pytest.raises(ValidationError):
        job.JobSpec.model_validate(value)


@pytest.mark.parametrize("uri", ["../secret", "file:///outside/a", "https://host/a", "a?x=1"])
def test_media_plan_cannot_escape_declared_root(uri: str) -> None:
    with pytest.raises(ValueError):
        job.lexical_media("/approved", uri)


class FakeACL:
    def __init__(self, text: str = "user::rwx\ngroup::---\nother::---\n") -> None:
        self.text = text
        self.writes: list[str] = []

    def read(self, path: Path) -> str:
        assert path == Path("/root")
        return self.text

    def write(self, path: Path, text: str) -> None:
        assert path == Path("/root")
        self.text = text
        self.writes.append(text)


def test_acl_restores_exactly_on_failure_and_keeps_recovery_record(tmp_path: Path) -> None:
    api = FakeACL()
    before = api.text
    evidence: dict[str, Any] = {}
    with pytest.raises(RuntimeError):
        with job.traverse_acl(49123, tmp_path / "backup.json", api=api) as evidence:
            assert "user:49123:--x" in api.text
            raise RuntimeError("worker failed")
    assert api.text == before
    assert evidence["restored"] is True
    assert evidence["before_sha256"] == evidence["after_sha256"]
    assert json.loads((tmp_path / "backup.json").read_text())["acl_before"] == before
    assert (tmp_path / "backup.json").stat().st_mode & 0o777 == 0o600


def test_acl_refuses_existing_named_entries(tmp_path: Path) -> None:
    api = FakeACL("user::rwx\nuser:42:rwx\ngroup::---\nmask::rwx\nother::---\n")
    with pytest.raises(ValueError, match="unexpected_root_acl"):
        with job.traverse_acl(49123, tmp_path / "backup.json", api=api):
            pytest.fail("must reject")
    assert not api.writes


def test_fit_exports_only_real_groups_and_exact_identity(tmp_path: Path, monkeypatch: Any) -> None:
    counts = {key: 2 for key in job.GROUP_COUNTS}
    monkeypatch.setattr(job, "GROUP_COUNTS", counts)
    groups = {}
    for key in counts:
        options = ["yes", "no"] if key.endswith("noul") else ["A", "B"]
        row = LogitRow(
            pool="calibration",
            logits_kind="raw_logits",
            complete_legal_logits=True,
            options=options,
            logits={options[0]: 8.0, options[1]: 7.0, "U": 6.0},
            target=options[0],
        )
        groups[key] = [row, row]
    identity = dict(
        base_revision="a" * 40,
        adapter_weights_sha256="b" * 64,
        official_template_sha256="c" * 64,
        answer_token_map_sha256="d" * 64,
        raw_logits_receipt_sha256="e" * 64,
        seed=42,
        code_revision="f" * 40,
    )
    entries = job.fit_groups(groups, identity, tmp_path)
    assert len(entries) == 7
    assert not list(tmp_path.glob("*score*"))
    for entry in entries:
        artifact = TemperatureArtifact.model_validate_json((tmp_path / entry["path"]).read_bytes())
        assert artifact.binding.answer_token_map_sha256 == "d" * 64
        assert artifact.binding.raw_logits_receipt_sha256 == "e" * 64
        assert artifact.nll_after <= artifact.nll_before
        private = json.loads(
            (tmp_path / entry["path"].replace("-temperature", "-input")).read_text()
        )
        assert max(private["rows"][0]["logits"].values()) == 8.0  # No log-softmax relabeling.
        assert artifact.stage_acceptance == "not_evaluated"


def test_broker_allowlists_calibration_and_does_not_echo_inputs(capsys: Any) -> None:
    assert broker.main(["--job", "execute-anything"]) == 1
    output = capsys.readouterr().out
    assert "execute-anything" not in output
    assert "rejected" in output


def test_backend_preserves_pre_softmax_values(tmp_path: Path, monkeypatch: Any) -> None:
    import sys
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")

    class Model:
        def requires_grad_(self, enabled: bool) -> None:
            assert enabled is False

        def eval(self) -> Any:
            return self

        def to(self, device: str) -> Any:
            assert device == "cuda"
            return self

        def __call__(self, **kwargs: Any) -> Any:
            assert kwargs["logits_to_keep"] == 1
            return SimpleNamespace(logits=torch.tensor([[[8.0, 7.0, 6.0]]]))

    model = Model()
    factory = SimpleNamespace(from_pretrained=lambda *a, **kw: model)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoModelForImageTextToText=factory)
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=factory))
    monkeypatch.setattr(job, "configure_torch_runtime", lambda module: None)
    monkeypatch.setattr(job, "configure_image_processor", lambda processor, budget: None)
    monkeypatch.setattr(job.baseline, "default_processor_factory", lambda path: object())
    monkeypatch.setattr(job.baseline, "move_inputs_to_cuda", lambda inputs, *args: inputs)
    _, forward, _ = job.raw_backend(tmp_path, tmp_path)
    actual = forward(SimpleNamespace(inputs={}, answer_token_ids={"A": 0, "B": 1, "U": 2}))
    assert actual == {"A": 8.0, "B": 7.0, "U": 6.0}
