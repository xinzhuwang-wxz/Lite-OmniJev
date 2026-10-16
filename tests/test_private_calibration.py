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


def test_private_media_snapshot_is_bounded_and_preserves_originals(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import hashlib
    import shutil
    from types import SimpleNamespace

    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    secret = source / "private.wav"
    duplicate = source / "duplicate.wav"
    public = source / "public.wav"
    for path, payload in [(secret, b"private"), (duplicate, b"private"), (public, b"public")]:
        path.write_bytes(payload)
        path.chmod(0o600)
    media = {
        str(path): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "type": "audio"}
        for path in (secret, duplicate, public)
    }

    def classify(path: Path) -> None:
        if path != public:
            raise ValueError("private")

    monkeypatch.setattr(job, "public_readonly", classify)
    monkeypatch.setattr(
        job.os, "statvfs", lambda path: SimpleNamespace(f_bavail=1024**3, f_frsize=1)
    )
    copied = []

    def copy_bound(path: Path, digest: str, target: Path) -> None:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        target.chmod(0o440)
        copied.append(target)

    mapping, summary = job.snapshot_media(
        media, [source], inputs, public_in_place=True, remaining_bytes=7, copy_bound=copy_bound
    )
    assert len(copied) == 1
    assert summary["private_snapshot_bytes"] == 7
    assert summary["public_assets_in_place"] == 1
    assert mapping[str(secret)] == mapping[str(duplicate)]
    assert Path("/" + mapping[str(secret)]).read_bytes() == b"private"
    assert Path("/" + mapping[str(public)]) == public
    assert secret.stat().st_mode & 0o777 == 0o600
    assert source.stat().st_mode & 0o777 == 0o700
    assert copied[0].stat().st_mode & 0o777 == 0o440
    with pytest.raises(ValueError, match="snapshot_budget_exceeded"):
        job.snapshot_media(
            media,
            [source],
            inputs,
            public_in_place=True,
            remaining_bytes=6,
            copy_bound=lambda *args: pytest.fail("copied early"),
        )
    monkeypatch.setattr(job.os, "statvfs", lambda path: SimpleNamespace(f_bavail=1, f_frsize=1))
    with pytest.raises(ValueError, match="private_media_disk_budget_exceeded"):
        job.snapshot_media(
            media,
            [source],
            inputs,
            public_in_place=True,
            remaining_bytes=100,
            copy_bound=lambda *args: pytest.fail("copied early"),
        )


def test_snapshot_rejects_undeclared_or_changed_media_before_copy(tmp_path: Path) -> None:
    path = tmp_path / "asset"
    path.write_bytes(b"changed")
    media = {str(path): {"sha256": "a" * 64, "type": "image"}}
    with pytest.raises(ValueError, match="media_changed"):
        job.snapshot_media(
            media,
            [tmp_path],
            tmp_path,
            public_in_place=False,
            remaining_bytes=100,
            copy_bound=lambda *args: pytest.fail("copied early"),
        )
    with pytest.raises(ValueError, match="media_plan_escape"):
        job.snapshot_media(
            media,
            [tmp_path / "other"],
            tmp_path,
            public_in_place=False,
            remaining_bytes=100,
            copy_bound=lambda *args: pytest.fail("copied early"),
        )


def progress_values() -> dict[str, Any]:
    return dict(
        phase="inference",
        cohort="text",
        processed=100,
        total=18000,
        elapsed_seconds=1.0,
        nonfinite_count=0,
        failure_category=None,
        failed_phase=None,
        spec_sha256="a" * 64,
        code_sha256="b" * 64,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"sample_id": "private"},
        {"processed": 18001},
        {"processed": True},
        {"elapsed_seconds": float("nan")},
        {"phase": "raw_traceback"},
        {"cohort": "private_path"},
    ],
)
def test_progress_is_strict_aggregate_only(mutation: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        job.Progress.model_validate({**progress_values(), **mutation})


def test_progress_atomic_publication_and_failed_final_state(
    tmp_path: Path, monkeypatch: Any
) -> None:
    report = tmp_path / "result.json"
    progress = report.with_suffix(".public-progress.json")

    def failed(*args: Any) -> Any:
        job.publish_progress(progress, job.Progress.model_validate(progress_values()))
        raise RuntimeError("private raw error must not be exported")

    monkeypatch.setattr(job, "_execute", failed)
    with pytest.raises(RuntimeError):
        job.execute(tmp_path / "spec", "a" * 64, tmp_path, report, 64001, 64002, 5)
    actual = json.loads(progress.read_text())
    assert actual["phase"] == "failed"
    assert actual["processed"] == 100
    assert set(actual) == set(progress_values())
    assert "private raw error" not in progress.read_text()
    assert progress.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_nonfinite_failure_exports_only_count(tmp_path: Path) -> None:
    state = {
        "output": str(tmp_path),
        "spec": {"datasets": [{"expected_rows": 18000}]},
        "source_spec_sha256": "a" * 64,
        "code_sha256": "b" * 64,
    }

    def bad(prepared: Any) -> dict[str, float]:
        raise ValueError("nonfinite_logits")

    with pytest.raises(ValueError):
        job.forward_private(bad, None, state, 100, 0.0, "text")
    actual = json.loads((tmp_path / "progress.json").read_text())
    assert actual["phase"] == "failed" and actual["nonfinite_count"] == 1
    assert actual["processed"] == 100


@pytest.mark.parametrize(
    "error, expected",
    [
        (PermissionError("/private/sample-secret"), "permission_denied"),
        (ImportError("private arbitrary module contents"), "import_failed"),
        (ValueError("public_readonly_asset_required"), "public_readonly_asset_required"),
        (ValueError("raw private target secret"), "input_contract_rejected"),
        (RuntimeError("private traceback"), "unclassified_failure"),
    ],
)
def test_error_categories_never_echo_private_exception(error: BaseException, expected: str) -> None:
    assert job.failure_category(error) == expected
    before = job.Progress.model_validate(progress_values())
    failed = job.failed_progress(before, error)
    assert failed.failure_category == expected and failed.failed_phase == "inference"
    assert "secret" not in failed.model_dump_json()


def test_worker_failure_preserves_runtime_phase_without_log_read(tmp_path: Path) -> None:
    state = {
        "output": str(tmp_path),
        "spec": {"datasets": [{"expected_rows": 18000}]},
        "source_spec_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "job_started_monotonic": 0.0,
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    job.record_worker_failure(path, "runtime", ImportError("DO NOT EXPORT raw path"))
    result = job.Progress.model_validate_json((tmp_path / "progress.json").read_bytes())
    assert result.failed_phase == "runtime" and result.failure_category == "import_failed"
    assert result.processed == 0


def test_private_adapter_is_snapshotted_while_public_base_stays_in_place(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import shutil

    model = tmp_path / "base"
    model.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    weight = adapter / "adapter_model.safetensors"
    weight.write_bytes(b"private adapter")
    weight.chmod(0o600)
    public = model / "model.safetensors"
    public.write_bytes(b"public base")
    seen = []
    monkeypatch.setattr(job, "public_readonly", lambda path: seen.append(path))

    def copy_bound(source: Path, digest: str, target: Path) -> None:
        assert job.sha256_file(source) == digest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o440)

    model_path, adapter_path = job.snapshot_model_assets(
        str(model),
        {"model.safetensors": job.sha256_file(public)},
        str(adapter),
        {"adapter_model.safetensors": job.sha256_file(weight)},
        inputs,
        True,
        copy_bound,
    )
    assert seen == [public]
    assert model_path == str(model) and adapter_path == str(inputs / "adapter")
    assert (Path(adapter_path) / weight.name).read_bytes() == b"private adapter"
    assert (Path(adapter_path) / weight.name).stat().st_mode & 0o777 == 0o440
    assert weight.stat().st_mode & 0o777 == 0o600
    assert not (inputs / "model").exists()
