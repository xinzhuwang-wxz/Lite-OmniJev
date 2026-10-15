"""CPU contract/failure tests; synthetic inputs do not establish GPU quality."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scripts import private_stage_eval as job
from tests.test_s0_development_baselines import FakeProcessor, _sample


def fixture(tmp_path: Path, **updates: Any) -> tuple[Path, job.Spec]:
    tmp_path.chmod(0o700)
    model = tmp_path / "model"
    model.mkdir()
    contents = {
        "config.json": {"model_type": "gemma4"},
        "tokenizer_config.json": {"chat_template": "official"},
        "tokenizer.json": {},
        "model.safetensors": {},
    }
    for name, value in contents.items():
        (model / name).write_text(json.dumps(value))
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(_sample(i)) + "\n" for i in range(3)))
    values = dict(
        mode="engineering",
        cohort="text",
        data={"path": str(data), "sha256": job.sha256_file(data)},
        pool="dev",
        media_root=str(tmp_path),
        model_dir=str(model),
        base_revision=job.FROZEN_REVISION,
        model_files={name: job.sha256_file(model / name) for name in contents},
        expected_rows=3,
        subset_limit=2,
        output_root=str(tmp_path / "private"),
        timeout_seconds=2,
    )
    values.update(updates)
    spec = job.Spec.model_validate(values)
    path = tmp_path / "spec.json"
    path.write_text(spec.model_dump_json())
    return path, spec


def backend(spec: job.Spec) -> tuple[Any, ...]:
    def forward(prepared: Any) -> dict[int, float]:
        return {
            index: math.log(0.6 if token == "A" else 0.1)
            for token, index in prepared.answer_token_ids.items()
        }

    return (
        FakeProcessor(),
        forward,
        lambda: None,
        {
            "python": "synthetic",
            "torch": "synthetic",
            "transformers": "synthetic",
            "device": "cpu fixture",
        },
        lambda: 0,
    )


def test_worker_reuses_official_nonthinking_raw_readout_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    path, spec = fixture(tmp_path)
    original = Path(spec.data.path).read_bytes()
    result = job.worker(path, backend)
    assert result.records == 2 and result.source_rows == 3
    assert result.quality_status == "not_evaluated"
    assert result.admission_status == "not_required_engineering"
    assert result.metrics.accuracy == 1
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert all(row["p_U"] == pytest.approx(1 / 9) for row in rows)
    assert all(sum(row["conditional_without_u"].values()) == pytest.approx(1) for row in rows)
    assert Path(spec.data.path).read_bytes() == original


@pytest.mark.parametrize(
    "update",
    [{"mode": "S4"}, {"pool": "holdout"}, {"seed": 43}, {"video_frames": 8}, {"backend": "fake"}],
)
def test_scope_rejections(tmp_path: Path, update: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        fixture(tmp_path, **update)


@pytest.mark.parametrize("kind", ["hash", "pool", "cohort", "partial", "model", "shard"])
def test_source_validation(tmp_path: Path, kind: str) -> None:
    _, spec = fixture(tmp_path)
    if kind == "hash":
        Path(spec.data.path).write_text("secret")
    elif kind == "pool":
        spec = spec.model_copy(update={"pool": "holdout"})
    elif kind == "cohort":
        spec = spec.model_copy(update={"cohort": "image"})
    elif kind == "partial":
        spec = spec.model_copy(update={"expected_rows": 4})
    elif kind == "model":
        (Path(spec.model_dir) / "unbound.safetensors").write_bytes(b"secret")
    else:
        shard = Path(spec.model_dir) / "model.safetensors.index.json"
        shard.write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
        spec.model_files[shard.name] = job.sha256_file(shard)
    with pytest.raises(job.Rejected):
        job.validate(spec)


def test_admission_exact_binding_and_no_stage_subset(tmp_path: Path) -> None:
    _, base = fixture(tmp_path)
    receipt = tmp_path / "admission.json"
    receipt.write_text(
        json.dumps(
            dict(
                schema_version="private-stage-data-admission-v1",
                status="admitted",
                data_sha256=base.data.sha256,
                selected_manifest_sha256="a" * 64,
                evidence_bundle_sha256="b" * 64,
                pool="holdout",
                cohort="text",
                expected_rows=3,
            )
        )
    )
    values = base.model_dump()
    values.update(
        mode="S4",
        pool="holdout",
        subset_limit=None,
        admission={"path": str(receipt), "sha256": job.sha256_file(receipt)},
        selected_manifest={"path": str(receipt), "sha256": job.sha256_file(receipt)},
        data_admission={"path": str(receipt), "sha256": job.sha256_file(receipt)},
        evidence_root=str(tmp_path),
    )
    stage = job.Spec.model_validate(values)
    with pytest.raises(job.Rejected):
        job.validate(stage)  # input dev pool never silently relabeled
    values["subset_limit"] = 1
    with pytest.raises(ValidationError):
        job.Spec.model_validate(values)


def test_receipt_unknown_fields_rejected(tmp_path: Path) -> None:
    path, _ = fixture(tmp_path)
    result = job.worker(path, backend).model_dump()
    result["secret"] = "private-prompt"
    with pytest.raises(ValidationError):
        job.Receipt.model_validate(result)


def test_real_subprocess_failure_is_redacted(tmp_path: Path, capsys: Any) -> None:
    path, _ = fixture(tmp_path)
    # Missing GPU dependencies cause fixed worker failure, never a fake GPU success.
    report = tmp_path / "public.json"
    assert job.main(["run", "--spec", str(path), "--report", str(report)]) == 1
    captured = capsys.readouterr()
    assert "data.jsonl" not in captured.out and "Traceback" not in captured.out
    assert "backend_failed" in captured.out or "timeout" in captured.out
    assert (
        job.FailureReceipt.model_validate_json(report.read_text()).quality_status == "not_evaluated"
    )
    runs = list((tmp_path / "private").iterdir())
    assert len(runs) == 1 and runs[0].stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("fixed", 1), KeyboardInterrupt()])
def test_timeout_and_interrupt_reap(
    tmp_path: Path, monkeypatch: Any, failure: BaseException
) -> None:
    path, _ = fixture(tmp_path)

    class Process:
        pid = 99999999
        calls = 0

        def poll(self) -> int:
            return 0

        def wait(self, timeout: int | None = None) -> int:
            self.calls += 1
            if timeout is not None:
                raise failure
            return -9

    process = Process()
    monkeypatch.setattr(job.subprocess, "Popen", lambda *a, **k: process)
    killed = []
    monkeypatch.setattr(job.os, "killpg", lambda pid, sig: killed.append(pid))
    with pytest.raises(job.Rejected, match="timeout|interrupted"):
        job.execute(path, tmp_path / "report.json")
    assert killed == [process.pid] and process.calls == 2


def test_no_overwrite(tmp_path: Path) -> None:
    path, _ = fixture(tmp_path)
    receipt = job.worker(path, backend)
    with pytest.raises(FileExistsError):
        job.publish(tmp_path / "receipt.json", receipt)


def test_raw_backend_exception_and_stdout_never_export(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    path, _ = fixture(tmp_path)

    class BrokenProcess:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["stdout"].write(b"SECRET prompt labels traceback")
            kwargs["stdout"].flush()

        def poll(self) -> int:
            return 0

        def wait(self, timeout: int | None = None) -> int:
            return 1

    monkeypatch.setattr(job.subprocess, "Popen", BrokenProcess)
    assert job.main(["run", "--spec", str(path), "--report", str(tmp_path / "public")]) == 1
    result = capsys.readouterr()
    assert "SECRET" not in result.out + result.err
    assert "backend_failed" in result.out


def test_partial_worker_records_rejected(tmp_path: Path, monkeypatch: Any) -> None:
    path, _ = fixture(tmp_path)
    monkeypatch.setattr(job.baseline, "run_pass", lambda *args, **kwargs: [])
    with pytest.raises(job.Rejected, match="partial_records"):
        job.worker(path, backend)


def test_bad_cli_redacts_unknown_arguments(capsys: Any) -> None:
    assert job.main(["run", "--secret", "sensitive-path"]) == 1
    assert "sensitive-path" not in capsys.readouterr().err


@pytest.mark.parametrize("tamper", ["unknown", "binding", "exit", "metrics", "admission"])
def test_supervisor_rejects_malformed_or_mismatched_receipt(
    tmp_path: Path, monkeypatch: Any, tamper: str
) -> None:
    path, _ = fixture(tmp_path)

    class Process:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            assert command[:4] == [
                job.sys.executable,
                "-m",
                "scripts.private_stage_eval",
                "_worker",
            ]
            private_spec = Path(command[-1])
            result = job.worker(private_spec, backend).model_dump()
            if tamper == "unknown":
                result["prompt"] = "SECRET"
            elif tamper == "binding":
                result["data_sha256"] = "a" * 64
            elif tamper == "metrics":
                result["metrics"]["count"] = 1
            elif tamper == "admission":
                result["admission_status"] = "admitted"
            else:
                result["exit_code"] = 1
            (private_spec.parent / "receipt.json").write_text(json.dumps(result))

        def poll(self) -> int:
            return 0

        def wait(self, timeout: int | None = None) -> int:
            return 0

    monkeypatch.setattr(job.subprocess, "Popen", Process)
    with pytest.raises(job.Rejected, match="invalid_receipt"):
        job.execute(path, tmp_path / "public.json")
    assert (
        job.FailureReceipt.model_validate_json((tmp_path / "public.json").read_text()).category
        == "invalid_receipt"
    )


def test_distinct_original_questions_can_share_family(tmp_path: Path) -> None:
    path, spec = fixture(tmp_path)
    rows = [_sample(i) for i in range(3)]
    for row in rows:
        row["meta"]["family_id"] = "shared-media-family"
    Path(spec.data.path).write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec.data.sha256 = job.sha256_file(Path(spec.data.path))
    path.write_text(spec.model_dump_json())
    result = job.worker(path, backend)
    assert result.records == 2 and result.unique_families == 1
    assert result.dependency_claim == "no_iid_claim"


@pytest.mark.parametrize(
    "media_types", [["image", "image"], ["video", "audio"], ["image", "audio", "video"]]
)
def test_crossmodal_requires_one_image_one_audio(tmp_path: Path, media_types: list[str]) -> None:
    _, spec = fixture(tmp_path, cohort="crossmodal")
    rows = [_sample(i) for i in range(3)]
    for row in rows:
        row["prefix"]["media"] = [
            {"type": kind, "uri": f"{j}.bin", "sha256": "a" * 64}
            for j, kind in enumerate(media_types)
        ]
        row["meta"]["modality"] = sorted(set(media_types))
    Path(spec.data.path).write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec.data.sha256 = job.sha256_file(Path(spec.data.path))
    with pytest.raises(job.Rejected, match="cohort_mismatch"):
        job.validate(spec)


def test_runtime_requires_all_fields_and_frozen_revision(tmp_path: Path) -> None:
    path, spec = fixture(tmp_path)
    values = spec.model_dump()
    values["base_revision"] = "e" * 40
    with pytest.raises(ValidationError):
        job.Spec.model_validate(values)
    result = job.worker(path, backend).model_dump()
    result["runtime"].pop("device")
    with pytest.raises(ValidationError):
        job.Receipt.model_validate(result)


def test_image_budget_is_passed_per_call() -> None:
    calls = []

    class Processor:
        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return kwargs

    processor = job.BudgetProcessor(Processor())
    processor(images=[object()], text="prompt")
    processor(audio=[object()], text="prompt")
    assert calls[0]["images_kwargs"] == {"max_soft_tokens": 1120}
    assert "images_kwargs" not in calls[1]


def test_environment_drops_tokens_tracking_and_python_injection(monkeypatch: Any) -> None:
    for key in (
        "HF_TOKEN",
        "SWANLAB_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "PYTHONPATH",
        "PYTHONSTARTUP",
    ):
        monkeypatch.setenv(key, "SECRET")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/cuda/lib")
    values = job.safe_environment()
    assert "SECRET" not in values.values()
    assert values["CUDA_VISIBLE_DEVICES"] == "0"
    assert values["LD_LIBRARY_PATH"] == "/cuda/lib"
    assert values["HF_HUB_OFFLINE"] == "1"


def test_failure_receipt_is_durable_and_never_overwrites(tmp_path: Path, monkeypatch: Any) -> None:
    path, _ = fixture(tmp_path)

    def fail(*args: Any) -> Any:
        raise job.Rejected("backend_failed", 7)

    monkeypatch.setattr(job, "_execute", fail)
    report = tmp_path / "failure.json"
    with pytest.raises(job.Rejected):
        job.execute(path, report)
    receipt = job.FailureReceipt.model_validate_json(report.read_text())
    assert receipt.exit_code == 7 and receipt.spec_sha256 == job.sha256_file(path)
    assert receipt.mode == "engineering" and receipt.cost_usd is None
    original = report.read_bytes()
    with pytest.raises(job.Rejected, match="report_exists"):
        job.execute(path, report)
    assert report.read_bytes() == original


def catalog_fixture(tmp_path: Path) -> job.Spec:
    _, base = fixture(tmp_path)
    cohorts = {"text", "image", "screenshot", "audio", "video", "crossmodal"}
    counts = {name: {"calibration": 3000, "holdout": 1000} for name in cohorts}
    original = {name: 500 for name in cohorts}
    inventory = "c" * 64
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "schema_version": "s4-selected-gold-manifest-v1",
                "inventory": {"sha256": inventory},
                "selected": [
                    {
                        "cohort": "text",
                        "pool": "holdout",
                        "path": base.data.path,
                        "sha256": base.data.sha256,
                        "expected_questions": 3,
                    }
                ],
                "auxiliary": [],
            }
        )
    )
    selected_hash = job.sha256_file(selected)
    reports: list[dict[str, Any]] = [
        {
            "outcome": "selected_counts_and_targets_verified",
            "manifest_sha256": selected_hash,
            "inventory_sha256": inventory,
            "selected_question_counts": counts,
            "original_development": original,
        },
        {
            "outcome": "mechanical_source_bindings_verified",
            "sources": [
                {"source": name, "status": "verified"}
                for name in (
                    "gui",
                    "clotho",
                    "speech",
                    "video",
                    "vqa",
                    "crossmodal",
                    "crossmodal_dev",
                )
            ],
        },
        {
            "jobs": [
                {"source": name, "mismatched_or_missing": 0}
                for name in (
                    "text",
                    "gui",
                    "clotho",
                    "speech",
                    "video",
                    "vqa",
                    "crossmodal",
                    "crossmodal_dev",
                )
            ]
        },
        {"outcome": "matched_original_gold"},
        {"outcome": "original_mapping_verified", "preserved_items": 120},
        {"outcome": "uri_only_changes_verified"},
        {"outcome": "known_exposures_reconciled", "inventory_sha256": inventory},
        {
            "outcome": "isolated",
            "receipt": {"manifest_sha256": inventory, "isolation": {"overlapping_components": 0}},
        },
        {
            "outcome": "isolated",
            "ordinary_identity_boundary": True,
            "source_manifest_sha256": inventory,
            "code_sha256": "d" * 64,
        },
        {
            "passed": True,
            "code_sha256": "d" * 64,
            "checks": {
                name: True
                for name in (
                    "valid",
                    "invalid",
                    "overlap",
                    "cli_redaction",
                    "hash_rejection",
                    "overwrite_rejected",
                    "malformed_receipt_rejected",
                    "extra_receipt_rejected",
                    "private_log_captured",
                    "exception_rejected",
                    "private_log_denied_after_run",
                    "timeout_reaped",
                    "exported_surfaces_redacted",
                )
            },
        },
        {
            name: True
            for name in (
                "live_unregistered_identity_rejected",
                "used_identity_rejected",
                "retained_runs_root_only",
                "prior_workers_denied_retained_runs",
            )
        },
    ]
    (tmp_path / "reports").mkdir()
    evidence = {}
    for name, report in zip(job.REQUIRED_EVIDENCE, reports, strict=True):
        path = tmp_path / "reports" / name
        path.write_text(json.dumps({**report, "exit_code": 0}))
        evidence["reports/" + name] = job.sha256_file(path)
    bundle = job.sha256_json(evidence)
    global_path = tmp_path / "data-admission.json"
    global_path.write_text(
        json.dumps(
            {
                "schema_version": "s4-gold-data-admission-v1",
                "status": "data_admitted",
                "exit_code": 0,
                "selected_manifest_sha256": selected_hash,
                "inventory_sha256": inventory,
                "evidence_bundle_sha256": bundle,
                "evidence": evidence,
                "counts": counts,
                "original_development": original,
            }
        )
    )
    per_data = tmp_path / "per-data.json"
    per_data.write_text(
        json.dumps(
            {
                "schema_version": "private-stage-data-admission-v1",
                "status": "admitted",
                "data_sha256": base.data.sha256,
                "selected_manifest_sha256": selected_hash,
                "evidence_bundle_sha256": bundle,
                "pool": "holdout",
                "cohort": "text",
                "expected_rows": 3,
            }
        )
    )
    values = base.model_dump()
    values.update(
        mode="S4",
        pool="holdout",
        subset_limit=None,
        evidence_root=str(tmp_path),
        selected_manifest={"path": str(selected), "sha256": selected_hash},
        data_admission={"path": str(global_path), "sha256": job.sha256_file(global_path)},
        admission={"path": str(per_data), "sha256": job.sha256_file(per_data)},
    )
    return job.Spec.model_validate(values)


@pytest.mark.parametrize(
    "tamper", [None, "mutated_evidence", "missing_report", "nonmember", "self_declared"]
)
def test_admission_rechecks_bound_catalog(tmp_path: Path, tamper: str | None) -> None:
    spec = catalog_fixture(tmp_path)
    assert spec.data_admission and spec.selected_manifest
    if tamper == "mutated_evidence":
        (tmp_path / "reports" / job.REQUIRED_EVIDENCE[0]).write_text("{}")
    elif tamper == "missing_report":
        path = Path(spec.data_admission.path)
        report = json.loads(path.read_text())
        report["evidence"].pop("reports/" + job.REQUIRED_EVIDENCE[0])
        path.write_text(json.dumps(report))
        spec.data_admission.sha256 = job.sha256_file(path)
    elif tamper == "nonmember":
        spec.data = job.Artifact(path=spec.data.path, sha256="e" * 64)
    elif tamper == "self_declared":
        spec.selected_manifest = None
        spec.data_admission = None
    if tamper is None:
        job.verify_catalog(spec)  # synthetic catalog validates mechanics, not actual gold
    else:
        with pytest.raises(job.Rejected, match="admission_mismatch"):
            job.verify_catalog(spec)


@pytest.mark.parametrize("signum", [job.signal.SIGTERM, job.signal.SIGHUP])
def test_real_supervisor_signal_kills_and_reaps_worker(tmp_path: Path, signum: int) -> None:
    import os
    import time

    path, _ = fixture(tmp_path, timeout_seconds=60)
    report = tmp_path / "signal-report.json"
    pidfile = tmp_path / "worker.pid"
    program = """
import subprocess,sys
from pathlib import Path
from scripts import private_stage_eval as job
spawn=subprocess.Popen
sleeper=("import os,time; from pathlib import Path; Path("+repr(sys.argv[3])
         +").write_text(str(os.getpid())); print('PRIVATE signal test'); time.sleep(60)")
def sleeping_worker(command, **kwargs):
    return spawn([sys.executable,'-c',sleeper], **kwargs)
job.subprocess.Popen=sleeping_worker
try:
    job.execute(Path(sys.argv[1]),Path(sys.argv[2]))
except job.Rejected:
    raise SystemExit(1)
"""
    supervisor = subprocess.Popen(
        [job.sys.executable, "-c", program, str(path), str(report), str(pidfile)],
        cwd=Path(job.__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pidfile.exists()
        worker_pid = int(pidfile.read_text())
        supervisor.send_signal(signum)
        stdout, stderr = supervisor.communicate(timeout=10)
        assert supervisor.returncode == 1
        assert b"PRIVATE" not in stdout + stderr
        receipt = job.FailureReceipt.model_validate_json(report.read_text())
        assert receipt.category == "interrupted" and receipt.exit_code == -9
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait()


def test_scoped_signal_handlers_restore(tmp_path: Path, monkeypatch: Any) -> None:
    path, _ = fixture(tmp_path)
    before = {sig: job.signal.getsignal(sig) for sig in (job.signal.SIGTERM, job.signal.SIGHUP)}

    def fail(*args: Any) -> Any:
        raise job.Rejected("backend_failed", 1)

    monkeypatch.setattr(job, "_execute", fail)
    with pytest.raises(job.Rejected):
        job.execute(path, tmp_path / "report.json")
    assert all(job.signal.getsignal(sig) == handler for sig, handler in before.items())


def test_file_uri_canonicalization_is_bound_and_never_mutates_input(tmp_path: Path) -> None:
    _, spec = fixture(tmp_path, cohort="image")
    media = tmp_path / "image.png"
    media.write_bytes(b"synthetic validation only")
    rows = [_sample(i) for i in range(3)]
    for row in rows:
        row["prefix"]["media"] = [
            {"type": "image", "uri": media.as_uri(), "sha256": job.sha256_file(media)}
        ]
        row["meta"]["modality"] = ["image"]
    data = Path(spec.data.path)
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    before = data.read_bytes()
    spec.data.sha256 = job.sha256_file(data)
    samples, evidence = job.validate(spec)
    assert all(sample.prefix.media[0].uri == "image.png" for sample in samples)
    assert evidence["canonicalized_input_sha256"] == job.sha256_json(
        [sample.model_dump(mode="json") for sample in samples]
    )
    assert data.read_bytes() == before
    rows[0]["prefix"]["media"][0]["uri"] = "file://remote-host/image.png"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    spec.data.sha256 = job.sha256_file(data)
    with pytest.raises(job.Rejected, match="input_rejected"):
        job.validate(spec)
