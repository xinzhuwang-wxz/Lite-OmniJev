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
