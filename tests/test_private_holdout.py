"""Synthetic CPU holdout contracts; no private data or GPU execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scripts import private_calibration as cal
from scripts import private_holdout as job
from tests.test_private_calibration import specification


def spec_values() -> dict[str, Any]:
    values = specification()
    values["schema_version"] = "private-holdout-job-v1"
    values["gates"] = {"path": "/root/gates.json", "sha256": "f" * 64}
    for source in values["datasets"]:
        source["adapter_files"]["adapter_config.json"] = "e" * 64
        source.update(
            mode="S4", pool="holdout", expected_rows=500 if source["cohort"] == "audio" else 1000
        )
    return values


def test_only_full_six_cohort_holdout_is_accepted() -> None:
    spec = job.JobSpec.model_validate(spec_values())
    assert sum(s.expected_rows for s in spec.datasets) == 6000
    assert job.GROUP_COUNTS["audio/choice"] == job.GROUP_COUNTS["audio/noul"] == 500
    assert spec.temperatures == []
    for pool in ("train", "dev", "calibration"):
        values = spec_values()
        values["datasets"][0]["pool"] = pool
        with pytest.raises(ValidationError):
            job.JobSpec.model_validate(values)


def test_partial_temperatures_or_self_declared_fit_are_rejected() -> None:
    values = spec_values()
    values["calibration_receipt"] = {"path": "/root/fake", "sha256": "a" * 64}
    with pytest.raises(ValidationError, match="fit_receipt_required"):
        job.JobSpec.model_validate(values)
    values = spec_values()
    values["datasets"][0]["expected_rows"] = 999
    with pytest.raises(ValidationError, match="six_full"):
        job.JobSpec.model_validate(values)


def test_same_logits_raw_and_temperature_metrics_preserve_argmax() -> None:
    raw: list[Any] = []
    scaled: list[Any] = []
    for target in ("A", "B"):
        logits = {"A": 8.0, "B": 7.0, "U": 6.0}
        job.append_readout(raw, logits, ["A", "B"], target, 1.0)
        job.append_readout(scaled, logits, ["A", "B"], target, 2.0)
    a, b = job.summarize(raw), job.summarize(scaled)
    assert a["count"] == b["count"] == 2
    assert a["correct"] == b["correct"] == 1
    assert a["brier"] != b["brier"]
    assert a["mean_p_insufficient"] != b["mean_p_insufficient"]
    assert a["joint_legal_plus_u_nll"] != b["joint_legal_plus_u_nll"]
    assert set(a) == {
        *job.stage.Metrics.model_fields,
        "correct",
        "mean_p_insufficient",
        "joint_legal_plus_u_nll",
    }


def test_missing_evidence_cannot_pass_and_thresholds_cannot_be_lowered() -> None:
    gates = json.loads(Path("configs/gates.json").read_text())
    raw = {
        c: {"accuracy": 0.9, "ece15": 0.02, "coverage_at_error": 0.7}
        for c in ("text", "image", "screenshot", "audio", "video", "crossmodal")
    }
    result = job.quality_observations(raw, {}, {}, gates)
    assert not result["s4_gap_observation"]["passed"]
    assert not result["s5_probability_observations"]["passed"]
    assert result["stage_acceptance"] == "not_evaluated"
    result = job.quality_observations(raw, raw, raw, gates)
    assert result["s4_gap_observation"]["passed"]
    assert result["s5_probability_observations"]["passed"]
    assert result["stage_acceptance"] == "not_evaluated"
    gates["stages"]["S5"]["max_modality_ece"]["max"] = 0.99
    with pytest.raises(ValueError, match="original_stage_thresholds"):
        job.quality_observations(raw, raw, raw, gates)


def test_holdout_export_rejects_row_fields_before_other_work(tmp_path: Path) -> None:
    spec = job.JobSpec.model_validate(spec_values())
    with pytest.raises(ValueError, match="unexpected_holdout_export_fields"):
        job.export_receipt(spec, tmp_path, {"rows": [{"private": "secret"}]}, "a" * 64, "b" * 64)


def test_dispatch_is_fixed_and_default_calibration_unchanged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls = []

    def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "synthetic"}

    monkeypatch.setattr(cal, "_execute", fake)
    args = (tmp_path / "spec", "a" * 64, tmp_path, tmp_path / "report", 65001, 65002, 5)
    cal.execute(*args)
    cal.execute(*args, job_kind="holdout-eval")
    assert calls == [{}, {"job_kind": "holdout-eval"}]
    with pytest.raises(ValueError, match="unknown_fixed_job"):
        cal.execute(*args, job_kind="arbitrary-module")


def synthetic_state(tmp_path: Path) -> dict[str, Any]:
    from tests.test_s0_development_baselines import _sample

    values = spec_values()
    values["gates"] = {
        "path": str(Path("configs/gates.json").resolve()),
        "sha256": job.sha256_file(Path("configs/gates.json")),
    }
    snapshots = []
    number = 0
    media_map = {}
    for index, source in enumerate(values["datasets"]):
        rows = []
        for _ in range(source["expected_rows"]):
            row = _sample(number, pool="holdout")
            number += 1
            cohort = source["cohort"]
            primitive = "noul" if cohort in ("video", "crossmodal") or index == 4 else "choice"
            if primitive == "noul":
                row["questions"][0].update(
                    type="noul",
                    options=None,
                    target={"kind": "hard", "value": "yes", "source": "fixture"},
                )
            kinds = (
                []
                if cohort == "text"
                else ["audio", "image"]
                if cohort == "crossmodal"
                else ["image" if cohort == "screenshot" else cohort]
            )
            row["prefix"]["media"] = [
                {"type": kind, "uri": kind + ".bin", "sha256": "a" * 64} for kind in kinds
            ]
            row["meta"]["modality"] = kinds or ["text"]
            for kind in kinds:
                media_map["/root/media/" + kind + ".bin"] = "media/" + kind + ".bin"
            rows.append(json.dumps(row))
        path = tmp_path / f"data-{index}.jsonl"
        path.write_text("\n".join(rows) + "\n")
        source["data"] = {"path": str(path), "sha256": job.sha256_file(path)}
        snapshots.append(str(path))
    output = tmp_path / "output"
    output.mkdir()
    media_path = tmp_path / "media-map.json"
    media_path.write_text(json.dumps(media_map))
    state = dict(
        spec=values,
        datasets=snapshots,
        output=str(output),
        model=str(tmp_path / "model"),
        adapter=str(tmp_path / "adapter"),
        media_root=str(tmp_path),
        media_map=str(media_path),
        metadata={"gates": values["gates"]["path"]},
        source_spec_sha256="a" * 64,
        code_sha256=job.stage.code_hash(),
    )
    return state


def test_full_synthetic_holdout_worker_exports_only_validated_aggregates(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from types import SimpleNamespace

    from data.schema import answer_tokens
    from scripts.check_tokens import ANSWERS

    state = synthetic_state(tmp_path)
    spec, samples = job.private_samples(state)
    assert len(samples) == 6000
    mapping = dict(zip(ANSWERS, range(33), strict=True))
    official = {"status": "passed", "answer_token_ids": mapping, "template_sha256": "c" * 64}
    monkeypatch.setattr(job, "probe", lambda path: official)
    monkeypatch.setattr(job.stage, "bind_files", lambda *args, **kwargs: "b" * 64)

    def prepare(sample: Any, processor: Any, root: Path) -> Any:
        tokens = [*answer_tokens(sample.questions[0]), "U"]
        return SimpleNamespace(answer_token_ids={t: mapping[t] for t in tokens}, tokens=tokens)

    monkeypatch.setattr(job.baseline, "prepare_sample", prepare)
    monkeypatch.setattr(
        cal,
        "raw_backend",
        lambda *args: (
            object(),
            lambda prepared: {t: float(i == 0) for i, t in enumerate(prepared.tokens)},
            object(),
        ),
    )
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    job.evaluate_worker(path)
    output = Path(state["output"])
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["total_count"] == 6000
    assert receipt["raw"]["groups"]["audio/choice"]["count"] == 500
    assert receipt["raw"]["cohorts"]["audio"]["count"] == 1000
    assert receipt["calibrated"] is None
    assert receipt["quality"]["stage_acceptance"] == "not_evaluated"
    assert "state 001" not in json.dumps(receipt)
    assert set(p.name for p in output.iterdir()) == {"receipt.json", "progress.json"}
    job.export_receipt(spec, output, receipt, "a" * 64, state["code_sha256"])
    receipt["raw"]["groups"]["text/choice"]["sample_id"] = "secret"
    with pytest.raises(ValueError, match="unexpected_metric_fields"):
        job.export_receipt(spec, output, receipt, "a" * 64, state["code_sha256"])


def test_private_worker_rejects_actual_calibration_row_in_holdout_file(tmp_path: Path) -> None:
    state = synthetic_state(tmp_path)
    path = Path(state["datasets"][0])
    rows = path.read_text().splitlines()
    row = json.loads(rows[0])
    row["meta"]["pool"] = "calibration"
    rows[0] = json.dumps(row)
    path.write_text("\n".join(rows) + "\n")
    state["spec"]["datasets"][0]["data"]["sha256"] = job.sha256_file(path)
    with pytest.raises(ValueError, match="wrong_pool"):
        job.private_samples(state)


@pytest.mark.parametrize("raw_fault", [None, "hash", "recipe"])
def test_seven_temperatures_require_actual_file_and_fit_receipt_bindings(
    tmp_path: Path, raw_fault: str | None
) -> None:
    import hashlib

    from scripts.check_tokens import ANSWERS
    from train.calibration import Binding, TemperatureArtifact, save_temperature

    official = {
        "status": "passed",
        "answer_token_ids": dict(zip(ANSWERS, range(33), strict=True)),
        "template_sha256": "c" * 64,
    }
    map_sha = hashlib.sha256(
        json.dumps(official["answer_token_ids"], sort_keys=True).encode()
    ).hexdigest()
    values = spec_values()
    raw_path = tmp_path / "raw-receipt.json"
    raw_meta = dict(
        runtime_recipe={
            "engine": "transformers",
            "dtype": "bfloat16",
            "adapter_mode": "native_lora",
        },
        model_files=values["datasets"][0]["model_files"],
        adapter_files=values["datasets"][0]["adapter_files"],
        official_template_sha256=official["template_sha256"],
        answer_token_map_sha256=map_sha,
        spec_sha256="a" * 64,
        code_sha256="b" * 64,
        logits_kind="raw_logits",
        complete_legal_logits=True,
    )
    raw_path.write_text(json.dumps(raw_meta))
    raw_sha = job.sha256_file(raw_path)
    references = []
    entries = []
    metadata = {}
    for index, key in enumerate(job.GROUP_COUNTS):
        cohort, primitive = key.split("/")
        binding = Binding.model_validate(
            dict(
                base_revision=job.stage.FROZEN_REVISION,
                adapter_weights_sha256="c" * 64,
                official_template_sha256="c" * 64,
                answer_token_map_sha256=map_sha,
                raw_logits_receipt_sha256=raw_sha,
                cohort=cohort,
                primitive=primitive,
                seed=42,
                code_revision="d" * 40,
                calibration_data_sha256="f" * 64,
            )
        )
        artifact = TemperatureArtifact(
            schema_version=1,
            method="bounded_scalar_joint_nll",
            fitter_implementation_sha256="0" * 64,
            binding=binding,
            temperature=2.0,
            temperature_min=0.05,
            temperature_max=20.0,
            rows=cal.GROUP_COUNTS[key],
            nll_before=1.0,
            nll_after=0.9,
            stage_acceptance="not_evaluated",
        )
        path = tmp_path / f"temperature-{index}.json"
        save_temperature(artifact, path)
        sha = job.sha256_file(path)
        references.append(
            {"artifact": {"path": str(path), "sha256": sha}, "binding": binding.model_dump()}
        )
        entries.append({"group": key, "sha256": sha, "binding": binding.model_dump()})
        metadata[f"temperature-{index}"] = str(path)
    receipt = tmp_path / "fit.json"
    receipt.write_text(
        json.dumps(
            dict(
                status="fitted",
                exit_code=0,
                ordinary_identity_boundary=True,
                boundary="fresh_ordinary_uid_fixed_calibration_job",
                artifacts=entries,
                spec_sha256="a" * 64,
                code_sha256="b" * 64,
                raw_logits_receipt_sha256=raw_sha,
            )
        )
    )
    values.update(
        temperatures=references,
        calibration_receipt={"path": str(receipt), "sha256": job.sha256_file(receipt)},
        calibration_raw_receipt={"path": str(raw_path), "sha256": raw_sha},
    )
    metadata["calibration-receipt"] = str(receipt)
    metadata["calibration-raw-receipt"] = str(raw_path)
    spec = job.JobSpec.model_validate(values)
    if raw_fault == "hash":
        raw_path.write_text(raw_path.read_text() + " ")
        with pytest.raises(ValueError, match="calibration_raw_receipt_binding_mismatch"):
            job.temperatures(spec, {"metadata": metadata}, official)
        return
    if raw_fault == "recipe":
        raw_meta["runtime_recipe"]["engine"] = "vllm"
        raw_path.write_text(json.dumps(raw_meta))
        changed_sha = job.sha256_file(raw_path)
        fit = json.loads(receipt.read_text())
        fit["raw_logits_receipt_sha256"] = changed_sha
        receipt.write_text(json.dumps(fit))
        values["calibration_raw_receipt"]["sha256"] = changed_sha
        values["calibration_receipt"]["sha256"] = job.sha256_file(receipt)
        spec = job.JobSpec.model_validate(values)
        with pytest.raises(ValueError, match="calibration_runtime_provenance_mismatch"):
            job.temperatures(spec, {"metadata": metadata}, official)
        return
    loaded = job.temperatures(spec, {"metadata": metadata}, official)
    assert set(loaded) == set(job.GROUP_COUNTS)
    assert all(t.temperature == 2.0 for t in loaded.values())
    with pytest.raises(ValueError, match="calibration_runtime_provenance_mismatch"):
        job.temperatures(spec, {"metadata": metadata}, {**official, "template_sha256": "0" * 64})
    Path(metadata["temperature-0"]).write_text("{}")
    with pytest.raises(ValueError, match="temperature_changed"):
        job.temperatures(spec, {"metadata": metadata}, official)


@pytest.mark.parametrize(
    "change,category",
    [
        ({"model_revision": "WRONG_BASE"}, "base_revision_missing_or_mismatched"),
        ({"video_num_frames": 8}, "video_frames_not_proven_16"),
        ({"image_max_soft_tokens": {"effective": 280}}, "image_budget_not_proven_1120"),
        ({"dtype": "float32"}, "bf16_evidence_missing_or_mismatched"),
        ({"dtype": None, "script_sha256": "unknown"}, "bf16_evidence_missing_or_mismatched"),
    ],
)
def test_dev_gap_rejects_wrong_runtime_recipe(change: dict[str, Any], category: str) -> None:
    report = {
        "model_revision": job.stage.FROZEN_REVISION,
        "video_num_frames": 16,
        "image_max_soft_tokens": {"effective": 1120},
        "dtype": "bfloat16",
    }
    assert job.dev_runtime_categories({**report, **change}, "video", job.stage.FROZEN_REVISION) == [
        category
    ]


def test_known_actual_source_proves_bf16_without_fabricated_dtype_field() -> None:
    report = {
        "model_revision": job.stage.FROZEN_REVISION,
        "video_num_frames": 16,
        "image_max_soft_tokens": {"requested": 1120, "effective": 1120},
        "script_sha256": "12d387e778a685a8f49ef223f5fa2d26538e3e6a052c7c8e5013677186ad9c68",
    }
    assert job.dev_runtime_categories(report, "video", job.stage.FROZEN_REVISION) == []
