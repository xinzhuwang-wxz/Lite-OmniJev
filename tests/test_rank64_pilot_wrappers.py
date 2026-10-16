"""Fail-closed CPU checks for root-launched rank64 research wrappers; no GPU evidence."""

import copy
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".scratch/s4"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def launch():
    return load("launch-rank64-matched-pilot.py")


@pytest.fixture
def baseline():
    return load("evaluate-rank64-initializer-dev50.py")


def config():
    return {
        "rank": 16,
        "alpha": 32,
        "sample_count": 1200,
        "steps": 100,
        "accumulation": 12,
        "seed": 42,
        "learning_rate": 2e-5,
        "video_num_frames": 16,
        "image_max_soft_tokens": 1120,
        "media_roots": ["/fixed"],
        "expected_data_versions": ["fixed-version"],
        "brier_weight": 0,
        "preparation_mode": "lazy",
    }


def review_inputs(launch):
    report = {
        "status": "completed",
        "exit_code": 0,
        "gpu_requested": True,
        "initializer_sha256": launch.INIT_SHA,
        "initializer_config_sha256": launch.INIT_CONFIG_SHA,
        "expansion_report_sha256": launch.EXPANSION_SHA,
        "policy_sha256": launch.POLICY_SHA,
        "parity_validated": False,
        "parent_quality_inherited": False,
        "paired_input_hashes_verified": 50,
        "candidate": {"count": 50, "accuracy": 0.64, "brier": 0.46},
    }
    decision = {
        "decision": "approve_first10",
        "initializer_sha256": launch.INIT_SHA,
        "baseline_report_sha256": "a" * 64,
        "policy_sha256": launch.POLICY_SHA,
        "max_steps": 10,
        "max_seconds": 360,
    }
    return report, decision


def test_config_changes_only_rank_alpha_and_keeps_replay(launch):
    original = config()
    result = launch.rank64_config(original)
    assert {k for k in result if result[k] != original[k]} == {"rank", "alpha"}
    assert (result["rank"], result["alpha"]) == (64, 128)
    assert original == config()


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 43),
        ("learning_rate", 1e-4),
        ("video_num_frames", 8),
        ("image_max_soft_tokens", 280),
        ("sample_count", 120),
        ("steps", 10),
        ("brier_weight", 1),
    ],
)
def test_drifted_training_recipe_rejected(launch, field, value):
    original = config()
    original[field] = value
    with pytest.raises(ValueError):
        launch.rank64_config(original)


def test_root_review_is_bound_to_actual_baseline_not_parity(launch):
    report, decision = review_inputs(launch)
    launch.validate_baseline_review(report, "a" * 64, decision)
    for key, value in [
        ("status", "cpu_prepared_only"),
        ("gpu_requested", False),
        ("initializer_sha256", "wrong"),
        ("parity_validated", True),
        ("parent_quality_inherited", True),
        ("paired_input_hashes_verified", 49),
    ]:
        changed = {**report, key: value}
        with pytest.raises(ValueError):
            launch.validate_baseline_review(changed, "a" * 64, decision)


@pytest.mark.parametrize(
    "metric,value",
    [
        ("count", 49),
        ("accuracy", 0.62),
        ("brier", 0.468),
        ("accuracy", float("nan")),
        ("brier", float("inf")),
    ],
)
def test_own_baseline_shortlist_cannot_be_overridden_by_review(launch, metric, value):
    report, decision = review_inputs(launch)
    report["candidate"][metric] = value
    with pytest.raises(ValueError):
        launch.validate_baseline_review(report, "a" * 64, decision)


@pytest.mark.parametrize(
    "key,value",
    [
        ("decision", "prepared"),
        ("baseline_report_sha256", "bad"),
        ("max_steps", 100),
        ("max_seconds", 361),
        ("parity_validated", True),
    ],
)
def test_review_rejects_wrong_binding_or_expanded_budget(launch, key, value):
    report, decision = review_inputs(launch)
    decision[key] = value
    with pytest.raises(ValueError):
        launch.validate_baseline_review(report, "a" * 64, decision)


def test_deployed_source_must_match_every_pinned_archive_file(launch, tmp_path, monkeypatch):
    archive = tmp_path / "source.tar"
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_bytes(b"original")
    with tarfile.open(archive, "w") as bundle:
        info = tarfile.TarInfo("module.py")
        info.size = 8
        bundle.addfile(info, io.BytesIO(b"original"))
    monkeypatch.setattr(launch, "ARCHIVE_SHA", hashlib.sha256(archive.read_bytes()).hexdigest())
    launch.verify_source_projection(source, archive)
    (source / "module.py").write_bytes(b"edited")
    with pytest.raises(ValueError, match="deployed source changed"):
        launch.verify_source_projection(source, archive)


def test_baseline_requires_failed_control_bound_exact_initializer(baseline):
    control = {
        "status": "completed",
        "exit_code": 0,
        "gpu_requested": True,
        "same_process": True,
        "expanded_weights_sha256": baseline.INIT_SHA,
        "expanded_config_sha256": baseline.CONFIG_SHA,
        "source_weights_sha256": baseline.PARENT_SHA,
        "expansion_report_sha256": baseline.EXPANSION_SHA,
        "parity_validated": False,
        "verified_complete_input_mapping": 12,
    }
    baseline.validate_failed_control(control)
    with pytest.raises(ValueError):
        baseline.validate_failed_control({**control, "parity_validated": True})
    with pytest.raises(ValueError):
        baseline.validate_failed_control({**control, "expanded_weights_sha256": "bad"})


def test_official_runner_child_timeout_and_explicit_execute(baseline, monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, "SOURCE", tmp_path)
    monkeypatch.setattr(baseline, "sha", lambda path: baseline.RUNNER_SHA)
    observed = []

    def run(command, **kwargs):
        observed.append((command, kwargs))
        compile(command[2], "child", "exec")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(baseline.subprocess, "run", run)
    for execute in [False, True]:
        baseline.run_evaluator(tmp_path / "adapter", "a" * 64, tmp_path / "output", execute)
        command, kwargs = observed[-1]
        assert command[-1] == str(int(execute))
        assert kwargs["timeout"] == 180
        assert kwargs["cwd"] == tmp_path


def test_paired_baseline_detects_real_input_drift(baseline, tmp_path):
    first, second = tmp_path / "parent", tmp_path / "candidate"
    for folder, adapter in [(first, baseline.PARENT_SHA), (second, baseline.INIT_SHA)]:
        folder.mkdir()
        receipt = {
            "status": "completed",
            "exit_code": 0,
            "identity": {
                "adapter_sha256": adapter,
                "adapter_config_sha256": adapter,
                "data_sha256": "same",
                "num_frames": 16,
            },
            "metrics": {"count": 50, "accuracy": 0.66, "brier": 0.44},
        }
        (folder / "receipt.json").write_text(json.dumps(receipt))
        rows = [
            {
                "id": str(i),
                "input_sha256": str(i),
                "input_summary_sha256": str(i),
                "answer_token_ids": {"yes": 1, "no": 2, "U": 3},
            }
            for i in range(50)
        ]
        (folder / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    baseline.paired_metrics(first, second, baseline.PARENT_SHA, baseline.INIT_SHA)
    rows[4]["input_sha256"] = "different input"
    (second / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="inputs/mapping"):
        baseline.paired_metrics(first, second, baseline.PARENT_SHA, baseline.INIT_SHA)


def test_trained_candidate_must_improve_own_baseline_and_reach34():
    candidate = load("evaluate-rank64-matched-dev50.py")
    own = {"count": 50, "accuracy": 0.64, "brier": 0.46}
    assert candidate.passes_screen(own, {"count": 50, "accuracy": 0.68, "brier": 0.45})
    assert not candidate.passes_screen(own, {"count": 50, "accuracy": 0.66, "brier": 0.40})
    assert not candidate.passes_screen(own, {"count": 50, "accuracy": 0.68, "brier": 0.47})
    stronger = copy.deepcopy(own)
    stronger["accuracy"] = 0.68
    assert not candidate.passes_screen(stronger, {"count": 50, "accuracy": 0.70, "brier": 0.40})
