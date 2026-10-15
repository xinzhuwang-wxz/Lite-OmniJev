"""CPU-only Brier objective identity and terminal exposure regression checks."""

import json

import pytest
from test_reconcile_temporal_curriculum import fixture as temporal_fixture


def fixture(root):
    code, index, _, _, prep, _ = temporal_fixture(root)
    config = {"seed": 42, "sample_count": 120, "steps": 10, "accumulation": 12}
    (prep / "config-ce.json").write_text(json.dumps(config))
    pr = json.loads((prep / "report.json").read_text())
    pr["output_sha256"]["config-ce.json"] = code.sha(prep / "config-ce.json")
    (prep / "report.json").write_text(json.dumps(pr))
    code.prepare(
        root,
        root / "legacy",
        root / "artifacts/s4-pilot-exposure-005",
        index,
        temporal_curriculum=True,
    )
    _, final, run, _ = code.pilot_paths(root, temporal_brier=True)[-1]
    run.mkdir()
    (run / "data.jsonl").write_bytes((prep / "train.jsonl").read_bytes())
    cfg = {**config, "brier_weight": 0.5}
    report = {
        "status": "completed",
        "exit_code": 0,
        "steps_completed": 10,
        "data_sha256": code.sha(run / "data.jsonl"),
        "init_adapter": pr["init_adapter"],
        "config": {**cfg, "init_adapter": pr["init_adapter"]},
    }
    final.write_text(json.dumps(report))
    (run / "run.json").write_text(json.dumps(report))
    experiment = root / "artifacts/s4-temporal-brier-pilot-001"
    experiment.mkdir()
    cp = experiment / "config-brier.json"
    cp.write_text(json.dumps(cfg))
    launch = root / "reports/s4-temporal-brier-launch-001.json"
    launch.write_text(
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "prepared_report_sha256": code.sha(prep / "report.json"),
                "config_sha256": code.sha(cp),
            }
        )
    )
    return code, index, final, run, cp, launch


def test_brier_only_public_config_contract():
    from test_reconcile_crossmodal_retention import module

    code = module()
    base = {"seed": 42, "sample_count": 120, "steps": 10, "accumulation": 12}
    code.brier_config_matches(base, {**base, "brier_weight": 0.5})
    with pytest.raises(ValueError, match="brier_objective_weight"):
        code.brier_config_matches(base, {**base, "brier_weight": 0.4})
    with pytest.raises(ValueError, match="brier_only_config_difference"):
        code.brier_config_matches(base, {**base, "brier_weight": 0.5, "learning_rate": 0.00002})


@pytest.mark.parametrize("failed_steps", [None, 3])
def test005_base_preserves_all_prior_runs_and_records_same_data_update(tmp_path, failed_steps):
    root = tmp_path
    code, index, final, run, _, launch = fixture(root)
    assert len(code.pilot_paths(root, temporal_brier=True)) == 7
    if failed_steps:
        r = json.loads(final.read_text())
        r.update(status="failed", exit_code=1, steps_completed=failed_steps)
        final.write_text(json.dumps(r))
        (run / "run.json").write_text(json.dumps(r))
        launch_report = json.loads(launch.read_text())
        launch_report.update(status="failed", exit_code=1)
        launch.write_text(json.dumps(launch_report))
    result = code.prepare(root, root / "legacy", root / "out", index, temporal_brier=True)
    assert result["temporal_brier_included"] and result["temporal_curriculum_included"]
    assert result["base_inventory"]["path"] == str(
        root / "artifacts/s4-pilot-exposure-005/inventory-manifest.json"
    )
    assert (
        result["pilots"][-1]["update_bearing"]
        and result["pilots"][-1]["canonical_equivalence"]["rows"] == 120
    )
    assert len(result["appended_artifacts"]) == 0  # Same canonical input already present in005.
    assert (
        result["discovered_update_bearing_reports"] == 14
        and result["unknown_update_bearing_inputs"] == 0
    )


@pytest.mark.parametrize(
    "change", ["live", "weight", "other_config", "launch_hash", "old005_input"]
)
def test_invalid_or_live_brier_rejected(tmp_path, change):
    code, index, final, run, cp, launch = fixture(tmp_path)
    if change == "old005_input":
        old = tmp_path / "reports/s4-temporal-curriculum-training-001.json"
        old.write_text(old.read_text() + "\n")
    elif change == "launch_hash":
        v = json.loads(launch.read_text())
        v["config_sha256"] = "0" * 64
        launch.write_text(json.dumps(v))
    else:
        r = json.loads(final.read_text())
        if change == "live":
            r["status"] = "running"
        elif change == "weight":
            r["config"]["brier_weight"] = 0.4
        else:
            r["config"]["learning_rate"] = 0.00002
        final.write_text(json.dumps(r))
        (run / "run.json").write_text(json.dumps(r))
    with pytest.raises(ValueError):
        code.prepare(tmp_path, tmp_path / "legacy", tmp_path / "out", index, temporal_brier=True)
    assert json.loads((tmp_path / "out/report.json").read_text())["status"] == "rejected"


def test_only_exact_bound_init_metadata_may_be_removed():
    from test_reconcile_crossmodal_retention import module

    code = module()
    base = {"seed": 42, "sample_count": 120, "steps": 10, "accumulation": 12}
    init = {"path": "/tmp/adapter", "weights_sha256": "a" * 64, "config_sha256": "b" * 64}
    actual = {**base, "brier_weight": 0.5, "init_adapter": init}
    code.brier_config_matches(base, actual, init)
    with pytest.raises(ValueError, match="brier_report_init_metadata"):
        code.brier_config_matches(base, actual)
    with pytest.raises(ValueError, match="brier_report_init_metadata"):
        code.brier_config_matches(base, actual, {**init, "weights_sha256": "c" * 64})
    with pytest.raises(ValueError):
        code.brier_config_matches(base, {**actual, "unknown_field": 1}, init)
