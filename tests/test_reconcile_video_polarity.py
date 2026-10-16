"""Single-arm exposure preserves failures and prevents shared replay current duplication."""

import copy

import pytest
from test_reconcile_crossmodal_retention import module


def reports(code, arm="control"):
    init = {
        "path": "/parent",
        "weights_sha256": "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3",
        "config_sha256": "a" * 64,
    }
    launch = {
        "arm": arm,
        "status": "completed",
        "exit_code": 0,
        "selected_init_adapter": init,
        "training_limit_seconds": 240,
        "dev50_limit_seconds": 180,
    }
    final = {
        "status": "completed",
        "exit_code": 0,
        "steps_completed": 10,
        "init_adapter": init,
        "data_sha256": code.POLARITY_DATA[arm],
        "config": {
            "sample_count": 120,
            "steps": 10,
            "accumulation": 12,
            "seed": 42,
            "midway_step": 5,
            "checkpoint_every": 5,
            "init_adapter": init,
        },
    }
    return launch, final


@pytest.mark.parametrize("arm", ["control", "polarity"])
def test_single_arm_keeps_all_prior_paths(tmp_path, arm):
    code = module()
    paths = code.pilot_paths(tmp_path, polarity_arm=arm)
    assert paths[:-1] == code.pilot_paths(tmp_path, coverage_segment002=True)
    assert paths[-1][1] == tmp_path / "reports" / f"s4-video-polarity-{arm}-training-001.json"
    assert paths[-1][3] == tmp_path / "artifacts/s4-video-polarity-pilot-001" / arm
    assert len(code.pilot_paths(tmp_path)) == 2


def test_failed_real_updates_are_valid_exposures():
    code = module()
    launch, final = reports(code)
    code.polarity_terminal_binding("control", launch, final)
    launch.update(status="failed", exit_code=1)
    final.update(status="failed", exit_code=1, steps_completed=3)
    code.polarity_terminal_binding("control", launch, final)
    assert final["steps_completed"] > 0


@pytest.mark.parametrize(
    "bad", ["live_launch", "live_train", "data", "parent", "steps", "objective", "arm"]
)
def test_reject_nonterminal_or_wrong_runtime_identity(bad):
    code = module()
    launch, final = reports(code)
    arm = "control"
    if bad == "live_launch":
        launch["status"] = "dev50_running"
    elif bad == "live_train":
        final["status"] = "running"
    elif bad == "data":
        final["data_sha256"] = code.POLARITY_DATA["polarity"]
    elif bad == "parent":
        final["init_adapter"] = copy.deepcopy(final["init_adapter"])
        final["init_adapter"]["weights_sha256"] = "0" * 64
    elif bad == "steps":
        final["steps_completed"] = 9
    elif bad == "objective":
        final["config"]["brier_weight"] = 0.5
    else:
        arm = "both"
    with pytest.raises(ValueError):
        code.polarity_terminal_binding(arm, launch, final)


def test_previous_current_inventory_becomes_history_before_new_arm():
    code = module()
    combined = {
        "samples": [
            {"path": "old009", "expected_pool": "train", "exposure": "current"},
            {"path": "control", "expected_pool": "train", "exposure": "current"},
            {"path": "dev", "expected_pool": "dev", "exposure": "current"},
        ]
    }
    artifact = {"path": "polarity", "expected_pool": "train", "exposure": "current"}
    code.arm_current_inventory(combined, artifact)
    assert [r["exposure"] for r in combined["samples"]] == [
        "historical",
        "historical",
        "current",
        "current",
    ]
    assert [
        r["path"]
        for r in combined["samples"]
        if r["expected_pool"] == "train" and r["exposure"] == "current"
    ] == ["polarity"]
