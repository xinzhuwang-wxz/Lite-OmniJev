"""Third coverage segment selects250ef independently of polarity branch weights."""

import copy

import pytest
from test_reconcile_crossmodal_retention import module


def test_third_paths_preserve_coverage_history(tmp_path):
    code = module()
    p = code.pilot_paths(tmp_path, coverage_segment003=True)
    assert p[:-1] == code.pilot_paths(tmp_path, coverage_segment002=True)
    assert p[-1][1] == tmp_path / "reports/s4-temporal-coverage-training-003.json"
    assert p[-1][3] == tmp_path / "artifacts/s4-temporal-coverage-pilot-003"
    assert len(code.pilot_paths(tmp_path)) == 2


@pytest.mark.parametrize("bad", [None, "f2db", "polarity", "live", "placeholder"])
def test_third_parent_requires_root_selected_second_segment(bad):
    code = module()
    chosen = {
        "path": "/second/adapter",
        "weights_sha256": "250ef3e3c8a5359e6d7b6aaaeafa6d12b253603669a4b9f8a7a70c52c4b72707",
        "config_sha256": "a" * 64,
    }
    prep = {
        "status": "data_ready_only",
        "exit_code": 0,
        "pending_parent_selection": True,
        "training_authorized": False,
        "inventory_isolation": "isolated",
        "source_admission": "verified_existing_scope",
        "init_adapter": {"path": "/placeholder"},
    }
    launch = {
        "status": "completed",
        "exit_code": 0,
        "parent_selection": "root_selected_second_segment_after_retention_and_full_video",
        "selected_init_adapter": chosen,
    }
    final = {
        "init_adapter": copy.deepcopy(chosen),
        "config": {
            "sample_count": 120,
            "steps": 10,
            "accumulation": 12,
            "seed": 42,
            "midway_step": 5,
            "checkpoint_every": 5,
            "init_adapter": chosen,
        },
    }
    if bad == "f2db":
        final["init_adapter"]["weights_sha256"] = (
            "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
        )
    elif bad == "polarity":
        final["init_adapter"]["weights_sha256"] = (
            "e3e155f884a85928315f75373ce2ff87a658d4057bd9b368edc43716ba633f93"
        )
    elif bad == "live":
        launch["status"] = "training"
    elif bad == "placeholder":
        final["init_adapter"] = prep["init_adapter"]
    if bad:
        with pytest.raises(ValueError):
            code.segment002_parent(prep, launch, final, segment003=True)
    else:
        code.segment002_parent(prep, launch, final, segment003=True)


def test_failed_third_updates_keep_terminal_contract():
    code = module()
    r = {
        "status": "failed",
        "exit_code": 1,
        "steps_completed": 3,
        "config": {"sample_count": 120, "steps": 10, "accumulation": 12, "seed": 42},
    }
    code.claim_terminal(r, "s4-temporal-coverage-segment003")
    assert r["steps_completed"] > 0
