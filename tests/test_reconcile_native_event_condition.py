"""Native official-gold pilot is independent of the stopped continuous branch."""

import copy
import json

import pytest
from test_reconcile_crossmodal_retention import module


def contracts(code):
    parent = {
        "path": "/f2db",
        "weights_sha256": code.CONTINUOUS_PARENT_SHA,
        "config_sha256": "a" * 64,
    }
    prep = {
        "status": "data_ready_only",
        "exit_code": 0,
        "parent_model": None,
        "training_authorized": False,
        "pending_parent_selection": True,
        "inventory_isolation": "isolated",
        "video_rows": 60,
        "video_families": 60,
        "original_official_gold": "unchanged; proxy is never a target or admission/quality gate",
    }
    launch = {
        "selected_init_adapter": parent,
        "parent_selection": "root_selected_f2db_balanced_retention_and_full_video_evidence",
    }
    final = {
        "status": "completed",
        "exit_code": 0,
        "steps_completed": 10,
        "init_adapter": parent,
        "config": {
            "init_adapter": parent,
            "seed": 42,
            "steps": 10,
            "sample_count": 120,
            "accumulation": 12,
            "video_num_frames": 16,
            "image_max_soft_tokens": 1120,
            "rank": 16,
            "alpha": 32,
        },
    }
    return prep, launch, final


def test_native_path_only_adds_new_slice_to_bound_base_history(tmp_path):
    code = module()
    paths = code.pilot_paths(tmp_path, native_event_condition=True)
    assert paths == [
        (
            "s4-native-event-condition",
            tmp_path / "reports/s4-native-event-condition-training-001.json",
            tmp_path / "artifacts/s4-native-event-condition-train-001",
            tmp_path / "artifacts/s4-native-event-condition-pilot-001",
        )
    ]
    assert len(code.pilot_paths(tmp_path)) == 2


@pytest.mark.parametrize(
    "bad", [None, "continuous", "wrong-parent", "new-loss", "proxy-gold", "paused"]
)
def test_native_independent_gold_ce_budget_and_parent(bad):
    code = module()
    prep, launch, final = contracts(code)
    if bad == "continuous":
        final["resumed_from"] = "/continuous/checkpoints/step-0020"
    elif bad == "wrong-parent":
        final["init_adapter"] = {**final["init_adapter"], "weights_sha256": "0" * 64}
    elif bad == "new-loss":
        final["config"]["brier_weight"] = 0.5
    elif bad == "proxy-gold":
        prep["original_official_gold"] = "proxy replaces gold"
    elif bad == "paused":
        final["status"] = "paused"
    if bad:
        with pytest.raises(ValueError):
            code.native_parent_binding(prep, launch, final)
    else:
        code.native_parent_binding(prep, launch, final)


def test_native_base_keeps_full015_bindings_and_rejects_mutation(tmp_path, monkeypatch):
    code = module()
    base = tmp_path / "artifacts/s4-pilot-exposure-015"
    base.mkdir(parents=True)
    source = tmp_path / "immutable.json"
    source.write_text("{}")
    inv = base / "inventory-manifest.json"
    inv.write_text("{}")
    report = base / "report.json"
    report.write_text(
        json.dumps(
            {
                "exit_code": 0,
                "unknown_update_bearing_inputs": 0,
                "all_bound_input_sha256": {str(source): code.sha(source)},
                "output_sha256": {"inventory-manifest.json": code.sha(inv)},
            }
        )
    )
    monkeypatch.setattr(code, "NATIVE_BASE_SHA", code.sha(report))
    path, binding = code.base_inventory(
        tmp_path, crossmodal_retention=False, native_event_condition=True
    )
    assert path == inv and binding["sha256"] == code.sha(report)
    source.write_text('{"mutated":true}')
    with pytest.raises(ValueError, match="native_prior_bindings_mutated"):
        code.base_inventory(tmp_path, crossmodal_retention=False, native_event_condition=True)


def test_failed_partial_native_remains_update_bearing():
    code = module()
    prep, launch, final = contracts(code)
    final = copy.deepcopy(final)
    final.update(status="failed", exit_code=1, steps_completed=3)
    code.native_parent_binding(prep, launch, final)
