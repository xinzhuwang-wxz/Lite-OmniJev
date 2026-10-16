"""Second strict continuous profile preserves legacy history and counts its own intervals."""

import json

import pytest
from test_reconcile_continuous_main_pool import integration_fixture
from test_reconcile_crossmodal_retention import module


def test_only_frozen_profile_is_executable_and_histories_have_distinct_keys(monkeypatch):
    code = module()
    old = code.continuous_profile()
    matched = code.continuous_profile(True)
    assert old["history_key"] == "continuous_invocations"
    assert matched["history_key"] == "continuous_matched_invocations"
    assert matched["base_sha"] == code.MATCHED_CONTINUOUS_BASE_SHA
    assert matched["data_sha"] != old["data_sha"]
    monkeypatch.setattr(code, "MATCHED_CONTINUOUS_PREP_SHA", None)
    with pytest.raises(ValueError, match="profile_not_frozen"):
        code.continuous_profile(True)


def test_two_matched_invocations_keep_original_two_snapshots_and_separate_draw_totals(
    tmp_path, monkeypatch
):
    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    legacy_raw = root / "legacy-continuous-run.json"
    legacy_raw.write_text("{}")
    old_history = [
        {"runtime_report_observation": code.bind(legacy_raw), "end_step": 10, "status": "paused"},
        {"runtime_report_observation": code.bind(legacy_raw), "end_step": 20, "status": "paused"},
    ]
    base_payload = json.loads(base.read_text())
    base_payload.update(
        continuous_invocations=old_history,
        actual_global_draws_seen=240,
        continuous_actual_global_draws_seen=240,
        native_actual_draws_seen=120,
        specialist_actual_draws_seen=120,
        text_matched_actual_draws_seen=120,
    )
    base.write_text(json.dumps(base_payload))
    monkeypatch.setattr(code, "MATCHED_CONTINUOUS_BASE_SHA", code.sha(base))
    monkeypatch.setattr(code, "MATCHED_CONTINUOUS_PREP_SHA", code.CONTINUOUS_PREP_SHA)
    monkeypatch.setattr(code, "MATCHED_CONTINUOUS_DATA_SHA", code.CONTINUOUS_DATA_SHA)
    monkeypatch.setattr(code, "MATCHED_CONTINUOUS_CONFIG_SHA", code.CONTINUOUS_CONFIG_SHA)
    monkeypatch.setattr(code, "continuous_matched_bindings", lambda *args: {})
    out1 = root / "artifacts/exposure019"
    first = code.prepare_continuous(
        root, legacy, out1, index, invoke(0, 10, base), matched_pool=True
    )
    assert first["continuous_invocations"] == old_history
    assert first["actual_global_draws_seen"] == 240
    assert first["continuous_matched_actual_global_draws_seen"] == 120
    assert len(first["continuous_matched_invocations"]) == 1
    assert str(legacy_raw) not in first["all_bound_input_sha256"]
    out2 = root / "artifacts/exposure020"
    second = code.prepare_continuous(
        root, legacy, out2, index, invoke(10, 20, out1 / "report.json"), matched_pool=True
    )
    assert second["continuous_invocations"] == old_history
    assert second["continuous_matched_actual_global_draws_seen"] == 240
    assert second["actual_global_draws_seen"] == 240
    assert second["appended_artifacts"] == []
    assert len(second["continuous_matched_invocations"]) == 2
    assert (
        second["specialist_actual_draws_seen"] == 120
        and second["text_matched_actual_draws_seen"] == 120
    )


def test_profile_does_not_allow_cross_branch_parent_or_config_identity():
    from test_reconcile_continuous_main_pool import final_report, fixed_config

    code = module()
    final = final_report(code)
    prep = {
        "status": "data_ready_only",
        "parent_model": None,
        "training_authorized": False,
        "inventory_isolation": "isolated",
    }
    code.continuous_fixed_identity(prep, fixed_config(), final)
    final["init_adapter"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="continuous_original_f2db_parent"):
        code.continuous_fixed_identity(prep, fixed_config(), final)
