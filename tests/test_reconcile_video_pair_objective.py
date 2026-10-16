"""Ranking exposure is independent from CE and never recounts the shared pool."""

import json
import shutil

import pytest
from test_reconcile_continuous_main_pool import final_report, fixed_config, integration_fixture
from test_reconcile_crossmodal_retention import module


def test_objective_profile_has_own_fixed_identity_and_history():
    code = module()
    objective = code.continuous_profile(video_objective=True)
    ce = code.continuous_profile(rank64=True)
    assert objective["history_key"] == "video_pair_objective_invocations"
    assert objective["data_sha"] == ce["data_sha"]
    assert objective["prep_sha"] != ce["prep_sha"]
    assert objective["config_sha"] != ce["config_sha"]
    assert (
        objective["base_sha"] == "24b2bcb85e7067c3759cba3f007782c46b174695fd77399da15bc468cfb3a35e"
    )


def test_ce_profile_refuses_video_objective_before_runtime_normalization():
    code = module()
    prep = dict(
        status="data_ready_only",
        parent=None,
        training_authorized=False,
        inventory_isolation="isolated",
    )
    with pytest.raises(ValueError, match="continuous_video_objective_scope"):
        code.continuous_fixed_identity(
            prep, {**fixed_config(), "video_pair_loss_weight": 0.1}, final_report(code)
        )


def test_objective_keeps_all_three_old_histories_and_counts_only_new_interval(
    tmp_path, monkeypatch
):
    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    first_dir = root / "artifacts/old-exposure"
    first = code.prepare_continuous(root, legacy, first_dir, index, invoke(0, 10, base))
    base = first_dir / "report.json"
    payload = json.loads(base.read_bytes())
    old_history = payload["continuous_invocations"]
    payload.update(
        continuous_matched_invocations=old_history,
        rank64_invocations=old_history,
        continuous_actual_global_draws_seen=240,
        continuous_matched_actual_global_draws_seen=240,
        rank64_actual_global_draws_seen=120,
        rank64_derivative_identity={"numerical_parity_validated": False},
        rank64_independent_candidate_included=True,
        actual_global_draws_seen=240,
    )
    base.write_text(json.dumps(payload))
    pooldir = root / "artifacts/s4-continuous-matched-pool-001"
    shutil.copytree(root / "artifacts/prep", pooldir)
    monkeypatch.setattr(code, "MATCHED_CONTINUOUS_PREP_SHA", code.sha(pooldir / "report.json"))
    monkeypatch.setattr(code, "RANK64_WEIGHTS_SHA", code.CONTINUOUS_PARENT_SHA)
    profile = dict(
        code.continuous_profile(),
        base_sha=code.sha(base),
        history_key="video_pair_objective_invocations",
        name="s4-video-pair-objective",
    )
    monkeypatch.setattr(code, "continuous_profile", lambda *a, **kw: profile)
    monkeypatch.setattr(code, "continuous_matched_bindings", lambda *a: {})
    monkeypatch.setattr(code, "continuous_fixed_identity", lambda *a, **kw: None)
    monkeypatch.setattr(code, "video_objective_bindings", lambda *a: ({}, {"audited_pairs": 300}))
    # A separate invocation has its own runtime and snapshot, while the original remains immutable.
    launcher = invoke(0, 5, base)
    launch = json.loads(launcher.read_bytes())
    raw = root / "artifacts/train/run.json"
    old_raw = first_dir / "old-run.json"
    old_raw.write_bytes((root / "artifacts/train/run-invocation-0010.json").read_bytes())
    payload = json.loads(base.read_bytes())
    for key in ("continuous_invocations", "continuous_matched_invocations", "rank64_invocations"):
        payload[key][0]["runtime_report_observation"] = code.bind(old_raw)
    base.write_text(json.dumps(payload))
    profile["base_sha"] = code.sha(base)
    launch["exposure_binding"]["base_exposure_report"] = code.bind(base)
    launch.pop("parent_selection")
    launcher.write_text(json.dumps(launch))
    out = root / "artifacts/exposure022"
    result = code.prepare_continuous(root, legacy, out, index, launcher, video_objective=True)
    assert result["video_pair_objective_actual_global_draws_seen"] == 60
    assert result["continuous_actual_global_draws_seen"] == 240
    assert result["continuous_matched_actual_global_draws_seen"] == 240
    assert result["rank64_actual_global_draws_seen"] == 120
    for key in ("continuous_invocations", "continuous_matched_invocations", "rank64_invocations"):
        assert result[key] == payload[key]
    assert result["appended_artifacts"] == []
    assert result["unknown_update_bearing_inputs"] == 0
    assert all(
        result["all_bound_input_sha256"].get(p) == h
        for p, h in first["all_bound_input_sha256"].items()
    )
    assert str(raw) not in result["all_bound_input_sha256"]
    assert result["parity_validated"] is result["parent_quality_inherited"] is False
