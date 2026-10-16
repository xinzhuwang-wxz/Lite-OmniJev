"""Program coverage preserves old histories and resumes one fixed pool cumulatively."""

import hashlib
import json
from pathlib import Path

import pytest
from test_gold_inventory import sample
from test_reconcile_continuous_main_pool import integration_fixture
from test_reconcile_crossmodal_retention import module

from data.schema import Sample
from scripts.s1_text_pools import canonical_sample_sha


def test_program_profile_is_separate_ce_rank16_identity():
    code = module()
    p = code.continuous_profile(program_stratified=True)
    assert p["history_key"] == "program_stratified_invocations"
    assert p["data_sha"] == "795f69fe9474523009bd953a0436bc505e85efdd4fb0f5e7c41e4d86c3bb028f"
    assert p["base_sha"] == "960a6b661f6414a514fa3274979b3a979669544979632251f52838f1f882edcb"
    assert p["config_sha"] != code.continuous_profile(True)["config_sha"]


def test_program_metadata_uses_official_program_and_unchanged_gold(tmp_path):
    code = module()
    programs = {}
    rows = {}
    metadata = []
    examples = [
        ["events", "filter_before", "exist"],
        ["events", "get_frame", "exist"],
        ["get_frame", "exist"],
        ["exist"],
    ]
    for group, program in zip(code.PROGRAM_GROUPS, examples, strict=True):
        for gold in ("yes", "no"):
            for n in range(75):
                item = f"{group}-{gold}-{n}"
                row = sample(item, "train", "video")
                row["questions"][0]["target"]["value"] = gold
                s = Sample.model_validate(row)
                rows[item] = row
                programs[item] = program
                metadata.append(
                    {
                        "item_id": item,
                        "family_id": s.meta.family_id,
                        "program_stratum": group,
                        "official_program_sha256": hashlib.sha256(
                            json.dumps(program, sort_keys=True).encode()
                        ).hexdigest(),
                        "official_source_sha256": s.meta.source_sha256,
                        "media_sha256": s.prefix.media[0].sha256,
                        "sample_sha256": canonical_sample_sha(s),
                        "original_gold": gold,
                        "has_collision_operator": False,
                    }
                )
    path = tmp_path / "program-strata-metadata.json"
    payload = {
        "metadata_only": True,
        "target_source": "original official gold only; exclusive program strata",
        "rows": metadata,
    }
    path.write_text(json.dumps(payload))
    prep = {"stratum_counts": {g: {"yes": 75, "no": 75} for g in code.PROGRAM_GROUPS}}
    assert len(code.program_strata_binding(tmp_path, prep, rows, programs)) == 600
    payload["rows"][0]["program_stratum"] = "exist"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="program_exact_metadata"):
        code.program_strata_binding(tmp_path, prep, rows, programs)


@pytest.mark.parametrize("monitored", [False, True])
@pytest.mark.parametrize("bad_prefix", [False, True])
def test_program_first10_and_resume20_preserve_four_old_histories(
    tmp_path, monkeypatch, bad_prefix, monitored
):
    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    old_raw = root / "old-runtime.json"
    old_raw.write_text("{}")
    history_keys = (
        "continuous_invocations",
        "continuous_matched_invocations",
        "rank64_invocations",
        "video_pair_objective_invocations",
    )
    payload = json.loads(base.read_bytes())
    for key in history_keys:
        payload[key] = [{"runtime_report_observation": code.bind(old_raw)}]
    payload.update(
        actual_global_draws_seen=240,
        continuous_actual_global_draws_seen=240,
        continuous_matched_actual_global_draws_seen=240,
        rank64_actual_global_draws_seen=120,
        video_pair_objective_actual_global_draws_seen=120,
    )
    base.write_text(json.dumps(payload))
    profile = {
        **code.continuous_profile(),
        "base_sha": code.sha(base),
        "history_key": "program_stratified_invocations",
        "name": "s4-program-stratified",
    }
    monkeypatch.setattr(code, "continuous_profile", lambda *a, **kw: profile)
    monkeypatch.setattr(code, "continuous_matched_bindings", lambda *a, **kw: {})
    monkeypatch.setattr(code, "program_training_bindings", lambda *a: {})
    prep_path = root / "artifacts/prep/report.json"
    prep = json.loads(prep_path.read_bytes())
    prep["exclusions"] = {
        "scope": "explicit source-bound pilot-family exclusions; not globally unseen"
    }
    prep_path.write_text(json.dumps(prep))
    profile["prep_sha"] = code.sha(prep_path)

    def launch(start, end, base_path):
        path = invoke(start, end, base_path)
        receipt = json.loads(path.read_bytes())
        eb = receipt["exposure_binding"]
        for target in (
            eb["final_report"]["path"],
            eb["runtime_report_snapshot"]["path"],
            eb["runtime_report_path"],
        ):
            target = Path(target)
            raw = json.loads(target.read_bytes())
            raw["losses"] = [1.0] * end
            if start and bad_prefix:
                raw["losses"][0] = 2.0
            target.write_text(json.dumps(raw))

        for key in ("final_report", "runtime_report_snapshot"):
            eb[key] = code.bind(Path(eb[key]["path"]))
        receipt.pop("parent_selection")
        if start and monitored:
            receipt["policy_family"] = "monitored_full_course"
            receipt["exposure_binding"]["base_exposure_report"] = code.bind(base)
        path.write_text(json.dumps(receipt))
        return path

    out1 = root / "artifacts/exposure023"
    first = code.prepare_continuous(
        root, legacy, out1, index, launch(0, 10, base), program_stratified=True
    )
    assert len(first["appended_artifacts"]) == 1
    assert first["program_stratified_actual_global_draws_seen"] == 120
    out2 = root / "artifacts/exposure024"
    if bad_prefix:
        with pytest.raises(ValueError, match="program_resume_loss_prefix"):
            code.prepare_continuous(
                root,
                legacy,
                out2,
                index,
                launch(10, 20, out1 / "report.json"),
                program_stratified=True,
                continuity_base_path=out1 / "report.json",
                reuse_unchanged_inventory_proof=monitored,
            )
        return
    second_launch = launch(10, 20, out1 / "report.json")
    if monitored:
        from test_reconcile_continuous_main_pool import checkpoint

        current_path = root / "artifacts/train/run.json"
        current = json.loads(current_path.read_bytes())
        current["steps_completed"] = 40
        current["losses"] = [1.0] * 40
        current_checkpoint = current_path.parent / "checkpoints/step-0040"
        current["checkpoint"] = str(current_checkpoint)
        checkpoint(code, current_checkpoint, current, 40)
        current_path.write_text(json.dumps(current))
    second = code.prepare_continuous(
        root,
        legacy,
        out2,
        index,
        second_launch,
        program_stratified=True,
        continuity_base_path=out1 / "report.json",
        reuse_unchanged_inventory_proof=monitored,
        saved_invocation=monitored,
    )
    assert second["inventory_verification_mode"] == (
        "reused_unchanged_manifest" if monitored else "fresh_full_inventory"
    )
    assert second["program_stratified_invocations"][-1]["current_runtime_steps_observed"] == (
        40 if monitored else 20
    )
    assert second["program_stratified_actual_global_draws_seen"] == 240
    assert second["appended_artifacts"] == []
    assert len(second["program_stratified_invocations"]) == 2
    for key in history_keys:
        assert second[key] == payload[key]
    assert [
        second[k]
        for k in (
            "continuous_actual_global_draws_seen",
            "continuous_matched_actual_global_draws_seen",
            "rank64_actual_global_draws_seen",
            "video_pair_objective_actual_global_draws_seen",
        )
    ] == [240, 240, 120, 120]
    assert second["unknown_update_bearing_inputs"] == 0


def test_twenty_step_interval_needs_explicit_budget_and_matching_invocation():
    from test_reconcile_continuous_main_pool import final_report

    code = module()
    raw = final_report(code, 20, 40)
    raw["invocation"]["steps_requested"] = 20
    with pytest.raises(ValueError, match="continuous_cumulative_steps"):
        code.continuous_interval(raw, 20, 40)
    interval = code.continuous_interval(raw, 20, 40, invocation_steps=20)
    assert interval["actual_draw_interval"] == [240, 480]
    assert interval["actual_global_draws_seen"] == 480
    assert interval["actual_draws_this_invocation"] == 240
    raw["invocation"]["steps_requested"] = 10
    with pytest.raises(ValueError, match="continuous_invocation_receipt"):
        code.continuous_interval(raw, 20, 40, invocation_steps=20)


def test_reused_proof_is_bound_and_rejects_changed_manifest(tmp_path):
    code = module()
    base_path = tmp_path / "report.json"
    manifest = tmp_path / "inventory-manifest.json"
    audit = tmp_path / "inventory-audit.json"
    inventory = {"samples": [{"sha256": "a" * 64}], "component_ledgers": []}
    manifest.write_text(json.dumps(inventory))
    audit.write_text(json.dumps({"outcome": "isolated"}))
    base = {"output_sha256": {p.name: code.sha(p) for p in (manifest, audit)}}
    out = tmp_path / "next"
    out.mkdir()
    bindings = {}
    origin = code.reuse_unchanged_inventory(base_path, base, inventory, out, bindings)
    assert origin["manifest"] == code.bind(manifest)
    assert (out / manifest.name).read_bytes() == manifest.read_bytes()
    assert bindings[str(audit)] == code.sha(audit)
    with pytest.raises(ValueError, match="reused_inventory_changed_manifest"):
        code.reuse_unchanged_inventory(base_path, base, {**inventory, "new": True}, out, {})
    audit.write_text(json.dumps({"outcome": "rejected"}))
    with pytest.raises(ValueError, match="reused_inventory_proof_sha"):
        code.reuse_unchanged_inventory(base_path, base, inventory, out, {})


def test_unfrozen_monitored_policy_cannot_authorize_a_segment(tmp_path, monkeypatch):
    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(code, "PROGRAM_MONITORED_RUNNER_SHA", None)
    launch = invoke(0, 10, base)
    receipt = json.loads(launch.read_bytes())
    receipt["policy_family"] = "monitored_full_course"
    launch.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="program_monitored_policy_not_frozen"):
        code.prepare_continuous(
            root,
            legacy,
            root / "not-authorized",
            index,
            launch,
            program_stratified=True,
            reuse_unchanged_inventory_proof=True,
        )


def test_saved_invocation_cannot_relax_legacy_runtime_identity(tmp_path, monkeypatch):
    code, root, legacy, index, base, invoke = integration_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="saved_invocation_requires_monitored_policy"):
        code.prepare_continuous(
            root,
            legacy,
            root / "legacy-replay-rejected",
            index,
            invoke(0, 10, base),
            saved_invocation=True,
        )


def controller_fixture(tmp_path):
    code = module()
    root = tmp_path / "root"
    reports = root / "reports"
    reports.mkdir(parents=True)
    segments = []
    screens = []
    for start, end in zip((10, 20, 40, 60, 80), (20, 40, 60, 80, 100), strict=True):
        adapter = str(end).zfill(64)
        raw = {
            "steps_completed": end,
            "adapter_sha256": adapter,
            "data_sha256": code.continuous_profile(program_stratified=True)["data_sha"],
        }
        snapshot = reports / f"s4-program-stratified-runtime-step{end:04d}.json"
        final = reports / f"s4-program-stratified-training-step{end:04d}.json"
        for path in (snapshot, final):
            path.write_text(json.dumps(raw))
        launch = reports / f"s4-program-stratified-launch-step{end:04d}.json"
        launch.write_text(
            json.dumps(
                {
                    "policy_family": "monitored_full_course",
                    "decision_sha256": code.PROGRAM_MONITORED_POLICY_SHA,
                    "wrapper_sha256": code.PROGRAM_MONITORED_RUNNER_SHA,
                    "stage_gate": False,
                    "status": "completed" if end == 100 else "paused",
                    "exit_code": 0,
                    "exposure_binding": {
                        "start_step": start,
                        "end_step": end,
                        "final_report": code.bind(final),
                        "runtime_report_snapshot": code.bind(snapshot),
                    },
                }
            )
        )
        screen = reports / f"s4-program-stratified-dev50-step{end:04d}.json"
        payload = {
            "status": "completed",
            "exit_code": 0,
            "step": end,
            "paired_input_hashes_verified": 50,
            "candidate": {"count": 50},
            "adapter_sha256": adapter,
        }
        screen.write_text(json.dumps(payload))
        screens.append(payload)
        segments.append(
            {
                "start_step": start,
                "end_step": end,
                "launch": code.bind(launch),
                "screen": code.bind(screen),
                "adapter_sha256": adapter,
            }
        )
    controller = {
        "policy_family": "monitored_full_course",
        "decision_sha256": code.PROGRAM_MONITORED_POLICY_SHA,
        "status": "course_completed",
        "exit_code": 0,
        "steps_completed": 100,
        "formal_quality_gate": "not_passed_by_this_controller",
        "full_course_scope": "all1200 fixed course draws; not all S4 data",
        "segments": segments,
        "best_checked": screens[0],
    }
    path = reports / "s4-program-stratified-full-controller-001.json"
    path.write_text(json.dumps(controller))
    output = tmp_path / "exposure"
    output.mkdir()
    return code, root, path, controller, output


def test_closed_controller_is_a_bound_summary_not_an_extra_optimizer_invocation(tmp_path):
    code, root, path, controller, output = controller_fixture(tmp_path)
    bindings = {}
    result = code.program_controller_summary(root, path, controller, output, bindings)
    assert str(path) not in bindings
    assert result["immutable_snapshot"]["sha256"] == code.sha(path)
    assert result["role"].endswith("not an additional optimizer invocation")
    assert len(bindings) == 21


@pytest.mark.parametrize("bad", ["quality_pass", "missing_segment", "mutated_screen"])
def test_controller_summary_must_preserve_quality_failure_and_complete_bound_chain(tmp_path, bad):
    code, root, path, controller, output = controller_fixture(tmp_path)
    if bad == "quality_pass":
        controller["formal_quality_gate"] = "passed"
    elif bad == "missing_segment":
        controller["segments"].pop()
    else:
        Path(controller["segments"][0]["screen"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="program_controller_"):
        code.program_controller_summary(root, path, controller, output, {})
