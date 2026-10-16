"""Specialist storage60 and actual120 draws retain native gold and independent CE budget."""

from collections import Counter

import pytest
from test_reconcile_crossmodal_retention import module
from test_reconcile_native_event_condition import contracts
from test_temporal_curriculum_preparation import row


def specialist_contracts(code):
    prep, launch, final = contracts(code)
    prep.update(
        unique_video_families=60, actual_video_draws=120, draws_per_sample=2, new_source_families=0
    )
    final["config"].update(sample_count=60, modality="video")
    return prep, launch, final


def test_specialist_source_rows_cycle_exactly_twice_in_frozen_order():
    code = module()
    samples = [row(str(i)) for i in range(60)]
    draws = code.specialist_draws(samples)
    assert len(draws) == 120 and draws[:60] == draws[60:]
    assert Counter(s.id for s in draws) == {s.id: 2 for s in samples}
    assert {s.id for s in draws} == {s.id for s in samples}
    with pytest.raises(ValueError, match="specialist60_video_train"):
        code.specialist_draws(samples[:59])


@pytest.mark.parametrize("bad", [None, "stored120", "new-family", "onepass", "resume", "steps20"])
def test_specialist_budget_is_ten_steps_with_sixty_storage_rows(bad):
    code = module()
    prep, launch, final = specialist_contracts(code)
    if bad == "stored120":
        final["config"]["sample_count"] = 120
    elif bad == "new-family":
        prep["new_source_families"] = 60
    elif bad == "onepass":
        prep["actual_video_draws"] = 60
    elif bad == "resume":
        final["resumed_from"] = "/continuous/checkpoints/step-0020"
    elif bad == "steps20":
        final["config"]["steps"] = 20
    if bad:
        with pytest.raises(ValueError):
            code.native_parent_binding(prep, launch, final, specialist=True)
    else:
        code.native_parent_binding(prep, launch, final, specialist=True)


def test_specialist_selects_only_its_delta_and_keeps_native_path_default(tmp_path):
    code = module()
    paths = code.pilot_paths(tmp_path, video_specialist=True)
    assert paths == [
        (
            "s4-video-specialist",
            tmp_path / "reports/s4-video-specialist-training-001.json",
            tmp_path / "artifacts/s4-video-specialist-train-001",
            tmp_path / "artifacts/s4-video-specialist-pilot-001",
        )
    ]
    assert (
        code.pilot_paths(tmp_path, native_event_condition=True)[0][0] == "s4-native-event-condition"
    )


def test_specialist_base_uses016_pin_and_retains_immutable_bindings(tmp_path, monkeypatch):
    import json

    code = module()
    base = tmp_path / "artifacts/s4-pilot-exposure-016"
    base.mkdir(parents=True)
    source = tmp_path / "old-snapshot.json"
    source.write_text("{}")
    manifest = base / "inventory-manifest.json"
    manifest.write_text("{}")
    report = base / "report.json"
    report.write_text(
        json.dumps(
            {
                "exit_code": 0,
                "unknown_update_bearing_inputs": 0,
                "all_bound_input_sha256": {str(source): code.sha(source)},
                "output_sha256": {"inventory-manifest.json": code.sha(manifest)},
            }
        )
    )
    monkeypatch.setattr(code, "SPECIALIST_BASE_SHA", code.sha(report))
    path, binding = code.base_inventory(tmp_path, crossmodal_retention=False, video_specialist=True)
    assert path == manifest and binding["sha256"] == code.sha(report)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="native_prior_bindings_mutated"):
        code.base_inventory(tmp_path, crossmodal_retention=False, video_specialist=True)
