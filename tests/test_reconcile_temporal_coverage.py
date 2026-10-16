"""Coverage extends settled exposure history without weakening canonical equality."""

import copy

import pytest
from test_reconcile_crossmodal_retention import module
from test_reconcile_learnability import video_rows


def test_coverage_paths_preserve_all_history(tmp_path):
    code = module()
    paths = code.pilot_paths(tmp_path, temporal_coverage=True)
    assert paths[:-1] == code.pilot_paths(tmp_path, learnability=True)
    assert paths[-1][1] == tmp_path / "reports/s4-temporal-coverage-training-001.json"
    assert paths[-1][3] == tmp_path / "artifacts/s4-temporal-coverage-pilot-001"
    assert len(paths) == 9 and len(code.pilot_paths(tmp_path)) == 2


def families_fixture():
    prior = {r["id"]: r for r in video_rows(120)}
    current = copy.deepcopy(prior)
    for i, (k, r) in enumerate(prior.items()):
        if i >= 60:
            r["meta"]["modality"] = ["text"]
            current[k]["meta"]["modality"] = ["text"]
        else:
            current[k]["meta"]["family_id"] = f"new:{i}"
    selection = {
        "selected_families": sorted(
            r["meta"]["family_id"] for r in current.values() if r["meta"]["modality"] == ["video"]
        ),
        "excluded_prior_families": sorted(
            r["meta"]["family_id"] for r in prior.values() if r["meta"]["modality"] == ["video"]
        ),
        "excluded_prior_family_count": 60,
    }
    return current, prior, selection


@pytest.mark.parametrize("mutation", [None, "overlap", "selection", "replay"])
def test_selection_and_exact_replay_binding(mutation):
    code = module()
    current, prior, selection = families_fixture()
    if mutation == "overlap":
        current[next(iter(current))]["meta"]["family_id"] = prior[next(iter(prior))]["meta"][
            "family_id"
        ]
    elif mutation == "selection":
        selection["selected_families"][0] = "wrong"
    elif mutation == "replay":
        current[list(current)[-1]]["questions"][0]["target"]["source"] = "changed"
    if mutation:
        with pytest.raises(ValueError):
            code.coverage_families(current, prior, selection)
    else:
        code.coverage_families(current, prior, selection)


def test_segment002_paths_retain_first_coverage(tmp_path):
    code = module()
    paths = code.pilot_paths(tmp_path, coverage_segment002=True)
    assert paths[:-1] == code.pilot_paths(tmp_path, temporal_coverage=True)
    assert paths[-1][1] == tmp_path / "reports/s4-temporal-coverage-training-002.json"
    assert paths[-1][3] == tmp_path / "artifacts/s4-temporal-coverage-pilot-002"


@pytest.mark.parametrize("mutation", [None, "placeholder", "running", "prep_ready", "objective"])
def test_segment002_parent_comes_from_terminal_launcher(mutation):
    code = module()
    chosen = {
        "path": "/adapter/selected",
        "weights_sha256": "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3",
        "config_sha256": "a" * 64,
    }
    prep = {
        "status": "data_ready_only",
        "exit_code": 0,
        "pending_parent_selection": True,
        "training_authorized": False,
        "inventory_isolation": "isolated",
        "source_admission": "verified_existing_scope",
        "init_adapter": {"path": "/adapter/placeholder"},
    }
    launch = {
        "status": "completed",
        "exit_code": 0,
        "parent_selection": "root_selected_after_first_segment_screening",
        "selected_init_adapter": chosen,
    }
    final = {
        "init_adapter": chosen,
        "config": {
            "sample_count": 120,
            "steps": 10,
            "accumulation": 12,
            "seed": 42,
            "init_adapter": chosen,
        },
    }
    if mutation == "placeholder":
        final["init_adapter"] = prep["init_adapter"]
    elif mutation == "running":
        launch["status"] = "running"
    elif mutation == "prep_ready":
        prep["status"] = "ready_for_preflight"
    elif mutation == "objective":
        final["config"]["brier_weight"] = 0.5
    if mutation:
        with pytest.raises(ValueError):
            code.segment002_parent(prep, launch, final)
    else:
        code.segment002_parent(prep, launch, final)
