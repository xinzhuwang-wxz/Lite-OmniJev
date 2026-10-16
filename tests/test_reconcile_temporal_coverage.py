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
