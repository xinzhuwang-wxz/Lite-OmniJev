"""TRAIN-only12-item repeated exposure contracts; synthetic CPU data."""

import json

import pytest
from test_gold_inventory import sample
from test_reconcile_crossmodal_retention import module

from data.schema import Sample


def video_rows(count=12):
    rows = []
    for n in range(count):
        r = sample(f"video-{n}", pool="train", modality="video")
        r["meta"]["family_id"] = f"family:{n}"
        r["questions"][0]["target"]["value"] = "yes" if n % 2 else "no"
        rows.append(Sample.model_validate(r).model_dump(mode="json"))
    return rows


def test_canonical12_exact_subset_and_balance(tmp_path):
    code = module()
    rows = video_rows(120)
    source = tmp_path / "source.jsonl"
    subset = tmp_path / "subset.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows))
    subset.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows[:12]))
    assert code.learnability_subset(subset, source) == {
        "unique_samples": 12,
        "unique_families": 12,
        "yes": 6,
        "no": 6,
        "scheduled_draws": 120,
    }
    rows[0]["questions"][0]["target"]["source"] = "changed"
    subset.write_text("".join(json.dumps(r) + "\n" for r in rows[:12]))
    with pytest.raises(ValueError, match="learnability_full_sample_subset"):
        code.learnability_subset(subset, source)


@pytest.mark.parametrize("change", ["live", "count", "accumulation", "objective", "steps"])
def test_reject_wrong_repeated_draw_or_nonterminal_contract(change):
    code = module()
    report = {
        "status": "completed",
        "exit_code": 0,
        "steps_completed": 10,
        "config": {"sample_count": 12, "seed": 42, "steps": 10, "accumulation": 12},
    }
    code.learnability_terminal(report)
    if change == "live":
        report["status"] = "running"
    elif change == "count":
        report["config"]["sample_count"] = 120
    elif change == "accumulation":
        report["config"]["accumulation"] = 1
    elif change == "objective":
        report["config"]["brier_weight"] = 0.5
    else:
        report["steps_completed"] = 9
    with pytest.raises(ValueError):
        code.learnability_terminal(report)


def test_new_paths_include_every_older_candidate(tmp_path):
    code = module()
    paths = code.pilot_paths(tmp_path, learnability=True)
    assert paths[:-1] == code.pilot_paths(tmp_path, temporal_brier=True)
    assert len(paths) == 8 and len(code.pilot_paths(tmp_path)) == 2
    assert paths[-1][1] == tmp_path / "reports/s4-train-learnability-training-001.json"
    assert paths[-1][3] == tmp_path / "artifacts/s4-train-learnability-001"
