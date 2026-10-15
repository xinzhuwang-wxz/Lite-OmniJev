"""Synthetic CPU reconciliation against the004 base; no actual sealed data."""

import json
from pathlib import Path

import pytest
from test_reconcile_claim_pair import fixture as claim_fixture


def fixture(root: Path):
    code, index = claim_fixture(root)
    base = root / "artifacts/s4-pilot-exposure-004"
    base.mkdir()
    old = root / "artifacts/s4-claim-pair-pilot-001"
    (base / "inventory-manifest.json").write_bytes((old / "inventory-manifest.json").read_bytes())
    (base / "report.json").write_text(
        json.dumps(
            {
                "status": "known_pilot_exposures_reconciled",
                "exit_code": 0,
                "unknown_update_bearing_inputs": 0,
                "output_sha256": {
                    "inventory-manifest.json": code.sha(base / "inventory-manifest.json")
                },
            }
        )
    )
    _, final, run, prep = code.pilot_paths(root, temporal_curriculum=True)[-1]
    run.mkdir()
    prep.mkdir()
    data = [json.loads(line) for line in (old / "train.jsonl").read_text().splitlines()]
    for row in data:
        row["id"] = "temporal-" + row["id"]
    payload = "".join(json.dumps(r) + "\n" for r in data)
    (run / "data.jsonl").write_text(payload)
    (prep / "train.jsonl").write_text(payload)
    init = {
        "weights_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "path": str(root / "fake-adapter"),
    }
    report = {
        "status": "completed",
        "exit_code": 0,
        "steps_completed": 10,
        "data_sha256": code.sha(run / "data.jsonl"),
        "init_adapter": init,
        "config": {"seed": 42, "sample_count": 120, "steps": 10, "accumulation": 12},
    }
    final.write_text(json.dumps(report))
    (run / "run.json").write_text(json.dumps(report))
    names = [
        "allocation-ledger.jsonl",
        "config-ce.json",
        "draw-recipe.json",
        "inventory-manifest.json",
        "inventory-audit.json",
    ]
    for name in names:
        (prep / name).write_text("{}")
    (prep / "report.json").write_text(
        json.dumps(
            {
                "status": "ready_for_preflight",
                "exit_code": 0,
                "data_ready_for_preflight": True,
                "inventory_isolation": "isolated",
                "source_admission": "verified_existing_scope",
                "total_draws": 120,
                "init_adapter": init,
                "output_sha256": {n: code.sha(prep / n) for n in [*names, "train.jsonl"]},
            }
        )
    )
    return code, index, final, run, prep, base


def test004_base_and_all_prior_pilots_preserved(tmp_path):
    code, index, _, _, _, base = fixture(tmp_path)
    assert len(code.pilot_paths(tmp_path)) == 2
    paths = code.pilot_paths(tmp_path, temporal_curriculum=True)
    assert len(paths) == 6 and paths[:5] == code.pilot_paths(tmp_path, claim_pair=True)
    result = code.prepare(
        tmp_path, tmp_path / "legacy", tmp_path / "out", index, temporal_curriculum=True
    )
    assert result["base_inventory"]["path"] == str(base / "inventory-manifest.json")
    assert result["temporal_curriculum_included"] and result["claim_pair_included"]
    assert len(result["appended_artifacts"]) == 1
    assert result["discovered_update_bearing_reports"] == 12
    assert result["unknown_update_bearing_inputs"] == 0
    assert result["pilots"][-1]["canonical_equivalence"]["rows"] == 120


@pytest.mark.parametrize("change", ["live", "data", "ledger", "init", "base", "admission"])
def test_reject_any_mutated_identity_or_live_training(tmp_path, change):
    code, index, final, run, prep, base = fixture(tmp_path)
    if change in ["live", "init"]:
        value = json.loads(final.read_text())
        if change == "live":
            value["status"] = "running"
        else:
            value["init_adapter"]["weights_sha256"] = "c" * 64
        final.write_text(json.dumps(value))
        (run / "run.json").write_text(json.dumps(value))
    elif change == "data":
        rows = [json.loads(line) for line in (run / "data.jsonl").read_text().splitlines()]
        rows[0]["questions"][0]["target"]["source"] = "changed"
        (run / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        value = json.loads(final.read_text())
        value["data_sha256"] = code.sha(run / "data.jsonl")
        final.write_text(json.dumps(value))
        (run / "run.json").write_text(json.dumps(value))
    elif change == "ledger":
        (prep / "allocation-ledger.jsonl").write_text("changed")
    elif change == "base":
        (base / "inventory-manifest.json").write_text("{}")
    else:
        value = json.loads((prep / "report.json").read_text())
        value["source_admission"] = "pending"
        (prep / "report.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        code.prepare(
            tmp_path, tmp_path / "legacy", tmp_path / "out", index, temporal_curriculum=True
        )
    assert json.loads((tmp_path / "out/report.json").read_text())["status"] == "rejected"
