"""Synthetic CPU terminal exposure checks; no live reports or sealed inventory."""

import json
from pathlib import Path

import pytest
from test_reconcile_crossmodal_retention import module

from data.synth import generate_samples


def fixture(root: Path):
    code = module()
    rows = [s.model_dump(mode="json") for s in generate_samples(seed=42, count=120)]
    for row in rows:
        row["meta"]["pool"] = "train"
    raw = "".join(json.dumps(r) + "\n" for r in rows)
    for name, final_path, run_dir, prep_dir in code.pilot_paths(root, claim_pair=True):
        final_path.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        prep_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "data.jsonl").write_text(raw)
        (prep_dir / "train.jsonl").write_text(raw)
        claim = name.startswith("s4-claim-pair-")
        report = {
            "status": "completed",
            "steps_completed": 10 if claim else 30,
            "exit_code": 0,
            "data_sha256": code.sha(run_dir / "data.jsonl"),
        }
        if claim:
            report["config"] = {
                "seed": 42,
                "sample_count": 120,
                "steps": 10,
                "accumulation": 12,
                "pair_loss_weight": 0.1 if name.endswith("pair01") else 0.0,
            }
        (run_dir / "run.json").write_text(json.dumps(report))
        final_path.write_text(json.dumps(report))
        prep = {
            "status": "prepared",
            "exit_code": 0,
            "output_sha256": {"train.jsonl": code.sha(prep_dir / "train.jsonl")},
        }
        if claim:
            prep.update(
                status="prepared_pending_pair_hashes",
                expanded_isolation="isolated",
                pilot_isolation="isolated",
                draws=120,
                pair_count=20,
            )
        (prep_dir / "report.json").write_text(json.dumps(prep))
    prep_dir = root / "artifacts/s4-claim-pair-pilot-001"
    version = rows[0]["meta"]["data_version"]
    inventory = {
        "schema_version": "gold-inventory-manifest-v1",
        "samples": [
            {
                "path": str(prep_dir / "train.jsonl"),
                "sha256": code.sha(prep_dir / "train.jsonl"),
                "expected_pool": "train",
                "exposure": "current",
                "cohorts_by_data_version": {version: "text"},
            }
        ],
        "component_ledgers": [],
    }
    (prep_dir / "inventory-manifest.json").write_text(json.dumps(inventory))
    prep = json.loads((prep_dir / "report.json").read_text())
    prep["output_sha256"]["inventory-manifest.json"] = code.sha(
        prep_dir / "inventory-manifest.json"
    )
    (prep_dir / "report.json").write_text(json.dumps(prep))
    index = root / "index.json"
    index.write_text(json.dumps({"status": "indexed", "unresolved": [], "files": []}))
    return code, index


def test_claim_option_keeps_all_older_paths(tmp_path):
    code = module()
    old = code.pilot_paths(tmp_path, crossmodal_retention=True)
    new = code.pilot_paths(tmp_path, claim_pair=True)
    assert new[:3] == old and len(new) == 5
    assert new[-1][1] == tmp_path / "reports/s4-claim-pair-pair01-training-001.json"
    assert new[-1][2] == tmp_path / "artifacts/s4-claim-pair-pair01-train-001"
    assert new[-1][3] == tmp_path / "artifacts/s4-claim-pair-pilot-001"


@pytest.mark.parametrize("failed_steps", [None, 3])
def test_both_terminal_arms_record_actual_updates(tmp_path, failed_steps):
    code, index = fixture(tmp_path)
    if failed_steps:
        _, final, run, _ = code.pilot_paths(tmp_path, claim_pair=True)[-1]
        value = json.loads(final.read_text())
        value.update(status="failed", steps_completed=failed_steps, exit_code=1)
        final.write_text(json.dumps(value))
        (run / "run.json").write_text(json.dumps(value))
    result = code.prepare(tmp_path, tmp_path / "legacy", tmp_path / "out", index, claim_pair=True)
    assert result["claim_pair_included"] and result["crossmodal_retention_included"]
    assert len(result["pilots"]) == 5
    assert result["discovered_update_bearing_reports"] == 10
    assert all(p["update_bearing"] for p in result["pilots"])
    assert result["pilots"][-1]["steps_completed"] == (failed_steps or 10)
    assert result["unknown_update_bearing_inputs"] == 0


@pytest.mark.parametrize(
    "mutation", ["running", "short_completed", "wrong_gold", "unknown", "manifest_sha"]
)
def test_reject_live_partial_changed_input_unknown_or_unbound_base(tmp_path, mutation):
    code, index = fixture(tmp_path)
    _, final, run, prep = code.pilot_paths(tmp_path, claim_pair=True)[-1]
    value = json.loads(final.read_text())
    if mutation == "running":
        value["status"] = "running"
    elif mutation == "short_completed":
        value["steps_completed"] = 9
    elif mutation == "wrong_gold":
        data = [json.loads(line) for line in (run / "data.jsonl").read_text().splitlines()]
        data[0]["questions"][0]["target"]["source"] = "changed"
        (run / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
        value["data_sha256"] = code.sha(run / "data.jsonl")
    elif mutation == "unknown":
        (tmp_path / "reports/unknown.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "steps_completed": 1,
                    "exit_code": 0,
                    "data_sha256": "f" * 64,
                }
            )
        )
    else:
        (prep / "inventory-manifest.json").write_text("{}")
    final.write_text(json.dumps(value))
    (run / "run.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        code.prepare(tmp_path, tmp_path / "legacy", tmp_path / "out", index, claim_pair=True)
    assert json.loads((tmp_path / "out/report.json").read_text())["status"] == "rejected"
