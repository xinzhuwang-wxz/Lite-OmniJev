"""Bounded CPU interface/binding checks; no live training or private inventory reads."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".scratch/s4/reconcile-pilot-exposure.py"


def module():
    spec = importlib.util.spec_from_file_location("reconcile_retention", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_optional_paths_preserve_old_candidates_and_add_explicit_new_paths(tmp_path):
    code = module()
    old = code.pilot_paths(tmp_path)
    new = code.pilot_paths(tmp_path, crossmodal_retention=True)
    assert len(old) == 2 and new[:2] == old and len(new) == 3
    assert new[-1] == (
        "s4-crossmodal-retention",
        tmp_path / "reports/s4-crossmodal-retention-training-001.json",
        tmp_path / "artifacts/s4-crossmodal-retention-train-001",
        tmp_path / "artifacts/s4-crossmodal-retention-pilot-002",
    )
    assert code.base_inventory(tmp_path, crossmodal_retention=False) == (
        tmp_path / "artifacts/s4-video-pilot-001/inventory-manifest.json",
        None,
    )


def test_new_base_requires_successful_preparation_and_exact_manifest_hash(tmp_path):
    code = module()
    prep = tmp_path / "artifacts/s4-crossmodal-retention-pilot-002"
    prep.mkdir(parents=True)
    inventory = prep / "inventory-manifest.json"
    inventory.write_text("{}")
    report = prep / "report.json"
    payload = {
        "status": "prepared",
        "exit_code": 0,
        "output_sha256": {"inventory-manifest.json": code.sha(inventory)},
    }
    report.write_text(json.dumps(payload))
    path, binding = code.base_inventory(tmp_path, crossmodal_retention=True)
    assert path == inventory and binding == {"path": str(report), "sha256": code.sha(report)}
    inventory.write_text('{"modified":true}')
    with pytest.raises(ValueError, match="base_inventory_preparation_sha"):
        code.base_inventory(tmp_path, crossmodal_retention=True)
    payload["status"] = "blocked"
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="preparation_not_passed"):
        code.base_inventory(tmp_path, crossmodal_retention=True)


@pytest.mark.parametrize("status", ["running", "ready", "preparing"])
def test_no_live_report_becomes_terminal(status):
    code = module()
    with pytest.raises(ValueError, match="nonterminal_report"):
        code.terminal_report({"status": status, "steps_completed": 15, "exit_code": 0})


def test_canonical_equivalence_still_rejects_changed_gold(tmp_path):
    from data.synth import generate_samples

    code = module()
    sample = generate_samples(seed=42, count=1)[0].model_dump(mode="json")
    sample["meta"]["pool"] = "train"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(sample) + "\n")
    second.write_text(json.dumps(sample, sort_keys=True) + "\n")
    assert code.equivalent(first, second)["rows"] == 1
    sample["questions"][0]["target"]["source"] = "different provenance"
    second.write_text(json.dumps(sample) + "\n")
    with pytest.raises(ValueError, match="canonical_sample_mismatch"):
        code.equivalent(first, second)
