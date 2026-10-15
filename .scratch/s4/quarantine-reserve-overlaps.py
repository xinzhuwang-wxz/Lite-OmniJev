"""Privately quarantine metadata-contaminated reserves; never remove training/dev history."""

import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
work = root / "artifacts/s4-gold-inventory-001"
source = work / "with-development-manifest.json"
spec = importlib.util.spec_from_file_location(
    "private_inventory", "/root/autodl-tmp/gold_inventory.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
captured = {}
original = module.audit_pool_isolation


def observe(value):
    captured.update(value)
    return original(value)


module.audit_pool_isolation = observe
receipt = module.audit_manifest(source)
assert receipt.categories == ["pool_overlap"]
rows = captured["rows"]
parents = {}


def find(x):
    parents.setdefault(x, x)
    while parents[x] != x:
        parents[x] = parents[parents[x]]
        x = parents[x]
    return x


def join(a, b):
    a, b = find(a), find(b)
    parents[b] = a


for row in rows:
    item = "i:" + row["item_sha256"]
    find(item)
    for key in row["original_family_sha256"] + row["family_alias_sha256"]:
        join(item, "f:" + key)
    for key in row["media_sha256"]:
        join(item, "m:" + key)
    for key in row["component_item_sha256"]:
        join(item, "i:" + key)
groups = defaultdict(list)
for row in rows:
    groups[find("i:" + row["item_sha256"])].append(row)
bad = [group for group in groups.values() if len({r["pool"] for r in group}) > 1]
assert len(bad) == receipt.isolation.overlapping_components
assert all("dev" in {r["pool"] for r in g} and "train" not in {r["pool"] for r in g} for g in bad)
family_exclusions = {x for g in bad for r in g for x in r["original_family_sha256"]}
media_exclusions = {x for g in bad for r in g for x in r["media_sha256"]}
private = work / "quarantined-reserves"
private.mkdir(mode=0o700, exist_ok=False)
exclusions = {"families": sorted(family_exclusions), "media": sorted(media_exclusions)}
p = private / "exclusions.json"
p.write_text(json.dumps(exclusions, sort_keys=True) + "\n")
os.chmod(p, 0o600)
manifest = json.loads(source.read_text())
changes = []
for index, artifact in enumerate(manifest["samples"]):
    if artifact["exposure"] != "current" or artifact["expected_pool"] not in {
        "calibration",
        "holdout",
    }:
        continue
    path = Path(artifact["path"])
    payload = path.read_bytes()
    assert module.digest(payload) == artifact["sha256"]
    kept = []
    removed = []
    for line in payload.decode().splitlines():
        sample = json.loads(line)
        contaminated = module.canonical_family_hash(
            sample["meta"]["family_id"]
        ) in family_exclusions or any(
            m["sha256"] in media_exclusions for m in sample["prefix"]["media"]
        )
        (removed if contaminated else kept).append(line)
    if not removed:
        continue
    assert kept
    output = private / f"reserved-{index}.jsonl"
    output.write_text("\n".join(kept) + "\n")
    os.chmod(output, 0o600)
    quarantine = private / f"excluded-{index}.jsonl"
    quarantine.write_text("\n".join(removed) + "\n")
    os.chmod(quarantine, 0o600)
    artifact.update(path=str(output), sha256=module.digest(output.read_bytes()))
    changes.append(
        {
            "cohort": artifact["cohort"],
            "pool": artifact["expected_pool"],
            "removed": len(removed),
            "remaining": len(kept),
        }
    )
output = work / "with-development-clean-reserves-manifest.json"
assert not output.exists()
output.write_text(json.dumps(manifest, indent=2) + "\n")
module.audit_pool_isolation = original
verified = module.audit_manifest(output)
assert verified.outcome == "isolated"
module.publish(verified, work / "development-inventory-005.json")
summary = {
    "status": "quarantined_and_reaudited",
    "exit_code": 0,
    "scope": "Supplied known history only; no real gold admission, quotas or permission acceptance",
    "overlapping_components": len(bad),
    "cohort_pool_changes": changes,
    "training_or_dev_removed": False,
    "source_reserves_modified": False,
    "quarantine_exclusions_sha256": module.digest(p.read_bytes()),
    "manifest_sha256": module.digest(output.read_bytes()),
    "remaining_overlap_components": verified.isolation.overlapping_components,
    "next": "Replenish reliable gold; source provenance and permissions still pending",
}
(work / "reserve-quarantine-summary-001.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
