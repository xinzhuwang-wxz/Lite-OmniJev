"""Apply the reviewed 13-item local-evaluation scope to a new manifest only."""

import hashlib
import json
import os
from pathlib import Path

from scripts.decision_index_offline import CATALOG_IDS, PreparationManifest

root = Path("/root/autodl-tmp/s1-text-lora-001")
original = root / "artifacts/s4-index-inputs-001/manifest.json"
proposal = Path("/root/autodl-tmp/decision-index-admission-proposal-001.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


assert sha(original) == "44e6782bae205b655ddbe8f8d4ca3afb5095932e78b59f4af3e189a2993ed77b"
assert sha(proposal) == "11e9bc766804206a8d9d5ed2310e14adffd9da5082f181db99b86a5f0efadded"
manifest = PreparationManifest.model_validate_json(original.read_bytes())
assert not manifest.admissions and len(manifest.files) == 1530
admissions = json.loads(proposal.read_bytes())
assert {a["catalog_id"] for a in admissions} == {10, 24, 26, 27, 29, 30, 31, 36, 43, 48, 50, 57, 62}
assert len(admissions) == 13
for item in admissions:
    evidence = json.loads(item["evidence"])
    assert evidence["proposal"] == "root_review_required"
    assert evidence["input_manifest_sha256"] == sha(original)
    files = sorted(
        [
            {"source": str(f.path), "sha256": f.sha256, "destination": f.destination}
            for f in manifest.files
            if item["catalog_id"] in f.catalog_ids
        ],
        key=lambda f: f["destination"],
    )
    if item["catalog_id"] in [26, 27]:
        marker = "ARC-Easy/" if item["catalog_id"] == 26 else "ARC-Challenge/"
        files = [f for f in files if marker in f["destination"]]
        assert len(files) == 1
    assert digest(files) == evidence["applicable_files_sha256"]
    evidence["proposal"] = "root_reviewed_evaluation_admitted"
    evidence["scope"] = (
        "Offline local evaluation/rebuild only; no training or dataset redistribution; "
        "retain attribution, licenses and transformation notices. No new terms accepted."
    )
    evidence["reviewed_proposal_sha256"] = sha(proposal)
    item["evidence"] = json.dumps(evidence, sort_keys=True)
new = manifest.model_dump(mode="json")
new["admissions"] = admissions
PreparationManifest.model_validate(new)
os.umask(0o077)
out = root / "artifacts/s4-index-evaluation-admission-001"
out.mkdir(exist_ok=False)
(out / "manifest.json").write_text(json.dumps(new, indent=2) + "\n")
assert {k: v for k, v in new.items() if k != "admissions"} == {
    k: v for k, v in manifest.model_dump(mode="json").items() if k != "admissions"
}
report = {
    "status": "partial_evaluation_admission_applied",
    "exit_code": 0,
    "ready": False,
    "source_manifest_sha256": sha(original),
    "proposal_sha256": sha(proposal),
    "output_manifest_sha256": sha(out / "manifest.json"),
    "admitted_catalog_ids": sorted(a["catalog_id"] for a in admissions),
    "remaining_catalog_ids": sorted(CATALOG_IDS - {a["catalog_id"] for a in admissions}),
    "training_admitted": False,
    "redistribution_admitted": False,
    "original_manifest_unchanged": True,
    "gpu_used": False,
    "runner_sha256": sha(Path(__file__)),
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
