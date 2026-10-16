"""Apply three reviewed dataset scopes while preserving the eighteen admitted sources."""

import hashlib
import json
import os
from pathlib import Path

from scripts.decision_index_offline import CATALOG_IDS, PreparationManifest

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
BASE = ROOT / "artifacts/s4-index-evaluation-admission-003/manifest.json"
ORIGINAL = ROOT / "artifacts/s4-index-evaluation-admission-002/manifest.json"
PROPOSAL = Path("/root/autodl-tmp/decision-index-next-reviewed-001.json")
CACHE = Path("/root/autodl-tmp/s4-index-publisher-cache-owned")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert sha(BASE) == "8ac03215c8497115c73626c3aa202113f472e436f6b50c0b868c3d7fc2292ce3"
assert sha(ORIGINAL) == "e5834b0bb49a4ba19c7cfda5573129c710ec8aed131e43cb781aed6f48d8cf5f"
assert sha(PROPOSAL) == "1821e118b8e3bd6d4e85cefbadcd0d648af3e022e53f03d86675f89e81dec0c7"
manifest = PreparationManifest.model_validate_json(BASE.read_bytes())
prior = PreparationManifest.model_validate_json(ORIGINAL.read_bytes())
assert len(manifest.admissions) == 18 and len(prior.admissions) == 16
assert manifest.files == prior.files
assert manifest.admissions[:16] == prior.admissions
proposal = json.loads(PROPOSAL.read_bytes())
assert proposal["base_manifest_sha256"] == sha(ORIGINAL)
assert proposal["status"] == "proposal_pending_root_review" and proposal["applied"] is False
assert {x["catalog_id"] for x in proposal["admissions"]} == {3, 32, 37}
files = {str(f.path): f for f in manifest.files}
additions = []
counts = {}
metadata_checks = {}
for original in proposal["admissions"]:
    addition = dict(original)
    evidence = json.loads(addition["evidence"])
    assert evidence["approved_base_sha256"] == sha(ORIGINAL)
    for entry in evidence["input_bindings"]:
        bound = files[entry["path"]]
        assert bound.destination == entry["destination"]
        assert bound.sha256 == entry["sha256"] == sha(bound.path)
        assert list(bound.catalog_ids) == entry["catalog_ids"] == [addition["catalog_id"]]
        assert bound.path.stat().st_size == entry["bytes"]
    counts[addition["catalog_id"]] = len(evidence["input_bindings"])
    for item in evidence["publisher_metadata"]:
        if "bytes" not in item:
            continue
        p = CACHE / item["sha256"]
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"]
        metadata_checks[item["url"]] = item["sha256"]
    evidence["root_review_required"] = False
    evidence["review"] = "root_reviewed_publisher_work_and_explicit_dataset_scope"
    evidence["reviewed_proposal_sha256"] = sha(PROPOSAL)
    evidence["extended_base_manifest_sha256"] = sha(BASE)
    evidence["purpose"] = (
        "Offline local evaluation/rebuild only; no training or redistribution. "
        "Retain original licenses, attribution, NOTICE and change notices. "
        "No third-party API execution, model generation or terms acceptance."
    )
    addition["evidence"] = json.dumps(evidence, sort_keys=True)
    additions.append(addition)
assert counts == {3: 319, 32: 3, 37: 2} and len(metadata_checks) == 8
new = manifest.model_dump(mode="json")
new["admissions"].extend(additions)
validated = PreparationManifest.model_validate(new)
assert len({a.catalog_id for a in validated.admissions}) == 21
os.umask(0o077)
out = ROOT / "artifacts/s4-index-evaluation-admission-004"
out.mkdir(exist_ok=False)
(out / "manifest.json").write_text(json.dumps(new, indent=2) + "\n")
report = {
    "status": "partial_evaluation_admission_applied",
    "exit_code": 0,
    "ready": False,
    "base_manifest_sha256": sha(BASE),
    "proposal_base_sha256": sha(ORIGINAL),
    "proposal_sha256": sha(PROPOSAL),
    "output_manifest_sha256": sha(out / "manifest.json"),
    "admitted_catalog_ids": sorted(a.catalog_id for a in validated.admissions),
    "remaining_catalog_ids": sorted(CATALOG_IDS - {a.catalog_id for a in validated.admissions}),
    "verified_inputs_by_catalog": counts,
    "publisher_metadata_checks": metadata_checks,
    "training_admitted": False,
    "redistribution_admitted": False,
    "original_manifests_unchanged": True,
    "gpu_used": False,
    "scope_boundary": (
        "API-Bank published Work scope is an explicit inference; "
        "MuSR uses publisher data CC-BY-4.0; ESCI uses dataset-project Apache-2.0."
    ),
    "runner_sha256": sha(Path(__file__)),
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
