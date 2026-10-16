"""Append three reviewed evaluation scopes while preserving the approved base."""

import hashlib
import json
import os
import tarfile
from pathlib import Path

from scripts.decision_index_offline import CATALOG_IDS, PreparationManifest

root = Path("/root/autodl-tmp/s1-text-lora-001")
base = root / "artifacts/s4-index-evaluation-admission-001/manifest.json"
proposal = Path("/root/autodl-tmp/decision-index-additional-admission-001.json")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert sha(base) == "b40157e74b88f411161d3905a75ff4077f9f5b6cf2ec1d1ba0ad887b04842ac1"
assert sha(proposal) == "4f94ec17881242204347b1ae79275369e8cb0a922c42bb2e920d8e11d3c94c83"
manifest = PreparationManifest.model_validate_json(base.read_bytes())
draft = json.loads(proposal.read_bytes())
assert draft["status"] == "proposal_pending_root_review"
assert draft["base_manifest_sha256"] == sha(base)
assert len(manifest.admissions) == 13
additions = draft["admissions"]
assert len(additions) == 3 and {a["catalog_id"] for a in additions} == {4, 5, 9}
assert sha(manifest.source_archive.path) == manifest.source_archive.sha256
files = {f.destination: f for f in manifest.files}
for item in additions:
    evidence = json.loads(item["evidence"])
    assert evidence["approved_base_sha256"] == sha(base)
    assert evidence["source_archive_sha256"] == manifest.source_archive.sha256
    for entry in evidence["inputs"]:
        bound = files[entry["destination"]]
        assert str(bound.path) == entry["path"] and bound.sha256 == entry["sha256"]
        assert item["catalog_id"] in bound.catalog_ids and sha(bound.path) == bound.sha256
        data = bound.path.read_bytes()
        assert len(data) == entry["bytes"]
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == entry["git_blob_sha1"] == entry["publisher"]["git_blob_sha1"]
    with tarfile.open(manifest.source_archive.path) as archive:
        for name, digest in evidence["source_files_sha256"].items():
            members = [
                m for m in archive.getmembers() if m.name == name or m.name.endswith("/" + name)
            ]
            assert len(members) == 1 and members[0].isfile() and members[0].size < 5_000_000
            stream = archive.extractfile(members[0])
            assert stream and hashlib.sha256(stream.read()).hexdigest() == digest
    evidence["review"] = "root_reviewed_evaluation_admitted"
    evidence["reviewed_proposal_sha256"] = sha(proposal)
    evidence["purpose"] = (
        "Offline local evaluation/rebuild only; no training or redistribution; "
        "retain attribution, licenses and transformation notices. No new terms accepted."
    )
    item["evidence"] = json.dumps(evidence, sort_keys=True)
new = manifest.model_dump(mode="json")
new["admissions"].extend(additions)
validated = PreparationManifest.model_validate(new)
assert len({a.catalog_id for a in validated.admissions}) == 16
os.umask(0o077)
out = root / "artifacts/s4-index-evaluation-admission-002"
out.mkdir(exist_ok=False)
(out / "manifest.json").write_text(json.dumps(new, indent=2) + "\n")
report = {
    "status": "partial_evaluation_admission_applied",
    "exit_code": 0,
    "ready": False,
    "base_manifest_sha256": sha(base),
    "proposal_sha256": sha(proposal),
    "output_manifest_sha256": sha(out / "manifest.json"),
    "admitted_catalog_ids": sorted(a.catalog_id for a in validated.admissions),
    "remaining_catalog_ids": sorted(CATALOG_IDS - {a.catalog_id for a in validated.admissions}),
    "training_admitted": False,
    "redistribution_admitted": False,
    "original_manifests_unchanged": True,
    "gpu_used": False,
    "runner_sha256": sha(Path(__file__)),
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
