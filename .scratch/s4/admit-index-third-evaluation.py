"""Apply reviewed fixed BFCL/BBH evaluation scopes; preserve prior manifests."""

import hashlib
import json
import os
from pathlib import Path

from scripts.decision_index_offline import CATALOG_IDS, PreparationManifest

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
BASE = ROOT / "artifacts/s4-index-evaluation-admission-002/manifest.json"
PROPOSAL = Path("/root/autodl-tmp/decision-index-third-admission-reviewed-001.json")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert sha(BASE) == "e5834b0bb49a4ba19c7cfda5573129c710ec8aed131e43cb781aed6f48d8cf5f"
assert sha(PROPOSAL) == "e138f2048fff5815c3585f671da91ff6be0ed0ba7a37afe0c43a57cc0a46dd4a"
manifest = PreparationManifest.model_validate_json(BASE.read_bytes())
proposal = json.loads(PROPOSAL.read_bytes())
assert proposal["base_manifest_sha256"] == sha(BASE)
assert proposal["status"] == "proposal_pending_root_review" and proposal["applied"] is False
assert len(manifest.admissions) == 16
assert {a["catalog_id"] for a in proposal["admissions"]} == {1, 58}
files = {f.destination: f for f in manifest.files}
checked_inputs = 0
for entry in proposal["input_bindings"]:
    if not {1, 58}.intersection(entry["catalog_ids"]):
        continue
    actual = files[entry["destination"]]
    assert str(actual.path) == entry["path"] and actual.sha256 == entry["sha256"]
    assert sha(actual.path) == actual.sha256 and actual.path.stat().st_size == entry["bytes"]
    assert set(entry["catalog_ids"]) <= set(actual.catalog_ids)
    checked_inputs += 1
assert checked_inputs == 68
# Only publisher-authored license/scope metadata is retrieved; no benchmark rows executed/read.
metadata_checks = []
for item in proposal["publisher_metadata"]:
    wanted = item.get("key") == "fixed_scope_followup" and item["path"] in {
        "berkeley-function-call-leaderboard/data/README.md",
        "bbh/README.md",
    }
    wanted |= item.get("key") in {"bfcl", "bbh"} and item.get("path") == "LICENSE"
    if not wanted:
        continue
    assert item["url"].startswith("https://raw.githubusercontent.com/")
    assert "/" + item["revision"] + "/" in item["url"]
    source_name = "bfcl" if item["repo"] == "gorilla-llm/gorilla" else "cand-BIG-Bench-Hard"
    local_metadata = (
        Path(
            "/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/repos"
        )
        / source_name
        / item["path"]
    )
    payload = local_metadata.read_bytes()
    assert len(payload) == item["bytes"] < 100_000
    assert hashlib.sha256(payload).hexdigest() == item["sha256"]
    metadata_checks.append({k: item[k] for k in ["url", "sha256", "bytes"]})
assert len(metadata_checks) == 4
additions = []
for original in proposal["admissions"]:
    addition = dict(original)
    evidence = json.loads(addition["evidence"])
    evidence["root_review_required"] = False
    evidence["review"] = "root_reviewed_fixed_publisher_scope_evaluation_only"
    evidence["reviewed_proposal_sha256"] = sha(PROPOSAL)
    evidence["purpose"] = (
        "Offline local evaluation only; never train or redistribute; preserve original licenses, "
        "task notices and canary. No external API execution or terms acceptance."
    )
    addition["evidence"] = json.dumps(evidence, sort_keys=True)
    additions.append(addition)
new = manifest.model_dump(mode="json")
new["admissions"].extend(additions)
validated = PreparationManifest.model_validate(new)
assert len({a.catalog_id for a in validated.admissions}) == 18
os.umask(0o077)
out = ROOT / "artifacts/s4-index-evaluation-admission-003"
out.mkdir(exist_ok=False)
(out / "manifest.json").write_text(json.dumps(new, indent=2) + "\n")
report = {
    "status": "partial_evaluation_admission_applied",
    "exit_code": 0,
    "ready": False,
    "base_manifest_sha256": sha(BASE),
    "proposal_sha256": sha(PROPOSAL),
    "output_manifest_sha256": sha(out / "manifest.json"),
    "admitted_catalog_ids": sorted(a.catalog_id for a in validated.admissions),
    "remaining_catalog_ids": sorted(CATALOG_IDS - {a.catalog_id for a in validated.admissions}),
    "verified_input_files": checked_inputs,
    "publisher_metadata_checks": metadata_checks,
    "training_admitted": False,
    "redistribution_admitted": False,
    "original_manifests_unchanged": True,
    "gpu_used": False,
    "runner_sha256": sha(Path(__file__)),
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
