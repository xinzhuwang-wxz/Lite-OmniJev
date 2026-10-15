"""Private CPU source audit: export only aggregate audio reserve capacity."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.s0_audio_dev import load_license_plan, load_metadata, load_train_groups

root = Path("/root/autodl-tmp/s1-text-lora-001")
source = Path("/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207")
meta_sha = "7505ba93c291d344b31e5c33e872b95d84255852cff93278bbdf549402acc1f5"
approved = load_license_plan(
    root / "reports/clotho-aqa-license-plan.json", metadata_sha256=meta_sha
)
metadata = load_metadata(
    source / "clotho_aqa_metadata.csv", metadata_sha256=meta_sha, approved=approved
)
ledger = root / "artifacts/s3-pools-audio-001/run-a/allocation-ledger.jsonl"
memberships = defaultdict(set)
for line in ledger.read_text().splitlines():
    row = json.loads(line)
    memberships[row["family_id"]].add(row["pool"])
dev_ledger = Path(
    "/root/autodl-tmp/s0-audio-dev-001/artifacts/audio-dev-001-a/exclusion-ledger.jsonl"
)
for line in dev_ledger.read_text().splitlines():
    row = json.loads(line)
    if row["pool"] == "dev_excluded":
        memberships[row["family_id"]].add("dev")
assert all(len(pools) == 1 for pools in memberships.values())
outputs = {}
all_questions = set()
unique_counts = Counter()
families = defaultdict(set)
for split in ("train", "val", "test"):
    path = source / f"clotho_aqa_{split}.csv"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = load_train_groups(path, train_sha256=sha)
    counts, rejected = Counter(), Counter()
    for (name, question), rows in groups.items():
        if name not in metadata:
            rejected["media_not_approved"] += 1
            continue
        answers = {row["answer"].strip().casefold() for row in rows}
        if len(rows) != 3 or len(answers) != 1 or not answers.issubset({"yes", "no"}):
            rejected["not_three_unanimous_yes_no"] += 1
            continue
        family = metadata[name]["sound_id"]
        pool = next(iter(memberships[family])) if family in memberships else "unallocated"
        counts[pool] += 1
        key = (family, question)
        if key not in all_questions:
            unique_counts[pool] += 1
            families[pool].add(family)
            all_questions.add(key)
    outputs[split] = {"sha256": sha, "candidate_counts": dict(counts), "rejected": dict(rejected)}
result = {
    "status": "source_capacity_upper_bound_only",
    "exit_code": 0,
    "scope": (
        "Private job parsed public source labels; only aggregates exported. No reserved payload "
        "file read, no pool changed. Duration, media hashes, complete historical exposure and "
        "crossmodal aliases still require admission checks. No model evaluation."
    ),
    "metadata_sha256": meta_sha,
    "allocation_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    "dev_exclusion_ledger_sha256": hashlib.sha256(dev_ledger.read_bytes()).hexdigest(),
    "source_splits": outputs,
    "unique_questions_by_existing_pool": dict(unique_counts),
    "families_by_existing_pool": {pool: len(values) for pool, values in families.items()},
}
out = root / "reports/s3-audio-reserve-capacity-001.json"
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
