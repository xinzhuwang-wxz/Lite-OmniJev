"""Private CPU count of pinned CLEVRER gold questions, exporting aggregates only."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
ledger = root / "artifacts/s3-pools-video-001/run-a/allocation-ledger.jsonl"
memberships = defaultdict(set)
for line in ledger.read_text().splitlines():
    row = json.loads(line)
    memberships[row["family_id"]].add(row["pool"])
dev = Path("/root/autodl-tmp/s0-media-development-002/artifacts/media-dev-002/video-a/dev.jsonl")
assert hashlib.sha256(dev.read_bytes()).hexdigest() == (
    "79ed30be3a687a6c767c4d7c55c24dc7ac08d5a1698a6f49e3bc7d47e74e63d2"
)
for line in dev.read_text().splitlines():
    memberships[json.loads(line)["meta"]["family_id"]].add("dev")
assert all(len(pools) == 1 for pools in memberships.values())
manifest = json.loads((root / "reports/clevrer-source-download-snapshot.json").read_text())
result = []
for entry in manifest["files"]:
    if entry["relative_path"] not in {"questions/train.json", "questions/validation.json"}:
        continue
    path = Path(entry["remote_path"])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == entry["local_sha256"]
    counts, families = Counter(), defaultdict(set)
    seen_questions = set()
    for scene in json.loads(path.read_text()):
        family = f"clevrer:{scene['scene_index']}"
        pool = next(iter(memberships[family])) if family in memberships else "unallocated"
        for question in scene["questions"]:
            if (
                question.get("question_type") != "descriptive"
                or question.get("question_subtype") != "exist"
                or str(question.get("answer", "")).casefold() not in {"yes", "no"}
            ):
                continue
            key = (family, question["question"])
            if key in seen_questions:
                continue
            seen_questions.add(key)
            counts[pool] += 1
            families[pool].add(family)
    result.append(
        {
            "source": entry["relative_path"],
            "sha256": sha,
            "unique_original_exist_questions_by_pool": dict(counts),
            "families_by_pool": {key: len(value) for key, value in families.items()},
        }
    )
assert len(result) == 2
output = {
    "status": "source_capacity_upper_bound_only",
    "exit_code": 0,
    "scope": (
        "Pinned public gold parsed inside CPU job, only aggregates exported. No reserved payload "
        "file read or pool changed. Full historical/alias audit and media availability "
        "still pending."
    ),
    "allocation_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
    "sources": result,
}
out = root / "reports/s3-video-reserve-capacity-001.json"
assert not out.exists()
out.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output))
