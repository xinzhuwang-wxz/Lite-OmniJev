"""Count reserved metadata without reading calibration or holdout payloads."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
sources = {
    "text": "s1-text-pools-v1-001",
    "image": "s2-media-pools-image-001",
    "screenshot": "s2-media-pools-screenshot-001",
    "audio": "s3-pools-audio-001",
    "video": "s3-pools-video-001",
}
results = {}
for cohort, directory in sources.items():
    path = root / "artifacts" / directory / "run-a/allocation-ledger.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(
        not ({"questions", "prefix", "target", "label", "answer"} & set(row)) for row in rows
    )
    counts = Counter(row["pool"] for row in rows)
    families = defaultdict(set)
    for row in rows:
        families[row["pool"]].add(row.get("family_hash", row.get("family_id")))
    results[cohort] = {
        "ledger_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "items_by_pool": dict(counts),
        "families_by_pool": {pool: len(values) for pool, values in families.items()},
        "holdout_target": 1000,
        "calibration_minimum_target": 3000,
    }
result = {
    "status": "inventory-only",
    "payloads_read": False,
    "scope": (
        "Existing reserved metadata only, not admission or release readiness; "
        "no new source selected"
    ),
    "cohorts": results,
    "crossmodal": {"status": "no_frozen_reserve_ledger_identified"},
}
(root / "reports/s3-reserve-inventory-001.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
