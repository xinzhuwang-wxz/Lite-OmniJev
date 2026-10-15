"""Summarize development question coverage; no reserved dataset access."""

import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
data = Path("/root/autodl-tmp/s0-media-development-002/artifacts/media-dev-002/image-a/dev.jsonl")
records = root / "artifacts/s2-text-lora-001/dev-eval-image-v2-corrected-001/records.jsonl"
pred = {
    r["id"]: r
    for r in map(json.loads, records.read_text().splitlines())
    if r["variant"] == "original" and r["pass"] == 0
}
groups = {}
coverage = Counter()
for row in map(json.loads, data.read_text().splitlines()):
    q = row["questions"][0]["instruction"].lower()
    words = q.rstrip("?").split()
    prefix = " ".join(words[:2])
    category = (
        "color"
        if "color" in words or "colour" in words
        else "count"
        if q.startswith("how many")
        else "why"
        if q.startswith("why ")
        else "other"
    )
    stat = groups.setdefault(category, {"count": 0, "correct": 0, "prefixes": Counter()})
    r = pred[row["id"]]
    stat["count"] += 1
    stat["correct"] += int(r["prediction"] == r["target"])
    stat["prefixes"][prefix] += 1
for g in groups.values():
    g["accuracy"] = g["correct"] / g["count"]
    g["prefixes"] = dict(g["prefixes"].most_common(8))
report = {
    "scope": (
        "development-only aggregate coverage audit, coarse lexical categories not semantic gold"
    ),
    "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
    "groups": groups,
}
(root / "reports/s2-question-family-audit-001.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
