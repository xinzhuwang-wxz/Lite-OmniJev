"""Audit frozen video development predictions and train label balance only."""

import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
report_path = root / "reports/s3-v2-video-dev500-diagnostic-001.json"
r = json.loads(report_path.read_text())
cmd = r["command"]
data = Path(cmd[cmd.index("--data") + 1])
sidecar = Path(cmd[cmd.index("--output-dir") + 1]) / "records.jsonl"
assert hashlib.sha256(data.read_bytes()).hexdigest() == r["data_sha256"]
assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == r["sidecar_sha256"]
rows = {row["id"]: row for row in map(json.loads, data.read_text().splitlines())}
pred = [
    row
    for row in map(json.loads, sidecar.read_text().splitlines())
    if row["variant"] == "original" and row["pass"] == 0
]
assert len(pred) == 500
confusion = Counter()
groups = {}
for record in pred:
    confusion[record["target"] + "->" + record["prediction"]] += 1
    words = rows[record["id"]]["questions"][0]["instruction"].lower().split()
    kind = (
        "temporal"
        if any(word in words for word in ["before", "after", "when", "while"])
        else "motion"
        if any(word in words for word in ["moving", "stationary"])
        else "existence"
    )
    group = groups.setdefault(kind, {"count": 0, "correct": 0})
    group["count"] += 1
    group["correct"] += record["target"] == record["prediction"]
train = root / "artifacts/s3-pools-video-001/run-a/train.jsonl"
train_rows = [json.loads(line) for line in train.read_text().splitlines()]
assert all(row["meta"]["pool"] == "train" for row in train_rows)
positive = [row["probabilities"]["yes"] for row in pred if row["target"] == "yes"]
negative = [row["probabilities"]["yes"] for row in pred if row["target"] == "no"]
auc = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (
    len(positive) * len(negative)
)
result = {
    "status": "passed",
    "scope": "CPU aggregate development-error and training-label audit; no heldout access",
    "dev_sha256": r["data_sha256"],
    "sidecar_sha256": r["sidecar_sha256"],
    "train_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
    "confusion": dict(confusion),
    "yes_probability_roc_auc": auc,
    "question_prefix_groups": groups,
    "train_labels": dict(Counter(row["questions"][0]["target"]["value"] for row in train_rows)),
}
(root / "reports/s3-video-error-audit-002.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
