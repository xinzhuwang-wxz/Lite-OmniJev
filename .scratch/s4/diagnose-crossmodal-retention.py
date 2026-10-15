"""Aggregate same-input public-dev changes; no new model runs or training data."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.s0_development_baselines import answer_tokens_for_record

root = Path("/root/autodl-tmp/s1-text-lora-001")
paths = {
    "retained": root / "artifacts/s4-pilot-crossmodal-original-001/records.jsonl",
    "video_ce": root / "artifacts/s4-video-pilot-crossmodal-original-001/records.jsonl",
    "reference_kl": root / "artifacts/s4-reference-retention-screen-001/original/records.jsonl",
    "coverage": root / "artifacts/s4-crossmodal-retention-screen-001/original/records.jsonl",
}
records = {
    name: [json.loads(x) for x in path.read_text().splitlines()] for name, path in paths.items()
}
indexes = {name: {r["id"]: r for r in rows} for name, rows in records.items()}
assert all(len(rows) == 50 and set(rows) == set(indexes["retained"]) for rows in indexes.values())
report = {
    "scope": "50 fixed original public-dev paired diagnosis, not stage acceptance",
    "sources": {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items()
    },
    "models": {},
    "paired": {},
}


def prediction(row):
    labels = answer_tokens_for_record(row)
    return labels[max(range(len(labels)), key=lambda i: row["conditional_without_u"][labels[i]])]


for name, rows in records.items():
    confusion = Counter()
    correct = 0
    yes = []
    for r in rows:
        labels = answer_tokens_for_record(r)
        assert labels == ("yes", "no")
        gold = labels[r["target_index"]]
        pred = prediction(r)
        confusion[gold + "->" + pred] += 1
        correct += pred == gold
        yes.append(r["conditional_without_u"]["yes"])
    report["models"][name] = {
        "correct": correct,
        "count": 50,
        "confusion": dict(confusion),
        "mean_p_yes": math.fsum(yes) / 50,
    }
for name in ["video_ce", "reference_kl", "coverage"]:
    corrected = regressed = unchanged = 0
    for key, a in indexes["retained"].items():
        b = indexes[name][key]
        for field in ["input_sha256", "payload_sha256", "answer_token_ids", "target_index"]:
            assert a[field] == b[field], field
        gold = answer_tokens_for_record(a)[a["target_index"]]
        x = prediction(a) == gold
        y = prediction(b) == gold
        corrected += not x and y
        regressed += x and not y
        unchanged += prediction(a) == prediction(b)
    report["paired"][name] = {
        "corrected": corrected,
        "regressed": regressed,
        "unchanged_predictions": unchanged,
    }
report["coverage_vs_ce_changed_predictions"] = sum(
    prediction(indexes["video_ce"][k]) != prediction(indexes["coverage"][k])
    for k in indexes["retained"]
)
report["interpretation"] = (
    "Coverage improved original cross by one vs video CE, but missed the frozen retention "
    "minimum. This paired public-dev diagnostic does not establish gradient conflict "
    "or stage acceptance."
)
with (root / "reports/s4-crossmodal-retention-diagnostic-001.json").open("x") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report))
