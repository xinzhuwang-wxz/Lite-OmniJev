"""Aggregate only existing public dev predictions and TRAIN exposure; no reserved pools."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from data.schema import Sample, answer_tokens
from train.readout import read_probabilities

root = Path("/root/autodl-tmp/s1-text-lora-001")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def kind(row):
    words = set(row["questions"][0]["instruction"].lower().split())
    return (
        "temporal"
        if words & {"before", "after", "when", "while"}
        else "motion"
        if words & {"moving", "stationary"}
        else "existence"
    )


def summarize(report_path):
    report = json.loads(report_path.read_text())
    command = report["command"]
    data = Path(command[command.index("--data") + 1])
    sidecar = Path(command[command.index("--output-dir") + 1]) / "records.jsonl"
    assert sha(data) == report["data_sha256"] and sha(sidecar) == report["sidecar_sha256"]
    samples = {r["id"]: r for r in rows(data)}
    assert all(r["meta"]["pool"] == "dev" for r in samples.values())
    records = [r for r in rows(sidecar) if r["pass"] == 0 and r["variant"] == "original"]
    groups = {}
    confusion = Counter()
    predictions = {}
    for r in records:
        sample = Sample.model_validate(samples[r["id"]])
        q = sample.questions[0]
        pred = read_probabilities(r["raw_logits"], list(answer_tokens(q)), temperature=1).prediction
        assert q.target.value == r["target"]
        g = groups.setdefault(kind(samples[r["id"]]), {"count": 0, "correct": 0})
        g["count"] += 1
        g["correct"] += pred == r["target"]
        confusion[r["target"] + "->" + pred] += 1
        predictions[r["id"]] = pred == r["target"]
    return {
        "data_sha256": sha(data),
        "sidecar_sha256": sha(sidecar),
        "groups": groups,
        "confusion": dict(confusion),
    }, predictions


before, bp = summarize(root / "reports/s3-stable-video-screen-001.json")
after, ap = summarize(root / "artifacts/s4-pilot-five-screen-001/video-report.json")
assert set(bp) == set(ap) and before["data_sha256"] == after["data_sha256"]
source = root / "artifacts/s3-pools-video-001/run-a/train.jsonl"
train = rows(source)
assert all(r["meta"]["pool"] == "train" for r in train)
previous = rows(root / "artifacts/s3-stable-probe-001/train.jsonl")
seen = {r["id"] for r in previous if r["meta"]["modality"] == ["video"]}
pilot = rows(root / "artifacts/s4-joint-pilot-001/train.jsonl")
order = list(range(len(pilot)))
random.Random(42).shuffle(order)
actual = [pilot[i] for i in order[:300] if pilot[i]["meta"]["modality"] == ["video"]]
unseen = [r for r in train if r["id"] not in seen]
report = {
    "scope": "public dev150 and known TRAIN-only exposure diagnostic; no holdout",
    "before": before,
    "after": after,
    "paired": dict(Counter(f"{int(bp[i])}->{int(ap[i])}" for i in bp)),
    "training": {
        "sha256": sha(source),
        "available": len(train),
        "s3_stable_video_seen": len(seen),
        "pilot_video_draws": len(actual),
        "pilot_video_new_beyond_s3": sum(r["id"] not in seen for r in actual),
        "unseen_capacity": len(unseen),
        "unseen_groups": dict(Counter(kind(r) for r in unseen)),
        "unseen_targets": dict(Counter(r["questions"][0]["target"]["value"] for r in unseen)),
    },
    "exit_code": 0,
}
p = root / "reports/s4-video-pilot-diagnostic-001.json"
with p.open("x") as h:
    json.dump(report, h, indent=2)
print(json.dumps(report))
