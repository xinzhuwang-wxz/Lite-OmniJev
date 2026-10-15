"""Freeze the video pilot slice and reuse hash-verified existing model predictions."""

import hashlib
import json
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
report_path = root / "reports/s3-v2-video-dev500-diagnostic-001.json"
report = json.loads(report_path.read_text())
command = report["command"]
data = Path(command[command.index("--data") + 1])
records = Path(command[command.index("--output-dir") + 1]) / "records.jsonl"
assert hashlib.sha256(data.read_bytes()).hexdigest() == report["data_sha256"]
assert hashlib.sha256(records.read_bytes()).hexdigest() == report["sidecar_sha256"]
assert (
    report["adapter_sha256"] == "db45fb13e651acf07ed5901584960f7bfa60abb1f1a26c609a6a89511c35a4ee"
)
rows = [json.loads(line) for line in data.read_text().splitlines()]
assert len(rows) == 500 and all(row["meta"]["pool"] == "dev" for row in rows)
selected = sorted(
    rows, key=lambda r: hashlib.sha256(("s3-quality-probe:42:" + r["id"]).encode()).hexdigest()
)[:150]
ids = {r["id"] for r in selected}
predictions = [
    r
    for r in map(json.loads, records.read_text().splitlines())
    if r["pass"] == 0 and r["variant"] == "original" and r["id"] in ids
]
assert len(predictions) == len(ids) == 150
output = root / "artifacts/s3-stable-probe-001/video-dev150.jsonl"
assert not output.exists()
output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
correct = sum(r["prediction"] == r["target"] for r in predictions)
result = {
    "status": "passed",
    "scope": "CPU extraction of existing GPU development predictions; no new inference",
    "count": 150,
    "baseline_correct": correct,
    "baseline_accuracy": correct / 150,
    "expansion_required_correct": max(95, correct + 3),
    "video_frames": 16,
    "adapter_sha256": report["adapter_sha256"],
    "parent_data_sha256": report["data_sha256"],
    "data_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "data_path": str(output),
    "sidecar_sha256": report["sidecar_sha256"],
    "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
}
(root / "reports/s3-video150-reference-001.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
