"""Record CPU-only metric corrections without replacing original GPU evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.s1_dev_eval import metrics_from_records

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
report = json.loads(args.report.read_text())
assert report["status"] == "passed" and report["exit_code"] == 0
sidecar_sha = hashlib.sha256(args.records.read_bytes()).hexdigest()
assert sidecar_sha == report["sidecar_sha256"]
records = [json.loads(line) for line in args.records.read_text().splitlines()]
originals = [row for row in records if row["pass"] == 0 and row["variant"] == "original"]
assert len(originals) == len({row["id"] for row in originals}) == report["samples"]
corrected = metrics_from_records(originals)
assert corrected["product_accuracy"] == report["metrics"]["product_accuracy"]
assert corrected["accuracy"] == corrected["product_accuracy"]
result = {
    "status": "passed",
    "exit_code": 0,
    "scope": "CPU correction of complete dev sidecar metrics; no model run or quality gate",
    "reason": "Noul ties use canonical yes/no order, independent of serialized JSON key order",
    "original_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
    "sidecar_sha256": sidecar_sha,
    "metric_code_sha256": hashlib.sha256(Path("scripts/s1_dev_eval.py").read_bytes()).hexdigest(),
    "model_revision": report["model_revision"],
    "adapter_sha256": report["adapter_sha256"],
    "data_sha256": report["data_sha256"],
    "samples": len(originals),
    "metrics": corrected,
    "changed_metrics": {
        key: {"original": report["metrics"][key], "corrected": value}
        for key, value in corrected.items()
        if value != report["metrics"][key]
    },
    "tied_top_rows": sum(
        sum(p == max(row["probabilities"].values()) for p in row["probabilities"].values()) > 1
        for row in originals
    ),
    "product_predictions_changed": False,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
