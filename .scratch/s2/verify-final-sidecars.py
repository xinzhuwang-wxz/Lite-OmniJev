"""Independently recheck stored S2 prediction artifacts without rerunning the model."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.s1_dev_eval import metrics_from_records

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cohorts",
    nargs="+",
    choices=["image", "screenshot", "text"],
    default=["image", "screenshot", "text"],
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
root = Path("/root/autodl-tmp/s1-text-lora-001")
results = []
for cohort in args.cohorts:
    path = root / f"reports/s2-dev-eval-{cohort}-v2-budget1120-001.json"
    report = json.loads(path.read_text())
    assert report["status"] == "passed" and report["exit_code"] == 0
    assert report["samples"] == 500 and report["passes"] == 2 and report["repeat_identical"] is True
    command = report["command"]
    data = Path(command[command.index("--data") + 1])
    sidecar = Path(command[command.index("--output-dir") + 1]) / "records.jsonl"
    assert hashlib.sha256(data.read_bytes()).hexdigest() == report["data_sha256"]
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == report["sidecar_sha256"]
    adapter = Path(report["adapter_dir"]) / "adapter_model.safetensors"
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == report["adapter_sha256"]
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    passes = [
        [{k: v for k, v in row.items() if k != "pass"} for row in records if row["pass"] == i]
        for i in [0, 1]
    ]
    assert passes[0] == passes[1]
    originals = [row for row in passes[0] if row["variant"] == "original"]
    assert len(originals) == len({row["id"] for row in originals}) == 500
    measured = metrics_from_records(originals)
    assert all(report["metrics"][key] == value for key, value in measured.items())
    if cohort != "text":
        counterfactuals = [row for row in passes[0] if row["variant"] == "counterfactual"]
        assert len(counterfactuals) == 500
        assert (
            sum(row["content_match"] for row in counterfactuals) / 500
            == report["metrics"]["counterfactual_content_match_rate"]
        )
        assert report["image_max_soft_tokens"] == {"requested": 1120, "effective": 1120}
    results.append(
        {
            "cohort": cohort,
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "data_sha256": report["data_sha256"],
            "sidecar_sha256": report["sidecar_sha256"],
            "adapter_sha256": report["adapter_sha256"],
            "records": len(records),
            "repeat_exact": True,
            "recomputed_metrics_exact": True,
        }
    )
result = {
    "status": "passed" if set(args.cohorts) == {"image", "screenshot", "text"} else "partial",
    "exit_code": 0,
    "scope": "CPU sidecar/data/adapter verification; numerical gate remains separate",
    "cohorts": results,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
