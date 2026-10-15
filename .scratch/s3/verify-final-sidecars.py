"""Recompute S3 dev evidence from completed artifacts, never partial journals."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.s1_dev_eval import metrics_from_records
from scripts.s3_gate import FROZEN_S3_DEV_SHA256, validate_identity
from train.readout import read_probabilities

COHORTS = {"image", "screenshot", "text", "audio", "video", "video8"}
ADAPTER = "1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755"
MODEL = "ee0ef6023621cff504d758262d4e04895a5af4a2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path, cohort: str) -> dict:
    path = root / f"reports/s3-dev-eval-{cohort}-stable-001.json"
    report = json.loads(path.read_text())
    original_cohort = "video" if cohort == "video8" else cohort
    validate_identity(report, original_cohort)
    assert report["status"] == "passed" and report["exit_code"] == 0
    assert report["adapter_sha256"] == ADAPTER and report["model_revision"] == MODEL
    passes = 1 if cohort == "video8" else 2
    assert report["samples"] == 500 and report["passes"] == passes
    assert report["counterfactual"] is False
    if passes == 2:
        assert report["repeat_identical"] is True
    if cohort in {"image", "screenshot"}:
        assert report["image_max_soft_tokens"] == {"requested": 1120, "effective": 1120}
    if cohort in {"video", "video8"}:
        assert report["video_num_frames"] == (8 if cohort == "video8" else 16)
    command = report["command"]
    data = Path(command[command.index("--data") + 1])
    sidecar = root / command[command.index("--output-dir") + 1] / "records.jsonl"
    assert digest(data) == FROZEN_S3_DEV_SHA256[original_cohort]
    source_rows = [json.loads(line) for line in data.read_text().splitlines()]
    source = {row["id"]: row for row in source_rows}
    assert len(source) == len(source_rows) == 500
    assert digest(sidecar) == report["sidecar_sha256"]
    assert digest(Path(report["adapter_dir"]) / "adapter_model.safetensors") == ADAPTER
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert len(records) == 500 * passes
    grouped = []
    for number in range(passes):
        rows = [
            {key: value for key, value in row.items() if key != "pass"}
            for row in records
            if row["pass"] == number
        ]
        assert len(rows) == len({row["id"] for row in rows}) == 500
        assert all(row["variant"] == "original" for row in rows)
        assert {row["id"] for row in rows} == set(source)
        for row in rows:
            sample = source[row["id"]]
            assert row["target"] == sample["questions"][0]["target"]["value"]
            assert row["family_hash"] == sample["meta"]["family_id"]
            assert row["pool"] == sample["meta"]["pool"] == "dev"
            readout = read_probabilities(
                row["raw_logits"], list(row["probabilities"]), temperature=1.0
            )
            for key in (
                "probabilities",
                "joint_probabilities",
                "p_insufficient",
                "confidence",
                "prediction",
            ):
                assert row[key] == getattr(readout, key)
        grouped.append(rows)
    assert all(rows == grouped[0] for rows in grouped)
    measured = metrics_from_records(grouped[0])
    assert all(report["metrics"][key] == value for key, value in measured.items())
    return {
        "cohort": cohort,
        "report_sha256": digest(path),
        "data_sha256": digest(data),
        "sidecar_sha256": digest(sidecar),
        "adapter_sha256": ADAPTER,
        "records": len(records),
        "passes": passes,
        "repeat_exact": passes == 2,
        "recomputed_metrics_exact": True,
        "source_targets_verified": True,
        "raw_logits_readout_exact": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohorts", nargs="+", choices=sorted(COHORTS), default=sorted(COHORTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert len(set(args.cohorts)) == len(args.cohorts)
    result = {
        "status": "passed" if set(args.cohorts) == COHORTS else "partial",
        "exit_code": 0,
        "scope": "CPU artifact verification only; S3 numerical gate remains separate",
        "cohorts": [verify(args.root, cohort) for cohort in args.cohorts],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
