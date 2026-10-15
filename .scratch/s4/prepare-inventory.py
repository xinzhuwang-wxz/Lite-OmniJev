"""Freeze a first bounded historical-training/current-reserve audit manifest."""

import hashlib
import json
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
out = root / "artifacts/s4-gold-inventory-001"
index = json.loads((out / "historical-training-index.json").read_text())
assert not index["unresolved"]
versions = {
    "aokvqa-train-image-choice-s2-pools-v1": "image",
    "aokvqa-train-image-choice-s2-smooth-v1": "image",
    "aokvqa-train-image-choice-s2-teacher-soft-v1": "image",
    "clevrer-train-video-noul-s3-pools-v1": "video",
    "clinc-oos-plus-train-v1-s1-pools-v1": "text",
    "clotho-aqa-s3-pools-v1": "audio",
    "gui-v2-screenshot-s2-pools-v1": "screenshot",
    "program-v1": "text",
    "vqa2-existing-train-choice-v1": "image",
    "vqa2-new-family-choice-v1": "image",
}
samples = []
for entry in index["files"]:
    assert set(entry["data_versions"]) <= set(versions)
    samples.append(
        {
            "path": entry["path"],
            "sha256": entry["sha256"],
            "cohorts_by_data_version": {v: versions[v] for v in entry["data_versions"]},
            "exposure": "historical",
            "expected_pool": "train",
        }
    )
cohorts = {
    "text": "s1-text-pools-v1-001",
    "image": "s2-media-pools-image-001",
    "screenshot": "s2-media-pools-screenshot-001",
    "audio": "s3-pools-audio-001",
    "video": "s3-pools-video-001",
}
for cohort, directory in cohorts.items():
    source = root / "artifacts" / directory / "run-a"
    report = json.loads((source / "report.json").read_text())
    assert report["status"] == "passed"
    for pool, filename in [
        ("train", "train.jsonl"),
        ("calibration", "calibration-reserve.jsonl"),
        ("holdout", "holdout-reserve.jsonl"),
    ]:
        path = source / filename
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha in report["outputs"].values()
        samples.append(
            {
                "path": str(path),
                "sha256": sha,
                "cohort": cohort,
                "exposure": "current",
                "expected_pool": pool,
            }
        )
manifest = {
    "schema_version": "gold-inventory-manifest-v1",
    "samples": samples,
    "component_ledgers": [],
}
p = out / "training-reserve-manifest.json"
assert not p.exists()
p.write_text(json.dumps(manifest, indent=2) + "\n")
print(
    json.dumps(
        {
            "status": "prepared",
            "artifacts": len(samples),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "scope": "training/reserve only; historical dev and crossmodal still absent",
        }
    )
)
