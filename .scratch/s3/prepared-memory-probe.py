"""Measure tensor bytes on a small train slice before eager full-pool preparation."""

import json
import resource
import time
from pathlib import Path

from data.schema import Sample
from train.trainer import prepare_mixed_examples

root = Path("/root/autodl-tmp/s1-text-lora-001")
start = time.time()
config = json.loads((root / "artifacts/s3-joint-pool-001/main-config.json").read_text())
by = {}
for line in (root / "artifacts/s3-joint-pool-001/joint-train.jsonl").read_text().splitlines():
    s = Sample.model_validate_json(line)
    key = s.meta.modality[0]
    by.setdefault(key, []).append(s)
selected = [s for values in by.values() for s in values[:3]]
examples = prepare_mixed_examples(
    selected,
    Path(
        "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef60236"
        "21cff504d758262d4e04895a5af4a2"
    ),
    Path(config["media_roots"][0]),
    4096,
    extra_roots=[Path(p) for p in config["media_roots"][1:]],
    video_num_frames=16,
)
measurements = {}
for sample, example in zip(selected, examples, strict=True):
    tensors = {
        k: {"shape": list(v.shape), "bytes": v.numel() * v.element_size()}
        for k, v in example.items()
        if hasattr(v, "numel")
    }
    measurements.setdefault(sample.meta.modality[0], []).append(
        {
            "tensor_bytes": sum(t["bytes"] for t in tensors.values()),
            "prompt_tokens": len(example["input_ids"]),
            "tensors": tensors,
        }
    )
report = {
    "status": "completed",
    "exit_code": 0,
    "scope": "CPU representative tensor allocation, estimate not full-pool memory acceptance",
    "samples_per_modality": 3,
    "pool_counts": {k: len(v) for k, v in by.items()},
    "measurements": measurements,
    "estimated_all_prepared_tensor_bytes": sum(
        len(by[k]) * max(r["tensor_bytes"] for r in rows) for k, rows in measurements.items()
    ),
    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    "duration_seconds": time.time() - start,
}
(root / "reports/s3-prepared-memory-probe-001.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
