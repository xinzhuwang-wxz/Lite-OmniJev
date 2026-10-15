"""Reconstruct deterministic training draws from hash-bound saved run inputs."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
results = []
for report_name, run_dir in [
    ("s2-smoke-run-001.json", "smoke-001"),
    ("s2-joint-run-001.json", "joint-001"),
    ("s2-joint-v2-run-001.json", "joint-v2-001"),
]:
    p = root / "reports" / report_name
    r = json.loads(p.read_text())
    data = root / "artifacts/s2-text-lora-001" / run_dir / "data.jsonl"
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    assert digest == r["data_sha256"]
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    cfg = r["config"]
    assert len(rows) == cfg["sample_count"]
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)

    def cohort(row):
        return "screenshot" if "gui" in row["meta"]["data_version"] else row["meta"]["modality"][0]

    draws = [rows[order[i % len(rows)]] for i in range(r["steps_completed"] * cfg["accumulation"])]
    adapter = data.parent / "adapter/adapter_model.safetensors"
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == r["adapter_sha256"]
    results.append(
        {
            "report": str(p),
            "report_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "data_sha256": digest,
            "pool_rows": len(rows),
            "pool_counts": dict(Counter(map(cohort, rows))),
            "steps": r["steps_completed"],
            "samples_seen": len(draws),
            "exposures": dict(Counter(map(cohort, draws))),
            "unique_exposed_families": {
                c: len({x["meta"]["family_id"] for x in draws if cohort(x) == c})
                for c in sorted(set(map(cohort, draws)))
            },
            "midway_adapters": r.get("midway_adapters", []),
            "adapter_sha256": r["adapter_sha256"],
            "seconds": r["duration_seconds"],
        }
    )
out = {
    "status": "passed",
    "scope": (
        "reconstruction from saved input SHA, seed and trainer fixed-shuffle modulo order"
        "; not new training or independent quality evidence"
    ),
    "runs": results,
}
(root / "reports/s2-training-exposure-audit-001.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
