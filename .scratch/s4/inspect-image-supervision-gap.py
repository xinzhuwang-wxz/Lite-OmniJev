"""Read canonical TRAIN runtime lineage metadata only; never inspect gate pools."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def digest(path):
    with path.open("rb") as h:
        return hashlib.file_digest(h, "sha256").hexdigest()


def summarize(path, steps, accumulation, seed):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows or any(r["meta"]["pool"] != "train" for r in rows):
        raise ValueError("strict_train_input_required")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    seen = [rows[order[d % len(rows)]] for d in range(steps * accumulation)]

    def group(rs):
        v = Counter(r["meta"]["data_version"] for r in rs)
        return {
            k: {
                "rows_or_draws": n,
                "unique_ids": len({r["id"] for r in rs if r["meta"]["data_version"] == k}),
                "unique_families": len(
                    {r["meta"]["family_id"] for r in rs if r["meta"]["data_version"] == k}
                ),
            }
            for k, n in sorted(v.items())
        }

    image = [
        r
        for r in seen
        if "aokvqa" in r["meta"]["data_version"] or "vqa2" in r["meta"]["data_version"]
    ]
    return {
        "path": str(path),
        "sha256": digest(path),
        "stored_rows": len(rows),
        "actual_draws": len(seen),
        "storage_by_version": group(rows),
        "exposure_by_version": group(seen),
    }, image


def run(root):
    current = root / "artifacts/s4-temporal-coverage-train-001"
    runs = []
    all_image = []
    s4_image = []
    visited = set()
    while current:
        if current in visited:
            raise ValueError("lineage_cycle")
        visited.add(current)
        receipt = current / "run.json"
        d = json.loads(receipt.read_bytes())
        config = d["config"]
        if d["status"] not in ["completed", "paused"] or d.get("steps_completed", 0) <= 0:
            raise ValueError("non_terminal_lineage")
        data = current / "data.jsonl"
        if digest(data) != d["data_sha256"]:
            raise ValueError("runtime_data_sha")
        row, image = summarize(data, d["steps_completed"], config["accumulation"], config["seed"])
        row.update(
            {
                "report": str(receipt),
                "report_sha256": digest(receipt),
                "steps": d["steps_completed"],
                "accumulation": config["accumulation"],
                "seed": config["seed"],
                "image_max_soft_tokens": config.get("image_max_soft_tokens", "not_explicit"),
                "adapter_sha256": d.get("adapter_sha256"),
                "init_adapter": d.get("init_adapter"),
            }
        )
        runs.append(row)
        all_image.extend(image)
        if current.name.startswith("s4-"):
            s4_image.extend(image)
        init = d.get("init_adapter") or config.get("init_adapter")
        if not init:
            break
        parent = Path(init["path"]).parent
        if not (parent / "run.json").is_file():
            raise ValueError("missing_parent_run:" + str(parent))
        pd = json.loads((parent / "run.json").read_bytes())
        weights = pd.get("adapter_sha256")
        if isinstance(weights, dict):
            weights = weights.get("weights_sha256", weights.get("adapter_model.safetensors"))
        if weights != init["weights_sha256"]:
            raise ValueError("parent_adapter_identity")
        current = parent
    pools = []
    for rel in ["artifacts/s2-vqa2-gold-001/train.jsonl", "artifacts/s2-vqa-probe-001/train.jsonl"]:
        path = root / rel
        row, _ = summarize(path, 0, 1, 42)
        pools.append(row)
    return {
        "status": "read_only_train_lineage_audited",
        "runs_current_to_oldest": runs,
        "lineage_image_draws": len(all_image),
        "lineage_unique_image_ids": len({r["id"] for r in all_image}),
        "lineage_unique_image_families": len({r["meta"]["family_id"] for r in all_image}),
        "lineage_image_version_counts": dict(Counter(r["meta"]["data_version"] for r in all_image)),
        "s4_image_draws": len(s4_image),
        "s4_unique_image_ids": len({r["id"] for r in s4_image}),
        "s4_unique_image_families": len({r["meta"]["family_id"] for r in s4_image}),
        "existing_train_capacities_not_exposures": pools,
        "gpu_used": False,
        "cost": None,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    print(json.dumps(run(p.parse_args().root), indent=2))
