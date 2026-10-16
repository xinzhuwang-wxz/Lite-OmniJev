"""Deterministic I5/D3P slices: first N TRAIN video rows by sha256('i5:d3p:'+id)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

TRAIN_SHA = "912c871f7467b36ea1837276536250539f16f2ae6c5eaf32f867912b9e49e948"


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    pool = Path(__file__).resolve().parents[2] / "artifacts/s4-ensemble-oracle-001/clevrer-train-pool.jsonl"
    digest = hashlib.sha256(pool.read_bytes()).hexdigest()
    if digest != TRAIN_SHA:
        raise SystemExit(f"pool sha mismatch: {digest}")
    rows = [
        json.loads(line)
        for line in pool.read_text().splitlines()
        if line.strip() and json.loads(line)["meta"]["modality"] == ["video"]
    ]
    ranked = sorted(rows, key=lambda row: hashlib.sha256(("i5:d3p:" + row["id"]).encode()).hexdigest())
    selected = ranked[:count]
    out_dir = pool.parent
    name = f"i5-slice-{count}.jsonl"
    (out_dir / name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8"
    )
    (out_dir / name.replace(".jsonl", "-ledger.jsonl")).write_text(
        "".join(json.dumps({"item_id": r["id"], "pool": "train"}) + "\n" for r in selected),
        encoding="utf-8",
    )
    summary = {
        "pool_rows": len(rows),
        "selected": len(selected),
        "slice_sha256": hashlib.sha256((out_dir / name).read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256(
            (out_dir / name.replace(".jsonl", "-ledger.jsonl")).read_bytes()
        ).hexdigest(),
        "first_id": selected[0]["id"],
        "last_id": selected[-1]["id"],
        "contains_slice_150": len(selected) >= 150,
        "rule": f"video rows sorted by sha256('i5:d3p:'+id), first {count}",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
