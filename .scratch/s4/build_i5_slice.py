"""Deterministic I5/D3P slice: first 150 TRAIN video rows by sha256('i5:d3p:'+id)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

TRAIN_SHA = "912c871f7467b36ea1837276536250539f16f2ae6c5eaf32f867912b9e49e948"


def main() -> int:
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
    selected = ranked[:150]
    out_dir = pool.parent
    (out_dir / "i5-slice-150.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8"
    )
    (out_dir / "i5-slice-150-ledger.jsonl").write_text(
        "".join(json.dumps({"item_id": r["id"], "pool": "train"}) + "\n" for r in selected),
        encoding="utf-8",
    )
    summary = {
        "pool_rows": len(rows),
        "selected": len(selected),
        "slice_sha256": hashlib.sha256((out_dir / "i5-slice-150.jsonl").read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256((out_dir / "i5-slice-150-ledger.jsonl").read_bytes()).hexdigest(),
        "first_id": selected[0]["id"],
        "last_id": selected[-1]["id"],
        "rule": "video rows sorted by sha256('i5:d3p:'+id), first 150",
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
