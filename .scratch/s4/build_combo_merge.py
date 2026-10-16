"""COMBO formal merge: D3P soft-target rows + ii5v SAS package, one version, deduped.

Merge rules (fixing the blocked naive concatenations):
1. every output row carries the single unified data_version
   clevrer-train-video-noul-s3-pools-v1+ii5v-v1+d3p-soft-v1;
2. id dedup with D3P priority: where a D3P soft-target row and an SAS main row
   share an item id, the D3P row wins and the SAS main is dropped;
3. ledger rows use the trainer's {item_id, pool} shape over the merged set.
"""

from __future__ import annotations

import json
from pathlib import Path

UNIFIED_VERSION = "clevrer-train-video-noul-s3-pools-v1+ii5v-v1+d3p-soft-v1"


def main() -> int:
    root = Path(__file__).resolve().parents[2] / "artifacts/s4-ensemble-oracle-001"
    d3p_rows = [
        json.loads(line)
        for line in (root / "d3p-student-150.jsonl").read_text().splitlines()
        if line.strip()
    ]
    sas_rows = [
        json.loads(line)
        for line in (root / "ii5v-mixed.jsonl").read_text().splitlines()
        if line.strip()
    ]
    d3p_ids = {row["id"] for row in d3p_rows}
    if len(d3p_ids) != len(d3p_rows):
        raise SystemExit("d3p rows contain duplicate ids")
    dropped_sas = [row["id"] for row in sas_rows if row["id"] in d3p_ids]
    merged = [*d3p_rows, *(row for row in sas_rows if row["id"] not in d3p_ids)]
    for row in merged:
        row["meta"]["data_version"] = UNIFIED_VERSION
    ids = [row["id"] for row in merged]
    if len(set(ids)) != len(ids):
        raise SystemExit("merged set contains duplicate ids")
    (root / "combo-merged.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8"
    )
    (root / "combo-merged-ledger.jsonl").write_text(
        "".join(json.dumps({"item_id": r["id"], "pool": "train"}) + "\n" for r in merged),
        encoding="utf-8",
    )
    import hashlib

    summary = {
        "d3p_rows": len(d3p_rows),
        "sas_rows": len(sas_rows),
        "dropped_sas_duplicates": len(dropped_sas),
        "merged_rows": len(merged),
        "merged_sha256": hashlib.sha256((root / "combo-merged.jsonl").read_bytes()).hexdigest(),
        "ledger_sha256": hashlib.sha256(
            (root / "combo-merged-ledger.jsonl").read_bytes()
        ).hexdigest(),
        "unified_data_version": UNIFIED_VERSION,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
