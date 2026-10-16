"""COMBO-FULL: three-source structured supervision in one training set.

Sources (priority order for id dedup):
1. D3P-600 teacher soft targets (600 rows, teacher conditional legal dist);
2. WSFT world-state factorized grid (292 rows incl. partial-U duplication);
3. SAS full-family program facets (3215 rows: 210 mains + 3005 facets).

One unified data_version on every row; colliding ids keep the higher-priority
source; ledger rows use the trainer's {item_id, pool} shape. U mass is legal
inside training per the revised ADR0024 (product-form ban only).
"""

from __future__ import annotations

import json
from pathlib import Path

UNIFIED_VERSION = "clevrer-train-video-noul-s3-pools-v1+combo3-d3p-wsft-sas-v1"


def main() -> int:
    root = Path(__file__).resolve().parents[2] / "artifacts/s4-ensemble-oracle-001"
    d3p_path = root / "d3p-student-600.jsonl"
    if not d3p_path.exists():
        raise SystemExit("missing d3p-student-600.jsonl (fetch from remote first)")
    sources = [
        ("d3p", [json.loads(l) for l in d3p_path.read_text().splitlines() if l.strip()]),
        (
            "wsft",
            [
                json.loads(l)
                for l in (root / "b1-wsft-grid.jsonl").read_text().splitlines()
                if l.strip()
            ],
        ),
        (
            "sas",
            [
                json.loads(l)
                for l in (root / "ii5v-full-mixed.jsonl").read_text().splitlines()
                if l.strip()
            ],
        ),
    ]
    merged: list[dict] = []
    seen: set[str] = set()
    dropped = {}
    for name, rows in sources:
        if len({row["id"] for row in rows}) != len(rows):
            raise SystemExit(f"{name} rows contain duplicate ids")
        keep = []
        for row in rows:
            if row["id"] in seen:
                dropped[name] = dropped.get(name, 0) + 1
                continue
            seen.add(row["id"])
            keep.append(row)
        merged.extend(keep)
    for row in merged:
        row["meta"]["data_version"] = UNIFIED_VERSION
    ids = [row["id"] for row in merged]
    if len(set(ids)) != len(ids):
        raise SystemExit("merged set contains duplicate ids")
    (root / "combo-full.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged), encoding="utf-8"
    )
    (root / "combo-full-ledger.jsonl").write_text(
        "".join(json.dumps({"item_id": r["id"], "pool": "train"}) + "\n" for r in merged),
        encoding="utf-8",
    )
    import hashlib
    from collections import Counter

    kinds = Counter(row["questions"][0]["target"]["kind"] for row in merged)
    counts = {
        "d3p": 600 - dropped.get("d3p", 0),
        "wsft": 292 - dropped.get("wsft", 0),
        "sas": 3215 - dropped.get("sas", 0),
    }
    print(
        json.dumps(
            {
                "merged_rows": len(merged),
                "kept_by_source": counts,
                "dropped_by_priority": dropped,
                "target_kinds": dict(kinds),
                "merged_sha256": hashlib.sha256((root / "combo-full.jsonl").read_bytes()).hexdigest(),
                "ledger_sha256": hashlib.sha256(
                    (root / "combo-full-ledger.jsonl").read_bytes()
                ).hexdigest(),
                "unified_data_version": UNIFIED_VERSION,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
