"""Generate K gold-preserving prompt variants per B1/WSFT grid row (R1-B groups).

Wrappers are semantically inert surface forms (they never touch comparatives,
objects, or polarity), so gold invariance holds by construction and is verified
by unwrapping. Variants reuse the WSFT target layer row-for-row. CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WRAPPERS = [
    ("pv0", lambda q: q),
    ("pv1", lambda q: f"Answer yes or no: {q}"),
    ("pv2", lambda q: f"Is the following true? {q}"),
    ("pv3", lambda q: f"{q} Answer yes or no."),
    ("pv4", lambda q: f"Based on the video: {q}"),
    ("pv5", lambda q: f"Question: {q}"),
    ("pv6", lambda q: f"Watch carefully. {q}"),
    ("pv7", lambda q: f"Decide: {q}"),
]


def unwrap(text: str, original: str) -> bool:
    """A valid variant must contain the original instruction verbatim."""
    return original in text


def build(grid_path: Path, out_path: Path, ledger_path: Path, *, wrappers: list) -> dict:
    rows = [json.loads(line) for line in grid_path.read_text().splitlines() if line.strip()]
    emitted = []
    for row in rows:
        original = row["questions"][0]["instruction"]
        for tag, wrap in wrappers:
            variant = json.loads(json.dumps(row))
            variant["id"] = f"{row['id']}-{tag}"
            text = wrap(original)
            if not unwrap(text, original):
                raise SystemExit(f"wrapper broke instruction for {row['id']} {tag}")
            variant["questions"][0]["instruction"] = text
            emitted.append(variant)
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in emitted), encoding="utf-8"
    )
    ledger_path.write_text(
        "".join(json.dumps({"item_id": r["id"], "pool": "train"}) + "\n" for r in emitted),
        encoding="utf-8",
    )
    return {
        "base_rows": len(rows),
        "wrappers": len(wrappers),
        "emitted": len(emitted),
        "unique_ids": len({r["id"] for r in emitted}) == len(emitted),
        "targets_preserved": all(
            json.dumps(emitted[index]["questions"][0]["target"], sort_keys=True)
            == json.dumps(rows[index // len(wrappers)]["questions"][0]["target"], sort_keys=True)
            for index in range(len(emitted))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--wrapper-subset",
        default="all",
        help="comma list of wrapper tags, or 'legacy4' for pv0,pv1,pv2,pv5",
    )
    args = parser.parse_args()
    if args.wrapper_subset == "all":
        wrappers = WRAPPERS
    elif args.wrapper_subset == "legacy4":
        wrappers = [w for w in WRAPPERS if w[0] in {"pv0", "pv1", "pv2", "pv5"}]
    else:
        chosen = set(args.wrapper_subset.split(","))
        wrappers = [w for w in WRAPPERS if w[0] in chosen]
    if not wrappers:
        raise SystemExit("no wrappers selected")
    stats = build(args.grid, args.output, args.ledger, wrappers=wrappers)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
