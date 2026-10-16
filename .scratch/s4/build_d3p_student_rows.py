"""D3P student rows: teacher cache records -> SoftTarget training rows.

CPU only; runs on the GPU box after the teacher cache lands. The soft target
is the teacher's conditional legal distribution (the existing readout
protocol); teacher p_U, where present, is recorded in the row's meta as
evidence and folded out by the readout's own normalization. No fabricated
targets: items lacking a valid teacher record are dropped and counted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--teacher-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    teacher = {}
    invalid = 0
    for line in args.teacher_records.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        probs = record.get("conditional_without_u")
        if (
            record.get("pass_index", 1) != 1
            or not isinstance(probs, dict)
            or set(probs) != {"yes", "no"}
            or any(v < 0 or v > 1 for v in probs.values())
            or abs(sum(probs.values()) - 1) > 1e-6
        ):
            invalid += 1
            continue
        teacher[record["id"]] = record
    emitted = dropped = 0
    total_p_u = 0.0
    with args.output.open("x", encoding="utf-8") as out, args.ledger.open("x", encoding="utf-8") as ledger:
        for line in args.slice.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            record = teacher.get(row["id"])
            if record is None:
                dropped += 1
                continue
            probs = record["conditional_without_u"]
            yes, no = probs["yes"], probs["no"]
            total = yes + no
            target = {
                "kind": "soft",
                "probs": [round(yes / total, 12), round(no / total, 12)],
                "source": "onejev-27b-conditional-legal",
            }
            row["questions"][0]["target"] = target
            row["meta"] = {
                **row["meta"],
                "data_version": row["meta"]["data_version"] + "+d3p-teacher-soft-v1",
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            ledger.write(json.dumps({"item_id": row["id"], "pool": "train"}) + "\n")
            emitted += 1
            total_p_u += record.get("p_U") or 0.0
    print(
        json.dumps(
            {
                "emitted": emitted,
                "dropped_missing": dropped,
                "invalid_teacher_records": invalid,
                "mean_teacher_p_U": round(total_p_u / max(1, emitted), 8),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
