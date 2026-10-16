"""I2 offline signal: base entropy + teacher-base disagreement vs candidate errors.

Zero GPU. Joins three existing per-item record sets on the original video dev500:
pure frozen base (no adapter), OneJev-27B teacher, candidate e465.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "artifacts/s4-ensemble-oracle-001"


def load_cand(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def load_base(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("pass_index", 1) == 1:
            rows[r["id"]] = r
    return rows


def load_teacher(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("pass_index", 1) == 1:
            rows[r["id"]] = r
    return rows


def entropy_bits(distribution: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0)


def main() -> int:
    cand = load_cand(ROOT / "video500-candidate-e465.jsonl")
    base = load_base(ROOT / "video500-purebase.records.jsonl")
    teacher_path = Path(
        "/Users/bamboo/Githubs/Lite-OmniJev/artifacts/s0-teacher-development-gpu-001/video/records.jsonl"
    )
    teacher = load_teacher(teacher_path)
    ids = sorted(set(cand) & set(base) & set(teacher))
    print("joined items:", len(ids))
    rows = []
    for i in ids:
        gold = ["yes", "no"][base[i]["target_index"]]
        base_probs = base[i]["conditional_without_u"]
        teacher_probs_list = teacher[i]["probabilities"]
        teacher_yes = teacher_probs_list[0]  # [p_yes, p_no] order per record format
        base_argmax = max(base_probs, key=base_probs.get)
        teacher_argmax = "yes" if teacher_yes >= 0.5 else "no"
        rows.append(
            {
                "id": i,
                "gold": gold,
                "base_entropy": entropy_bits(base_probs),
                "base_teacher_disagree": base_argmax != teacher_argmax,
                "cand_correct": cand[i]["prediction"] == gold,
                "cand_confidence": cand[i]["confidence"],
            }
        )
    # candidate error rate vs base-entropy terciles
    rows.sort(key=lambda r: r["base_entropy"])
    thirds = [rows[: len(rows) // 3], rows[len(rows) // 3 : 2 * len(rows) // 3], rows[2 * len(rows) // 3 :]]
    print("base-entropy terciles (candidate error rate):")
    for index, bucket in enumerate(thirds):
        errors = sum(1 for r in bucket if not r["cand_correct"])
        print(f"  tercile {index}: mean H={sum(r['base_entropy'] for r in bucket)/len(bucket):.3f} err={errors/len(bucket):.3f}")
    flagged_disagree = [r for r in rows if r["base_teacher_disagree"]]
    print(f"base-teacher argmax disagreement: {len(flagged_disagree)} items, candidate error rate {sum(1 for r in flagged_disagree if not r['cand_correct'])/max(1,len(flagged_disagree)):.3f}")
    agree = [r for r in rows if not r["base_teacher_disagree"]]
    print(f"agreement items: {len(agree)}, candidate error rate {sum(1 for r in agree if not r['cand_correct'])/max(1,len(agree)):.3f}")
    for tau in (0.5, 0.8, 0.95):
        flagged = [r for r in rows if r["base_entropy"] > tau or r["base_teacher_disagree"]]
        if not flagged:
            continue
        errors = sum(1 for r in flagged if not r["cand_correct"])
        unflagged_errors = sum(1 for r in rows if r not in flagged and not r["cand_correct"])
        print(
            f"tau={tau}: flagged {len(flagged)} ({len(flagged)/len(rows):.1%}), "
            f"err-in-flagged {errors/len(flagged):.3f}, err-in-unflagged {unflagged_errors/max(1,len(rows)-len(flagged)):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
