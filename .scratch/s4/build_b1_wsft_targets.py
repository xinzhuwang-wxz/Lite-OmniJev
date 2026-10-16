"""B1+I1 WSFT layer: predicate-factorized soft targets over the B1 grid.

Contract-legal only: SoftTarget over real answers (U implicit zero) and
InsufficientTarget. Partial-U mass on perception-unresolvable order cells is
expressed by deterministic row duplication (hard:U = 7:3 or 17:3), which
reproduces the mixture CE up to integer quantization. Observability is judged
against the actual uniform 16-frame sampling of the family video, derived from
the same official annotation truth the grid was built from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EPSILON = 0.05  # near-miss softness on the opposite polarity (cap well below 0.3)
HARD_U_DUPLICATION = (7, 3)  # fully-determined but perception-unresolvable: 70/30 hard:U
SOFT_U_DUPLICATION = (17, 3)  # borderline (single bracketing frame): 85/15
UNIFORM_FRAMES = 16


def sampled_indices(total_frames: int) -> list[int]:
    return [round(i * (total_frames - 1) / (UNIFORM_FRAMES - 1)) for i in range(UNIFORM_FRAMES)]


def order_observability(first: int, second: int, indices: list[int]) -> str:
    """Classify whether the 16-frame input can resolve first<second."""
    if first == second:
        return "tie"
    low, high = (first, second) if first < second else (second, first)
    separators = [s for s in indices if low < s <= high]
    if not separators:
        return "unresolvable"
    if len(separators) == 1:
        return "bracketed_tight"
    return "resolvable"


def soft_binary(gold_yes: bool, epsilon: float) -> dict:
    return {
        "kind": "soft",
        "probs": [1 - epsilon, epsilon] if gold_yes else [epsilon, 1 - epsilon],
        "source": "b1_wsft_predicate_factorized",
    }


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: build_b1_wsft_targets.py GRID_JSONL GRID_MANIFEST OUTPUT_JSONL"
        )
    grid_path, manifest_path, out_path = (Path(a) for a in sys.argv[1:])
    rows = {json.loads(line)["questions"][0]["qid"]: json.loads(line) for line in grid_path.read_text().splitlines() if line.strip()}
    manifest = json.loads(manifest_path.read_text())
    family = manifest["families"][0]
    total_frames = family["final_frame"] + 1
    indices = sampled_indices(total_frames)
    derivation_index = {d["qid"]: d for d in manifest["derivations"]}
    emitted: list[dict] = []
    audit: list[dict] = []
    for qid, row in rows.items():
        derivation = derivation_index[qid]
        cell = derivation["cell"]
        target = row["questions"][0]["target"]
        gold_yes = target["value"] == "yes" if target["kind"] == "hard" else None
        observability = "applicable"
        if cell.startswith("order_direct") and gold_yes is not None:
            observability = order_observability(
                derivation["entry_left"], derivation["entry_right"], indices
            )
        elif cell.startswith("collision_order") and gold_yes is not None:
            frames = [derivation["first"][-1], derivation["second"][-1]]
            observability = order_observability(frames[0], frames[1], indices)
        elif cell.startswith("enter_collision") and gold_yes is not None:
            frames = [derivation["entry"], derivation["collision"][-1]]
            observability = order_observability(frames[0], frames[1], indices)
        if target["kind"] == "insufficient":
            emitted.append(row)
            audit.append({"qid": qid, "form": "hard_U", "observability": "premise_undecidable"})
            continue
        if observability in ("unresolvable", "bracketed_tight"):
            hard, u_copies = (
                HARD_U_DUPLICATION if observability == "unresolvable" else SOFT_U_DUPLICATION
            )
            for copy in range(hard):
                copy_row = json.loads(json.dumps(row))
                copy_row["id"] = f"{row['id']}-wsft{copy}"
                emitted.append(copy_row)
            insufficient_row = json.loads(json.dumps(row))
            insufficient_row["questions"][0]["target"] = {"kind": "insufficient"}
            for copy in range(u_copies):
                insufficient_row["id"] = f"{row['id']}-wsftu{copy}"
                emitted.append(json.loads(json.dumps(insufficient_row)))
            audit.append(
                {
                    "qid": qid,
                    "form": f"duplication_{hard}:{u_copies}",
                    "observability": observability,
                    "u_mass": u_copies / (hard + u_copies),
                }
            )
            continue
        soft_row = json.loads(json.dumps(row))
        soft_row["questions"][0]["target"] = soft_binary(gold_yes, EPSILON)
        emitted.append(soft_row)
        audit.append(
            {
                "qid": qid,
                "form": "soft_binary",
                "observability": observability,
                "epsilon": EPSILON,
            }
        )
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in emitted), encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for entry in audit:
        counts[entry["form"]] = counts.get(entry["form"], 0) + 1
    summary = {
        "rows_out": len(emitted),
        "base_items": len(rows),
        "forms": counts,
        "sampled_frame_indices": indices,
        "total_frames": total_frames,
        "epsilon": EPSILON,
    }
    print(json.dumps(summary, indent=2))
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps({"summary": summary, "audit": audit}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
