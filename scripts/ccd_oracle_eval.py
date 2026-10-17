"""Oracle accuracy of the CCD executor on ground-truth CLEVRER worlds.

Reads a subset directory with ``questions.json`` (official question list) and
``annotations/annotation_NNNNN.json``; builds GT worlds (+ physics for unseen and
counterfactual events), executes every question and writes a JSON accuracy table.
No question is dropped: execution errors count as wrong and are listed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval.ccd_executor import ExecutionError, answer_choice, answer_open
from eval.ccd_physics import PhysicsConfig, simulate_world
from eval.ccd_world import World

FAILURE_SAMPLE = 40


def _ratio(hit: int, total: int) -> float | None:
    return hit / total if total else None


def evaluate_subset(root: Path, physics: PhysicsConfig | None = None) -> dict[str, Any]:
    questions = json.loads((root / "questions.json").read_text())
    desc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    opt: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_q: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: dict[str, int] = defaultdict(int)
    physics_seconds = 0.0
    for scene in questions:
        idx = int(scene["scene_index"])
        ann = json.loads((root / "annotations" / f"annotation_{idx:05d}.json").read_text())
        world = World.from_annotation(ann)
        t0 = time.perf_counter()
        world = simulate_world(world, physics or PhysicsConfig())
        physics_seconds += time.perf_counter() - t0
        for q in scene["questions"]:
            qtype = q["question_type"]
            ident = {"scene_index": idx, "question_id": q["question_id"]}
            if qtype == "descriptive":
                sub = q["question_subtype"]
                try:
                    got = answer_open(world, q["program"])
                except ExecutionError as exc:
                    got = f"error: {exc}"
                    errors[qtype] += 1
                ok = got == q["answer"]
                desc[sub][0] += ok
                desc[sub][1] += 1
                if not ok:
                    failures[f"descriptive/{sub}"].append(
                        {**ident, "expected": q["answer"], "got": got}
                    )
                continue
            all_ok = True
            for c in q["choices"]:
                try:
                    pred: bool | str = answer_choice(world, q["program"], c["program"])
                except ExecutionError as exc:
                    pred = f"error: {exc}"
                    errors[qtype] += 1
                ok = pred is (c["answer"] == "correct")
                opt[qtype][0] += ok
                opt[qtype][1] += 1
                all_ok &= ok
                if not ok:
                    failures[qtype].append(
                        {**ident, "choice_id": c["choice_id"], "expected": c["answer"], "got": pred}
                    )
            per_q[qtype][0] += all_ok
            per_q[qtype][1] += 1
    d_hit = sum(v[0] for v in desc.values())
    d_tot = sum(v[1] for v in desc.values())
    return {
        "descriptive": {
            "accuracy": _ratio(d_hit, d_tot),
            "count": d_tot,
            "by_subtype": {
                k: {"accuracy": _ratio(*v), "count": v[1]} for k, v in sorted(desc.items())
            },
        },
        **{
            t: {
                "per_option": _ratio(*opt[t]),
                "options": opt[t][1],
                "per_question": _ratio(*per_q[t]),
                "questions": per_q[t][1],
            }
            for t in ("explanatory", "predictive", "counterfactual")
        },
        "execution_errors": dict(errors),
        "failure_counts": {k: len(v) for k, v in sorted(failures.items())},
        "failure_samples": {k: v[:FAILURE_SAMPLE] for k, v in sorted(failures.items())},
        "scenes": len(questions),
        "physics_ms_per_video": 1000 * physics_seconds / max(1, len(questions)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subset", type=Path, help="dir with questions.json and annotations/")
    parser.add_argument("--out", type=Path, help="write JSON report here")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    table = evaluate_subset(args.subset)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    table["run"] = {
        "subset": str(args.subset),
        "code_commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seconds": time.perf_counter() - t0,
        "physics_config": PhysicsConfig().__dict__,
    }
    text = json.dumps(table, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
