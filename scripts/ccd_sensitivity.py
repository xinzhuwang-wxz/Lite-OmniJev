"""Perception sensitivity of the CCD executor + physics on CLEVRER-style subsets.

Starts from ground-truth annotations, synthesizes what a perception model would report
(per sampled frame, per object: in_view and a floor position) under degradations, rebuilds
worlds with ``eval.ccd_perceive.world_from_observations`` and scores every question with
the same executor + physics as ``scripts/ccd_oracle_eval.py``.

Grid: frames x position (exact | gaussian sigma | k x k floor-cell centre) x presence flip
rate. Reference rows: ground-truth velocities+collisions ("gt") and perceive from exact
observations at every frame ("obs-128-exact", the perceive module's own loss).
Deterministic: per (condition, scene) RNG seeded from ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from eval.ccd_executor import ExecutionError, answer_choice, answer_open
from eval.ccd_perceive import Observation, PerceiveConfig, world_from_observations
from eval.ccd_physics import PhysicsConfig, simulate_world
from eval.ccd_world import OBSERVED_FRAMES, SceneObject, World

# Floor box covering in-view object centres (DEV 1st..99th percentile is x -5.4..2.0,
# y -2.8..2.9; extremes x -6.6..2.6, y -5.0..5.0). Binned positions are cell centres.
FLOOR = (-6.75, 2.75, -5.0, 5.0)
FRAMES = (16, 32, 64, 128)
POSITIONS = ("exact", "noise0.05", "noise0.1", "noise0.2", "bin6", "bin8", "bin12")
FLIPS = (0.0, 0.01, 0.03)
KEY_ROWS = ("gt", "obs-128-exact", "exact-32-flip0", "noise0.1-32-flip0")
TYPES = ("explanatory", "predictive", "counterfactual")


@dataclass(frozen=True)
class Condition:
    name: str
    frames: int = OBSERVED_FRAMES
    position: str = "exact"
    flip: float = 0.0
    ground_truth: bool = False


def grid() -> list[Condition]:
    rows = [Condition("gt", ground_truth=True), Condition("obs-128-exact")]
    for frames in FRAMES:
        for pos in POSITIONS:
            for flip in FLIPS:
                rows.append(Condition(f"{pos}-{frames}-flip{flip:g}", frames, pos, flip))
    return rows


def sample_frames(n: int, horizon: int = OBSERVED_FRAMES) -> list[int]:
    """``n`` evenly spaced frames including the first and the last."""
    if n >= horizon:
        return list(range(horizon))
    return sorted({round(i * (horizon - 1) / (n - 1)) for i in range(n)})


def _degrade(x: float, y: float, position: str, rng: random.Random) -> tuple[float, float]:
    if position == "exact":
        return x, y
    if position.startswith("noise"):
        sigma = float(position[len("noise") :])
        return x + rng.gauss(0.0, sigma), y + rng.gauss(0.0, sigma)
    if position.startswith("bin"):
        k = int(position[len("bin") :])
        x0, x1, y0, y1 = FLOOR
        cx = min(k - 1, max(0, int((x - x0) / (x1 - x0) * k)))
        cy = min(k - 1, max(0, int((y - y0) / (y1 - y0) * k)))
        return x0 + (cx + 0.5) * (x1 - x0) / k, y0 + (cy + 0.5) * (y1 - y0) / k
    raise ValueError(f"unknown position mode {position}")


def observe(
    annotation: dict[str, Any], cond: Condition, rng: random.Random
) -> tuple[dict[int, dict[int, Observation]], list[int]]:
    """Synthetic perception: presence (with flips) and degraded position per sampled frame."""
    frames = sample_frames(cond.frames)
    keep = set(frames)
    obs: dict[int, dict[int, Observation]] = {}
    for row in annotation["motion_trajectory"]:
        f = int(row["frame_id"])
        if f not in keep:
            continue
        per: dict[int, Observation] = {}
        for o in row["objects"]:
            vis = bool(o["inside_camera_view"])
            if cond.flip and rng.random() < cond.flip:
                vis = not vis
            if vis:
                x, y = _degrade(
                    float(o["location"][0]), float(o["location"][1]), cond.position, rng
                )
                per[int(o["object_id"])] = (True, x, y)
            else:
                per[int(o["object_id"])] = (False, None, None)
        obs[f] = per
    return obs, frames


def perceive_config(cond: Condition) -> PerceiveConfig:
    """Perceive settings for a degradation (tuned on DEV only)."""
    spacing = (OBSERVED_FRAMES - 1) / max(1, cond.frames - 1)
    cfg = PerceiveConfig(presence_window=(1 if cond.frames <= 32 else 2) if cond.flip else 0)
    if cond.position.startswith("noise"):
        # Motion is near-linear between collisions: long fits beat noise; the still-speed
        # snap grows with the per-sample noise relative to sample density.
        sigma = float(cond.position[len("noise") :])
        return replace(
            cfg,
            fit_window=30.0,
            fit_min_samples=4,
            side_window=16.0,
            side_min_samples=4,
            impulse_min=0.3,
            contact_margin=max(0.25, 2 * sigma),
            still_speed=0.7 if sigma * spacing**0.5 >= 0.2 else 0.4,
        )
    if cond.position.startswith("bin"):
        cell = (FLOOR[1] - FLOOR[0]) / int(cond.position[len("bin") :])
        return replace(
            cfg,
            fit_window=45.0,
            fit_min_samples=4,
            side_window=36.0,
            side_min_samples=4,
            impulse_min=0.3,
            contact_margin=cell,
            align_min=0.0,
            still_speed=0.5,
        )
    return cfg


def _objects(annotation: dict[str, Any]) -> list[SceneObject]:
    return [
        SceneObject(
            object_id=int(p["object_id"]),
            color=p["color"],
            material=p["material"],
            shape=p["shape"],
        )
        for p in annotation["object_property"]
    ]


def build_world(annotation: dict[str, Any], cond: Condition, seed: int) -> World:
    if cond.ground_truth:
        return World.from_annotation(annotation)
    rng = random.Random(f"{seed}:{cond.name}:{annotation['scene_index']}")
    obs, frames = observe(annotation, cond, rng)
    return world_from_observations(_objects(annotation), obs, frames, perceive_config(cond))


def _ratio(hit: int, total: int) -> float | None:
    return hit / total if total else None


def evaluate(root: Path, cond: Condition, seed: int, limit: int | None) -> dict[str, Any]:
    questions = json.loads((root / "questions.json").read_text())[:limit]
    desc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    opt: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_q: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    errors: dict[str, int] = defaultdict(int)
    t0 = time.perf_counter()
    for scene in questions:
        idx = int(scene["scene_index"])
        ann = json.loads((root / "annotations" / f"annotation_{idx:05d}.json").read_text())
        world = simulate_world(build_world(ann, cond, seed), PhysicsConfig())
        for q in scene["questions"]:
            qtype = q["question_type"]
            if qtype == "descriptive":
                try:
                    got = answer_open(world, q["program"])
                except ExecutionError:
                    got, errors[qtype] = "error", errors[qtype] + 1
                cell = desc[q["question_subtype"]]
                cell[0] += got == q["answer"]
                cell[1] += 1
                continue
            all_ok = True
            for c in q["choices"]:
                try:
                    pred: bool | str = answer_choice(world, q["program"], c["program"])
                except ExecutionError:
                    pred, errors[qtype] = "error", errors[qtype] + 1
                ok = pred is (c["answer"] == "correct")
                opt[qtype][0] += ok
                opt[qtype][1] += 1
                all_ok &= ok
            per_q[qtype][0] += all_ok
            per_q[qtype][1] += 1
    return {
        "condition": asdict(cond),
        "scenes": len(questions),
        "descriptive": {
            "accuracy": _ratio(sum(v[0] for v in desc.values()), sum(v[1] for v in desc.values())),
            "count": sum(v[1] for v in desc.values()),
            "by_subtype": {k: _ratio(*v) for k, v in sorted(desc.items())},
        },
        **{
            t: {
                "per_option": _ratio(*opt[t]),
                "options": opt[t][1],
                "per_question": _ratio(*per_q[t]),
                "questions": per_q[t][1],
            }
            for t in TYPES
        },
        "execution_errors": dict(errors),
        "seconds": time.perf_counter() - t0,
    }


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f}"


def markdown(rows: list[dict[str, Any]]) -> str:
    subtypes = sorted({k for r in rows for k in r["descriptive"]["by_subtype"]})
    head = ["condition", "desc", *subtypes, *(f"{t[:4]} opt/q" for t in TYPES)]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        d = r["descriptive"]
        cells = [r["condition"]["name"], _pct(d["accuracy"])]
        cells += [_pct(d["by_subtype"].get(s)) for s in subtypes]
        cells += [f"{_pct(r[t]['per_option'])}/{_pct(r[t]['per_question'])}" for t in TYPES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _run(args: tuple[Path, Condition, int, int | None]) -> dict[str, Any]:
    return evaluate(*args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subset", type=Path, help="dir with questions.json and annotations/")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path (.md beside)")
    parser.add_argument("--limit", type=int, help="first N scenes only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", help="comma-separated condition names, or key (default: grid)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    conds = grid()
    if args.rows:
        wanted = list(KEY_ROWS) if args.rows == "key" else args.rows.split(",")
        by_name = {c.name: c for c in conds}
        unknown = [w for w in wanted if w not in by_name]
        if unknown:
            parser.error(f"unknown rows: {unknown}")
        conds = [by_name[w] for w in wanted]
    t0 = time.perf_counter()
    jobs = [(args.subset, c, args.seed, args.limit) for c in conds]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(_run, jobs))
    else:
        rows = [_run(j) for j in jobs]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    report = {
        "rows": rows,
        "run": {
            "subset": str(args.subset),
            "limit": args.limit,
            "seed": args.seed,
            "code_commit": commit,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "seconds": time.perf_counter() - t0,
            "floor_box": FLOOR,
            "default_perceive_config": asdict(PerceiveConfig()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    table = markdown(rows)
    args.out.with_suffix(".md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
