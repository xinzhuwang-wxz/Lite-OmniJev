"""Read-only cached GUI capacity probe. JSON config path is the sole argument.

Run from a repo root with PYTHONPATH=. . No pools or candidate files are written.
All stdout is fixed-schema aggregate JSON; exceptions never expose source text.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import platform
import sys
import time
from pathlib import Path

from data.schema import Sample
from scripts import s0_gui_dev as gui
from scripts.gold_inventory import (
    Artifact,
    ComponentRecord,
    Manifest,
    audit_manifest,
    bound_lines,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bound(spec: dict) -> bytes:
    artifact = Artifact.model_validate(spec)
    path = Path(artifact.path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("input")
    content = path.read_bytes()
    if sha(content) != artifact.sha256:
        raise ValueError("digest")
    return content


def main(config: dict) -> dict:
    manifest_bytes = bound(config["inventory"])
    receipt = audit_manifest(Path(config["inventory"]["path"]))
    if receipt.outcome != "isolated":
        raise ValueError("inventory")
    manifest = Manifest.model_validate_json(manifest_bytes)
    families, media = set(), set()
    for artifact in manifest.samples:
        for line in bound_lines(artifact):
            sample = Sample.model_validate_json(line)
            families.add(gui.sha256_bytes(sample.meta.family_id.encode()))
            families.add(sample.meta.family_id)
            media.update(m.sha256 for m in sample.prefix.media)
    for artifact in manifest.component_ledgers:
        for line in bound_lines(artifact):
            record = ComponentRecord.model_validate_json(line)
            for component in (record.image, record.audio):
                families.update((component.family_id, component.family_hash))
                media.add(component.media_sha256)
    quarantine_bytes = bound(config["quarantine"])
    quarantine = json.loads(quarantine_bytes)
    if set(quarantine) != {"families", "media"}:
        raise ValueError("quarantine")
    for key in ("families", "media"):
        if not isinstance(quarantine[key], list) or any(
            not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in quarantine[key]
        ):
            raise ValueError("quarantine")
    families.update(quarantine["families"])
    media.update(quarantine["media"])
    plan_bytes = bound(config["plan"])
    plan = json.loads(plan_bytes)
    if plan.get("partition") != "train":
        raise ValueError("plan")
    episodes = plan["selected_episodes"]
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("plan")
    annotation_root = Path(config["annotation_root"])
    media_root = Path(config["media_root"])
    if not annotation_root.is_absolute() or not media_root.is_absolute():
        raise ValueError("root")
    counts = dict.fromkeys(
        (
            "cached_plan_episodes",
            "excluded_assigned_or_quarantined_episodes",
            "missing_annotations",
            "invalid_annotations",
            "scanned_episodes",
            "original_steps",
            "original_click_steps",
            "missing_media_steps",
            "invalid_media_steps",
            "invalid_geometry_steps",
            "empty_instruction_steps",
            "insufficient_negative_steps",
            "eligible_original_steps_before_dedup",
            "duplicate_media_steps",
            "duplicate_question_steps",
            "history_media_colliding_episodes",
            "eligible_originals",
            "eligible_families",
            "eligible_connected_components",
        ),
        0,
    )
    annotation_hashes = []
    observed_media_hashes = []
    candidates = []
    seen_episodes = set()
    colliding_families = set()
    for entry in episodes:
        episode = gui.safe_episode_id(entry["episode_id"])
        if episode in seen_episodes or entry.get("partition") != "train":
            raise ValueError("plan")
        seen_episodes.add(episode)
        counts["cached_plan_episodes"] += 1
        family = f"gui:{episode}"
        if family in families or sha(family.encode()) in families:
            counts["excluded_assigned_or_quarantined_episodes"] += 1
            continue
        annotation_path = annotation_root / f"{episode}.json"
        if not annotation_path.is_file():
            counts["missing_annotations"] += 1
            continue
        if annotation_path.is_symlink():
            raise ValueError("annotation")
        raw_annotation = annotation_path.read_bytes()
        annotation_hashes.append(sha(raw_annotation))
        annotation = json.loads(raw_annotation)
        steps = annotation.get("steps")
        if (
            str(annotation.get("episode_id")) != episode
            or not isinstance(steps, list)
            or annotation.get("step_length") != len(steps)
            or any(not isinstance(s, dict) or type(s.get("step")) is not int for s in steps)
            or sorted(s["step"] for s in steps) != list(range(len(steps)))
            or any(s.get("screenshot") != f"{episode}_{s['step']}.png" for s in steps)
        ):
            counts["invalid_annotations"] += 1
            continue
        counts["scanned_episodes"] += 1
        for step in steps:
            counts["original_steps"] += 1
            if step.get("action") != "CLICK":
                continue
            counts["original_click_steps"] += 1
            instruction = str(step.get("low_level_instruction", "")).strip()
            if not instruction:
                counts["empty_instruction_steps"] += 1
                continue
            path = media_root / "screenshots" / step["screenshot"]
            if not path.is_file():
                counts["missing_media_steps"] += 1
                continue
            try:
                info = gui.image_info(path)
            except (ValueError, OSError):
                counts["invalid_media_steps"] += 1
                continue
            observed_media_hashes.append(info["sha256"])
            if info["sha256"] in media:
                colliding_families.add(family)
                continue
            try:
                box = gui.parse_box(step.get("sam2_bbox"))
                point = gui.parse_click_point(step.get("info"))
            except ValueError:
                counts["invalid_geometry_steps"] += 1
                continue
            if not gui.box_is_legal(
                box, width=info["width"], height=info["height"]
            ) or not gui.contains(box, point):
                counts["invalid_geometry_steps"] += 1
                continue
            negatives = gui.candidate_negative_boxes(
                box, point, width=info["width"], height=info["height"]
            )
            if len(negatives) < 3:
                counts["insufficient_negative_steps"] += 1
                continue
            counts["eligible_original_steps_before_dedup"] += 1
            rgb = info["decoded_rgb_sha256"]
            # No instruction/options/coordinates leave this process.
            question = sha(
                json.dumps([rgb, " ".join(instruction.split()), box], sort_keys=True).encode()
            )
            candidates.append((family, info["sha256"], rgb, question))
    counts["history_media_colliding_episodes"] = len(colliding_families)
    # Link candidate episodes BEFORE dropping duplicate images; otherwise a
    # duplicate screenshot could hide a connection between eventual pools.
    parent = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    for family, raw, rgb, _question in candidates:
        union("f:" + family, "m:" + raw)
        union("f:" + family, "r:" + rgb)
    blocked_components = {find("f:" + f) for f in colliding_families}
    seen_media, seen_questions, accepted_families = set(), set(), set()
    component_sizes = {}
    for family, raw, rgb, question in candidates:
        component = find("f:" + family)
        if component in blocked_components:
            continue
        if question in seen_questions:
            counts["duplicate_question_steps"] += 1
            continue
        seen_questions.add(question)
        if rgb in seen_media or raw in seen_media:
            counts["duplicate_media_steps"] += 1
            continue
        seen_media.update((rgb, raw))
        accepted_families.add(family)
        counts["eligible_originals"] += 1
        component_sizes[component] = component_sizes.get(component, 0) + 1
    counts["eligible_families"] = len(accepted_families)
    counts["eligible_connected_components"] = len(component_sizes)
    # Feasibility witness only, not an assignment: whole components go either
    # to holdout or calibration; excess originals within a pool need not be used.
    need_cal, need_hold = 2881, 941
    total = counts["eligible_originals"]
    reachable = 1
    for size in component_sizes.values():
        reachable |= reachable << size
    feasible = total >= need_cal + need_hold and bool(
        (reachable >> need_hold) & ((1 << (total - need_cal - need_hold + 1)) - 1)
    )
    return {
        "counts": counts,
        "full_shortfall_component_feasible": feasible,
        "shortfall_calibration": need_cal,
        "shortfall_holdout": need_hold,
        "annotation_inventory_sha256": sha("\n".join(sorted(annotation_hashes)).encode()),
        "scanned_media_inventory_sha256": sha("\n".join(sorted(observed_media_hashes)).encode()),
        "input_sha256": [config[k]["sha256"] for k in ("inventory", "quarantine", "plan")],
        "historical_decoded_rgb_aliases": "not_available_from_inventory",
        "counting": "one_original_step_per_unique_media_conservative",
    }


if __name__ == "__main__":
    start = time.monotonic()
    result = {
        "schema_version": "gui-capacity-private-v1",
        "status": "rejected",
        "failure": "job_failed",
        "gpu_used": False,
        "network_bytes": 0,
        "pool_assignment_written": False,
        "gold_admission": "not_validated",
        "permissions": "not_validated",
        "python": platform.python_version(),
        "script_sha256": sha(Path(__file__).read_bytes()),
        "exit_code": 1,
    }
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            config_bytes = Path(sys.argv[1]).read_bytes()
            result["config_sha256"] = sha(config_bytes)
            result.update(main(json.loads(config_bytes)))
            result.update(status="measured", failure=None, exit_code=0)
    except Exception:
        pass
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(result["exit_code"])
