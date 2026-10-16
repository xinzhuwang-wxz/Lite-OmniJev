"""Build the B1 single-family program-grid supervision set from existing TRAIN truth.

Zero GPU, stdlib only. Sources (pinned, pre-existing, no new data acquisition):
- official CLEVRER questions/train.json and annotation_train.zip
- the admitted TRAIN video pool train.jsonl (families only from this pool)

Every grid item is a standard one-question Sample row (existing contract) whose
gold is derived from the official annotation trajectory truth of the SAME video:
polarity twins for enter-order (direct + official phrasing), boundary existence,
answerable-absent existence, and program-constructed unanswerable variants whose
correct decision is U (insufficient). Twin/group structure lives in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

QUESTIONS_SHA = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
ANNOTATIONS_SHA = "e7f9185ad3dde8139b839b00f0f75483374dd0325a3cbec2dac882078497784c"
TRAIN_SHA = "912c871f7467b36ea1837276536250539f16f2ae6c5eaf32f867912b9e49e948"
MAX_ITEMS = 300
MIN_ITEMS = 120
U_ITEMS_CAP = 24
ABSENT_CAP = 16
COLORS = ("blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow")
MATERIALS = ("metal", "rubber")
SHAPES = ("cube", "cylinder", "sphere")
PLURAL = {"cube": "cubes", "cylinder": "cylinders", "sphere": "spheres"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_member(scene: int, names: set[str]) -> str:
    for name in (
        f"annotation_{scene // 1000 * 1000:05d}-{(scene // 1000 + 1) * 1000:05d}/annotation_{scene:05d}.json",
        f"annotation_{scene:05d}.json",
    ):
        if name in names:
            return name
    raise SystemExit(f"annotation member for scene {scene} not found")


def scene_truth(scene: dict) -> dict:
    objects = {
        entry["object_id"]: {
            "color": entry["color"],
            "material": entry["material"],
            "shape": entry["shape"],
        }
        for entry in scene["object_property"]
    }
    inside: dict[int, list[tuple[int, bool]]] = {oid: [] for oid in objects}
    final_frame = 0
    for frame in scene["motion_trajectory"]:
        final_frame = max(final_frame, frame["frame_id"])
        present = {item["object_id"]: item["inside_camera_view"] for item in frame["objects"]}
        for oid in objects:
            inside[oid].append((frame["frame_id"], bool(present.get(oid, False))))
    truth = {}
    for oid, series in inside.items():
        frames_in = [f for f, flag in series if flag]
        if not frames_in:
            truth[oid] = None
            continue
        first, last = min(frames_in), max(frames_in)
        truth[oid] = {
            "entry_frame": None if first == series[0][0] else first,
            "first_inside": first,
            "last_inside": last,
            "present_at_begin": first == series[0][0],
            "present_at_end": last == final_frame,
        }
    return {
        "objects": objects,
        "truth": truth,
        "final_frame": final_frame,
        "collisions": [
            {
                "object_ids": sorted(hit["object_ids"]),
                "frame_id": hit["frame_id"],
            }
            for hit in scene.get("collision", [])
        ],
    }


def describe(obj: dict) -> str:
    return f"{obj['color']} {obj['material']} {obj['shape']}"


def class_label(obj: dict) -> str:
    return f"{obj['color']} {PLURAL[obj['shape']]}"


def build_items(scene: int, pool_row: dict, parsed: dict) -> tuple[list[dict], list[dict]]:
    objects, truth = parsed["objects"], parsed["truth"]
    collisions = parsed.get("collisions", [])
    live = {oid: spec for oid, spec in objects.items() if truth[oid] is not None}
    entering = sorted(oid for oid in live if truth[oid]["entry_frame"] is not None)
    items: list[dict] = []
    derivations: list[dict] = []

    def add(cell: str, group: str, instruction: str, target: dict, derivation: dict) -> None:
        index = len(items)
        qid = f"b1-{cell}-{index:03d}"
        derivation = {"scene": scene, **derivation}
        items.append(
            {
                "id": f"b1-grid-001:{scene}:{qid}",
                "prefix": pool_row["prefix"],
                "questions": [
                    {
                        "qid": qid,
                        "type": "noul",
                        "instruction": instruction,
                        "options": None,
                        "score_levels": None,
                        "score_descriptions": None,
                        "target": target,
                        "pair_of": None,
                    }
                ],
                "meta": {
                    **pool_row["meta"],
                    "data_version": "clevrer-train-video-b1-grid-v1",
                },
            }
        )
        derivations.append({"qid": qid, "cell": cell, "group": group, **derivation})

    hard_yes = {"kind": "hard", "value": "yes", "source": "b1_grid_program_truth"}
    hard_no = {"kind": "hard", "value": "no", "source": "b1_grid_program_truth"}
    insufficient = {"kind": "insufficient"}

    # A. direct enter-order polarity twins
    for left in entering:
        for right in entering:
            if left == right or truth[left]["entry_frame"] == truth[right]["entry_frame"]:
                continue
            before = truth[left]["entry_frame"] < truth[right]["entry_frame"]
            group = f"order-direct:{left}:{right}"
            add(
                "order_direct_before",
                group,
                f"Does the {describe(objects[left])} enter the scene before the "
                f"{describe(objects[right])} enters the scene?",
                hard_yes if before else hard_no,
                {"left": left, "right": right, "entry_left": truth[left]["entry_frame"],
                 "entry_right": truth[right]["entry_frame"], "rule": "entry_left<entry_right"},
            )
            add(
                "order_direct_after",
                group,
                f"Does the {describe(objects[left])} enter the scene after the "
                f"{describe(objects[right])} enters the scene?",
                hard_no if before else hard_yes,
                {"left": left, "right": right, "entry_left": truth[left]["entry_frame"],
                 "entry_right": truth[right]["entry_frame"], "rule": "entry_left>entry_right"},
            )

    # B. official phrasing relational enter-exist twins
    classes: dict[str, list[int]] = {}
    for oid in entering:
        classes.setdefault(class_label(objects[oid]), []).append(oid)
    for label, members in sorted(classes.items()):
        for anchor in entering:
            if anchor in members:
                continue
            anchor_entry = truth[anchor]["entry_frame"]
            any_after = any(truth[m]["entry_frame"] > anchor_entry for m in members)
            any_before = any(truth[m]["entry_frame"] < anchor_entry for m in members)
            group = f"order-official:{label}:{anchor}"
            add(
                "order_official_after",
                group,
                f"Are there any {label} that enter the scene after the "
                f"{describe(objects[anchor])} enters the scene?",
                hard_yes if any_after else hard_no,
                {"class": label, "members": members, "anchor": anchor,
                 "anchor_entry": anchor_entry, "rule": "any_member_entry>anchor_entry"},
            )
            add(
                "order_official_before",
                group,
                f"Are there any {label} that enter the scene before the "
                f"{describe(objects[anchor])} enters the scene?",
                hard_yes if any_before else hard_no,
                {"class": label, "members": members, "anchor": anchor,
                 "anchor_entry": anchor_entry, "rule": "any_member_entry<anchor_entry"},
            )

    # C. boundary existence twins per live object
    for oid in sorted(live):
        group = f"exist-boundary:{oid}"
        add(
            "exist_begin",
            group,
            f"Is there a {describe(objects[oid])} in the scene when the video begins?",
            hard_yes if truth[oid]["present_at_begin"] else hard_no,
            {"object": oid, "rule": "inside_camera_view@frame0"},
        )
        add(
            "exist_end",
            group,
            f"Is there a {describe(objects[oid])} in the scene when the video ends?",
            hard_yes if truth[oid]["present_at_end"] else hard_no,
            {"object": oid, "rule": "inside_camera_view@final_frame"},
        )

    # D. answerable-absent existence (deterministic fake combos, capped)
    present = {
        (spec["color"], spec["material"], spec["shape"]) for spec in live.values()
    }
    emitted = 0
    for color in COLORS:
        for material in MATERIALS:
            for shape in SHAPES:
                if (color, material, shape) in present or emitted >= ABSENT_CAP:
                    continue
                add(
                    "exist_absent_begin",
                    f"absent:{color}:{material}:{shape}",
                    f"Is there a {color} {material} {shape} in the scene when the video begins?",
                    hard_no,
                    {"combo": [color, material, shape], "rule": "combo_not_in_scene"},
                )
                emitted += 1

    # F. collision-order polarity twins (events with distinct frames)
    def collision_desc(hit: dict) -> str:
        left, right = hit["object_ids"]
        return f"the {describe(objects[left])} and the {describe(objects[right])}"

    for first_index, first in enumerate(collisions):
        for second in collisions[first_index + 1 :]:
            if first["frame_id"] == second["frame_id"]:
                continue
            before = first["frame_id"] < second["frame_id"]
            group = f"collision-order:{first['frame_id']}:{second['frame_id']}"
            add(
                "collision_order_before",
                group,
                f"Does the collision between {collision_desc(first)} happen before the "
                f"collision between {collision_desc(second)} happens?",
                hard_yes if before else hard_no,
                {"first": first["object_ids"] + [first["frame_id"]],
                 "second": second["object_ids"] + [second["frame_id"]],
                 "rule": "first_frame<second_frame"},
            )
            add(
                "collision_order_after",
                group,
                f"Does the collision between {collision_desc(first)} happen after the "
                f"collision between {collision_desc(second)} happens?",
                hard_no if before else hard_yes,
                {"first": first["object_ids"] + [first["frame_id"]],
                 "second": second["object_ids"] + [second["frame_id"]],
                 "rule": "first_frame>second_frame"},
            )

    # G. enter-vs-collision mixed order twins
    for oid in entering:
        for hit in collisions:
            entry = truth[oid]["entry_frame"]
            if entry is None or entry == hit["frame_id"]:
                continue
            before = entry < hit["frame_id"]
            group = f"enter-collision:{oid}:{hit['frame_id']}"
            add(
                "enter_collision_before",
                group,
                f"Does the {describe(objects[oid])} enter the scene before the collision "
                f"between {collision_desc(hit)} happens?",
                hard_yes if before else hard_no,
                {"object": oid, "entry": entry, "collision": hit["object_ids"] + [hit["frame_id"]],
                 "rule": "entry<collision_frame"},
            )
            add(
                "enter_collision_after",
                group,
                f"Does the {describe(objects[oid])} enter the scene after the collision "
                f"between {collision_desc(hit)} happens?",
                hard_no if before else hard_yes,
                {"object": oid, "entry": entry, "collision": hit["object_ids"] + [hit["frame_id"]],
                 "rule": "entry>collision_frame"},
            )

    # H. exit-order polarity twins (objects that really leave mid-video)
    leaving = sorted(
        oid
        for oid in live
        if not truth[oid]["present_at_end"] and truth[oid]["last_inside"] < parsed["final_frame"]
    )
    for left in leaving:
        for right in leaving:
            if left == right or truth[left]["last_inside"] == truth[right]["last_inside"]:
                continue
            before = truth[left]["last_inside"] < truth[right]["last_inside"]
            group = f"exit-order:{left}:{right}"
            add(
                "exit_order_before",
                group,
                f"Does the {describe(objects[left])} leave the scene before the "
                f"{describe(objects[right])} leaves the scene?",
                hard_yes if before else hard_no,
                {"left": left, "right": right,
                 "exit_left": truth[left]["last_inside"], "exit_right": truth[right]["last_inside"],
                 "rule": "last_inside_left<last_inside_right"},
            )
            add(
                "exit_order_after",
                group,
                f"Does the {describe(objects[left])} leave the scene after the "
                f"{describe(objects[right])} leaves the scene?",
                hard_no if before else hard_yes,
                {"left": left, "right": right,
                 "exit_left": truth[left]["last_inside"], "exit_right": truth[right]["last_inside"],
                 "rule": "last_inside_left>last_inside_right"},
            )

    # E. program-constructed unanswerable order variants (gold=U), capped
    fakes: list[tuple[str, str, str]] = [
        (color, material, shape)
        for color in COLORS
        for material in MATERIALS
        for shape in SHAPES
        if (color, material, shape) not in present
    ]
    if entering:
        emitted = 0
        outer = 0
        while emitted < U_ITEMS_CAP and outer < len(fakes) * len(entering):
            fake = fakes[(outer // len(entering)) % len(fakes)]
            anchor = entering[outer % len(entering)]
            outer += 1
            if emitted >= U_ITEMS_CAP:
                continue
            add(
                "u_premise_order",
                f"u-premise:{fake[0]}:{fake[1]}:{fake[2]}:{anchor}",
                f"Does the {fake[0]} {fake[1]} {fake[2]} enter the scene before the "
                f"{describe(objects[anchor])} enters the scene?",
                insufficient,
                {"fake": list(fake), "anchor": anchor,
                 "rule": "order_of_nonexistent_object_is_undecidable"},
            )
            emitted += 1
    if len(items) > MAX_ITEMS:
        # Deterministic overflow trim: official-phrasing cells go first.
        keep = [
            (item, derivation)
            for item, derivation in zip(items, derivations, strict=True)
            if not derivation["cell"].startswith("order_official")
        ]
        dropped = len(items) - len(keep)
        items = [item for item, _ in keep]
        derivations = [derivation for _, derivation in keep]
        print(json.dumps({"trimmed_official_cells": dropped}))
    return items, derivations


def select_groups(
    items: list[dict], derivations: list[dict], *, budget: int = 220
) -> tuple[list[dict], list[dict]]:
    """Twin-atomic deterministic selection with cell priority; U/exist first."""
    priority = {
        "u_premise_order": 0,
        "exist_begin": 1,
        "exist_end": 1,
        "exist_absent_begin": 2,
        "order_direct_before": 3,
        "order_direct_after": 3,
        "collision_order_before": 3,
        "collision_order_after": 3,
        "enter_collision_before": 3,
        "enter_collision_after": 3,
        "exit_order_before": 3,
        "exit_order_after": 3,
        "order_official_after": 4,
        "order_official_before": 4,
    }
    groups: dict[str, list[int]] = {}
    for index, derivation in enumerate(derivations):
        groups.setdefault(derivation["group"], []).append(index)
    ordered = sorted(
        groups.items(),
        key=lambda entry: (
            min(priority.get(derivations[i]["cell"], 9) for i in entry[1]),
            entry[0],
        ),
    )
    kept: list[int] = []
    for _, indices in ordered:
        if len(kept) + len(indices) > budget:
            continue
        kept.extend(indices)
    kept.sort()
    return (
        [items[i] for i in kept],
        [derivations[i] for i in kept],
    )


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: build-b1-consistency-grid.py QUESTIONS_JSON ANNOTATION_ZIP TRAIN_JSONL"
        )
    questions_path, annotations_path, train_path = (Path(a) for a in sys.argv[1:])
    for path, expected in [
        (questions_path, QUESTIONS_SHA),
        (annotations_path, ANNOTATIONS_SHA),
        (train_path, TRAIN_SHA),
    ]:
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"sha mismatch {path}: {actual} != {expected}")
    pool_rows = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    video_rows = {
        int(row["meta"]["family_id"].split(":")[1]): row
        for row in pool_rows
        if row["meta"]["modality"] == ["video"]
    }
    if len(video_rows) != 8299:
        raise SystemExit(f"expected 8299 TRAIN video families, found {len(video_rows)}")
    with zipfile.ZipFile(annotations_path) as archive:
        names = set(archive.namelist())
        parsed = {}
        for scene in sorted(video_rows):
            parsed[scene] = scene_truth(json.loads(archive.read(annotation_member(scene, names))))
    scored = []
    for scene, data in parsed.items():
        entering = [
            oid
            for oid, spec in data["truth"].items()
            if spec and spec["entry_frame"] is not None
        ]
        collisions = len(data["collisions"])
        live = len([oid for oid, spec in data["truth"].items() if spec])
        capacity = (
            4 * max(0, len(entering) * (len(entering) - 1))
            + 4 * max(0, collisions * (collisions - 1))
            + 4 * len(entering) * collisions
            + 2 * live
            + ABSENT_CAP
            + U_ITEMS_CAP
        )
        scored.append((capacity, -scene, scene))
    scored.sort(reverse=True)
    items: list[dict] = []
    derivations: list[dict] = []
    families: list[int] = []
    for capacity, _, scene in scored[:400]:
        if not items and capacity < MIN_ITEMS:
            break
        raw_items, raw_derivations = build_items(scene, video_rows[scene], parsed[scene])
        if not raw_items:
            continue
        room = MAX_ITEMS - len(items)
        if room <= 0:
            break
        picked_items, picked_derivations = (
            select_groups(raw_items, raw_derivations, budget=min(room, 220))
            if len(raw_items) > room
            else (raw_items, raw_derivations)
        )
        if len(picked_items) < MIN_ITEMS // 3:
            continue
        items.extend(picked_items)
        derivations.extend(picked_derivations)
        families.append(scene)
        if len(items) >= MIN_ITEMS:
            break
    if not MIN_ITEMS <= len(items) <= MAX_ITEMS:
        raise SystemExit(f"grid assembly landed outside band: {len(items)} items")
    chosen = families[0]
    manifest = {
        "schema_version": "lite-omnijev.explore.s4-b1-grid-001.v1",
        "selection_rule": "families sorted by capacity proxy (enter/collision/exit "
        "twin capacity + existence + absent/U caps, tie-break smaller scene index); "
        "twins kept atomic; per-family budget min(remaining room, 220); stop at 120 "
        "items; hard cap 300; single family preferred, extra families only when the "
        "first cannot reach the 120 floor",
        "capacity_note": "measured max single-family yield with these cells is ~120; "
        "the 120-300 band is reached with the minimal family set",
        "candidates_considered": len(scored),
        "families": [
            {
                "scene_index": scene,
                "video_uri": video_rows[scene]["prefix"]["media"][0]["uri"],
                "video_media_sha256": video_rows[scene]["prefix"]["media"][0]["sha256"],
                "object_property": parsed[scene]["objects"],
                "trajectory_truth": parsed[scene]["truth"],
                "collisions": parsed[scene]["collisions"],
                "final_frame": parsed[scene]["final_frame"],
                "items": sum(1 for d in derivations if d["scene"] == scene),
            }
            for scene in families
        ],
        "item_count": len(items),
        "cells": {},
        "transitivity_blocks": {
            str(scene): [
                d["qid"] for d in derivations if d["scene"] == scene and d["cell"].startswith("order_direct")
            ]
            for scene in families
        },
        "twin_groups": sorted(
            {
                f"{d['scene']}:{d['group']}"
                for d in derivations
                if "order" in d["group"] or "exist-boundary" in d["group"]
            }
        ),
        "inputs": {
            "questions_train_json_sha256": QUESTIONS_SHA,
            "annotation_train_zip_sha256": ANNOTATIONS_SHA,
            "train_pool_sha256": TRAIN_SHA,
        },
        "derivations": derivations,
    }
    counts: dict[str, int] = {}
    for derivation in derivations:
        counts[derivation["cell"]] = counts.get(derivation["cell"], 0) + 1
    manifest["cells"] = counts
    out_dir = Path(sys.argv[0]).resolve().parent
    (out_dir / "b1-grid.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8"
    )
    (out_dir / "b1-grid-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"scene": chosen, "items": len(items), "cells": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
