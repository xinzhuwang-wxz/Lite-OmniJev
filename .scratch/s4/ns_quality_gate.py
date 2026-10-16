"""Quality gate for VLM scene extraction + per-predicate F1 diagnostics.

Gate: valid>=80%, object recall>=70% (color+shape key, material tolerated),
cascade measured vs D3P-600's 123/150. Diagnostics: per-predicate F1
(color/shape/material presence sets, enter/exit/moving object sets, collision
pair sets) — the low-F1 families are the RL scene-formalizer training targets.
VLM 5-phase words map onto the 8-grid (start->0, early->1, middle->3, late->5,
end->7); this mapping affects only the VLM arm, never the binding oracle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ns_cascade import END_INDEX, Scene, execute  # noqa: E402

VLM_PHASE_MAP = {"start": 0, "early": 1, "middle": 3, "late": 5, "end": 7}
KEY_IGNORE = {"metal", "rubber", "object"}
PREDICATES = ("colors", "shapes", "materials", "enterers", "exiters", "moving", "collisions")


def object_key(name: str) -> frozenset:
    return frozenset(w for w in name.split() if w not in KEY_IGNORE)


def mapped_scene(summary: dict) -> dict:
    mapped = json.loads(json.dumps(summary))
    for obj in mapped.get("objects", []):
        obj["first_seen"] = [
            k for k, v in VLM_PHASE_MAP.items() if v == VLM_PHASE_MAP.get(obj.get("first_seen"))
        ] and obj["first_seen"]
        obj["first_seen"] = obj["first_seen"]
    # encode mapped numeric phase via a trick: temporarily set to word whose
    # PHASES value matches the 8-grid position of the 5-grid word
    word_by_value = {0: "start", 1: "early", 2: "quarter", 3: "middle", 4: "three_quarters", 5: "late", 6: "very_late", 7: "end"}
    for obj in mapped.get("objects", []):
        value = VLM_PHASE_MAP.get(obj.get("first_seen"))
        if value is not None:
            obj["first_seen"] = word_by_value[value]
        value = VLM_PHASE_MAP.get(obj.get("last_seen"))
        if value is not None:
            obj["last_seen"] = word_by_value[value]
    return mapped


def prf_counts() -> dict:
    return {"tp": 0, "fp": 0, "fn": 0}


def prf_from(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def sets_from(summary: dict | None) -> dict | None:
    if not summary:
        return None
    objects = summary.get("objects", [])
    colors, shapes, materials, enterers, exiters, moving = set(), set(), set(), set(), set(), set()
    for obj in objects:
        words = set(obj["name"].split())
        key = object_key(obj["name"])
        colors |= words & {"yellow", "red", "blue", "green", "gray", "purple", "brown", "cyan"}
        shapes |= words & {"sphere", "cube", "cylinder"}
        materials |= words & {"metal", "rubber"}
        first = VLM_PHASE_MAP.get(obj.get("first_seen"), obj.get("first_seen"))
        last = VLM_PHASE_MAP.get(obj.get("last_seen"), obj.get("last_seen"))
        if isinstance(first, int) and first > 0:
            enterers.add(key)
        if isinstance(last, int) and last < END_INDEX:
            exiters.add(key)
        if obj.get("moving"):
            moving.add(key)
    collisions = {
        frozenset(object_key(name) for name in hit.get("between", []))
        for hit in summary.get("collisions", [])
        if len(hit.get("between", [])) == 2
    }
    return {
        "colors": colors,
        "shapes": shapes,
        "materials": materials,
        "enterers": enterers,
        "exiters": exiters,
        "moving": moving,
        "collisions": collisions,
    }


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: ns_quality_gate.py QUESTIONS VLM_SCENES TRUTH_SCENES D3P_RECORDS")
    questions_path, vlm_path, truth_path, d3p_path = (Path(a) for a in sys.argv[1:])
    questions = [json.loads(l) for l in questions_path.read_text().splitlines() if l.strip()]
    vlm = {json.loads(l)["id"]: json.loads(l) for l in vlm_path.read_text().splitlines() if l.strip()}
    truth = {json.loads(l)["id"]: json.loads(l) for l in truth_path.read_text().splitlines() if l.strip()}
    d3p = {}
    for line in d3p_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("pass", 0) == 0:
                d3p[row["id"]] = row
    total = valid = 0
    matched_truth = matched_vlm = truth_object_total = 0
    buckets = {k: prf_counts() for k in PREDICATES}
    cascade_correct = d3p_correct = resolved = executor_correct = 0
    for question in questions:
        gold = question["target"]["value"]
        direct = d3p.get(question["id"], {})
        d3p_correct += int(direct.get("prediction") == gold)
        row = vlm.get(question["id"])
        answer = None
        if row and row.get("summary_valid") and row.get("summary"):
            total += 1
            valid += 1
            summary = row["summary"]
            truth_summary = truth.get(question["id"], {}).get("summary")
            v_sets = sets_from(summary)
            t_sets = sets_from(truth_summary) if truth_summary else None
            if t_sets:
                # per-video micro counts (never set-union across videos)
                for key in PREDICATES:
                    overlap = v_sets[key] & t_sets[key]
                    buckets[key]["tp"] += len(overlap)
                    buckets[key]["fp"] += len(v_sets[key] - t_sets[key])
                    buckets[key]["fn"] += len(t_sets[key] - v_sets[key])
                v_keys = {object_key(o["name"]) for o in summary.get("objects", [])}
                t_keys = {object_key(o["name"]) for o in truth_summary.get("objects", [])}
                matched_truth += len(v_keys & t_keys)
                matched_vlm += len(v_keys)
                truth_object_total += len(t_keys)
            scene = Scene(mapped_scene(summary))
            answer = execute(question["instruction"], scene)
        if answer is not None:
            resolved += 1
            executor_correct += int(answer == gold)
            cascade_correct += int(answer == gold)
        else:
            cascade_correct += int(direct.get("prediction") == gold)
    report = {
        "gate": {
            "scenes_seen": len(vlm),
            "valid_rate": round(valid / max(1, total), 3),
            "object_recall_color_shape_key": round(matched_truth / max(1, truth_object_total), 3),
            "object_precision_color_shape_key": round(matched_truth / max(1, matched_vlm), 3),
            "cascade_accuracy": round(cascade_correct / len(questions), 4),
            "cascade_correct": cascade_correct,
            "d3p_direct": d3p_correct,
            "gate_pass": {
                "valid_ge_80": valid / max(1, total) >= 0.8,
                "recall_ge_70": matched_truth / max(1, truth_object_total) >= 0.7,
            },
        },
        "executor": {"resolved": resolved, "correct": executor_correct},
        "predicate_diagnostics": {k: prf_from(v) for k, v in buckets.items()},
        "rl_formalizer_targets": sorted(
            k for k, v in buckets.items() if v["fn"] > 0 and v["tp"] / (v["tp"] + v["fn"]) < 0.7
        ),
        "vlm_phase_map_5to8": VLM_PHASE_MAP,
    }
    out = Path(__file__).resolve().parents[2] / "reports/explore/s4-ns-quality-gate-001.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["gate"], indent=2))
    print(json.dumps(report["predicate_diagnostics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
