"""Formalizer reward: executor coverage/correctness + event-predicate F1, offline.

R(scene, video_record) = VALIDITY_FLOOR
    + w_cov * executor_resolved_fraction
    + w_acc * executor_correct_fraction
    + sum_event w_p * F1_p(generated, truth)
Every term is computable from the generated scene text alone (CPU); truth is
the cached annotation scene. No GPU, no new labels.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ns_cascade import Scene, execute  # noqa: E402
from ns_quality_gate import salvage_summary  # noqa: E402

VALIDITY_FLOOR = -1.0
W_CORRECTNESS = 0.6  # fraction of probe questions answered correctly
W_WRONG = -0.2  # penalty per wrongly-resolved fraction (empty scenes must not farm coverage)
W_COVERAGE = 0.1  # small credit for resolvability itself
EVENT_WEIGHTS = {"enterers": 0.05, "exiters": 0.05, "moving": 0.05, "collisions": 0.05}
VLM_PHASE_MAP = {"start": 0, "early": 1, "middle": 3, "late": 5, "end": 7}


def scene_from_generation(raw_text: str | None, summary: dict | None):
    if summary:
        return summary, "native"
    if raw_text:
        salvaged, _ = salvage_summary(raw_text)
        if salvaged:
            return salvaged, "salvaged"
    return None, "invalid"


def _sets(summary: dict) -> dict:
    objects = summary.get("objects", [])
    out = {
        "enterers": set(),
        "exiters": set(),
        "moving": set(),
        "collisions": set(),
    }
    ignore = {"metal", "rubber", "object"}

    def key(name):
        return frozenset(w for w in name.split() if w not in ignore)

    for obj in objects:
        k = key(obj["name"])
        first = VLM_PHASE_MAP.get(obj.get("first_seen"), obj.get("first_seen"))
        last = VLM_PHASE_MAP.get(obj.get("last_seen"), obj.get("last_seen"))
        if isinstance(first, int) and first > 0:
            out["enterers"].add(k)
        if isinstance(last, int) and last < 7:
            out["exiters"].add(k)
        if obj.get("moving"):
            out["moving"].add(k)
    for hit in summary.get("collisions", []):
        pair = hit.get("between", [])
        if len(pair) == 2:
            out["collisions"].add(frozenset(key(n) for n in pair))
    return out


def f1(predicted: set, truth: set) -> float:
    if not predicted and not truth:
        return 1.0
    tp = len(predicted & truth)
    if not tp:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return 2 * precision * recall / (precision + recall)


def reward(
    generation: dict,
    record: dict,
    *,
    mapped: bool = False,
) -> dict:
    summary, source = scene_from_generation(
        generation.get("raw_text"), generation.get("summary")
    )
    if summary is None:
        return {"reward": VALIDITY_FLOOR, "source": "invalid", "terms": {}}
    terms = {"validity": 1.0, "source_kind": source}
    questions = record.get("probe_questions", [])
    resolved = correct = 0
    scene = Scene(summary if not mapped else _mapped(summary))
    for question in questions:
        answer = execute(question["instruction"], scene)
        if answer is None:
            continue
        resolved += 1
        correct += int(answer == question["gold"])
    if questions:
        wrong = resolved - correct
        terms["coverage"] = W_COVERAGE * resolved / len(questions)
        terms["correctness"] = W_CORRECTNESS * correct / len(questions)
        terms["wrong_penalty"] = W_WRONG * wrong / len(questions)
    truth_sets = _sets(record["truth_scene"])
    gen_sets = _sets(summary)
    event_f1 = {}
    for predicate, weight in EVENT_WEIGHTS.items():
        value = f1(gen_sets[predicate], truth_sets[predicate])
        event_f1[predicate] = round(value, 4)
        terms[f"f1_{predicate}"] = weight * value
    total = math.fsum(v for k, v in terms.items() if isinstance(v, (int, float)))
    return {
        "reward": round(total, 4),
        "source": source,
        "terms": terms,
        "event_f1": event_f1,
        "resolved": resolved,
        "correct": correct,
        "questions": len(questions),
    }


def _mapped(summary: dict) -> dict:
    from ns_quality_gate import mapped_scene

    return mapped_scene(summary)


def group_advantages(rewards: list[float]) -> list[float]:
    """GRPO per-prompt group advantage over the G sampled scenes."""
    mean = math.fsum(rewards) / len(rewards)
    std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + 1e-6) for r in rewards]


def _self_test() -> None:
    record = {
        "truth_scene": {
            "objects": [
                {"name": "red rubber cube", "first_seen": "start", "last_seen": "end", "moving": False},
                {"name": "blue metal sphere", "first_seen": "early", "last_seen": "late", "moving": True},
            ],
            "entry_order": ["blue metal sphere before red rubber cube"] if False else [],
            "collisions": [],
            "present_at_start": ["red rubber cube"],
            "present_at_end": ["red rubber cube"],
        },
        "probe_questions": [
            {
                "instruction": "Are there any red cubes when the video ends?",
                "gold": "yes",
                "predicate": "exiters",
            },
            {
                "instruction": "Are there any moving blue spheres?",
                "gold": "yes",
                "predicate": "moving",
            },
        ],
    }
    perfect = reward({"summary": record["truth_scene"]}, record)
    assert perfect["source"] == "native" and perfect["reward"] > 0.8, perfect
    assert perfect["correct"] == perfect["resolved"] == 2, perfect
    invalid = reward({"raw_text": "no json at all"}, record)
    assert invalid["reward"] == VALIDITY_FLOOR
    empty_scene = reward(
        {"summary": {"objects": [], "entry_order": [], "collisions": [],
                     "present_at_start": [], "present_at_end": []}},
        record,
    )
    assert empty_scene["reward"] < perfect["reward"]
    advantages = group_advantages([0.2, 0.4, 0.4, 1.0])
    assert abs(math.fsum(advantages)) < 1e-6 and max(advantages) > 0 > min(advantages)
    print(json.dumps({"self_test": "passed", "perfect_reward": perfect["reward"],
                      "empty_reward": empty_scene["reward"], "invalid_reward": invalid["reward"],
                      "perfect_event_f1": perfect["event_f1"]}, indent=2))


if __name__ == "__main__":
    _self_test()
