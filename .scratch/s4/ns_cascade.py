"""Neuro-symbolic cascade: template parser + deterministic executor over scene JSON.

Arms:
- d3p: D3P-600 adapter's direct video150 predictions (fallback floor).
- cascade: executor on base-VLM scene symbols where resolvable, else d3p.
- oracle: same executor on annotation-truth scenes (upper bound of the pipe).

Scene schema (from caption_probe): objects[{name, first_seen, last_seen}],
entry_order edges, collisions[{between, when}], present_at_start/end; phases
start<early<middle<late<end. Motion questions are outside the schema and
always fall back (recorded).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PHASES = {"start": 0, "early": 1, "middle": 2, "late": 3, "end": 4}
ATTR = r"(?:yellow|red|blue|green|gray|purple|brown|cyan|metal|rubber|sphere|spheres|cube|cubes|cylinder|cylinders|object|objects|moving|stationary)"

RE_ENTER_ANY = re.compile(rf"^Are there any ((?:{ATTR} ?)+)that enter the scene\?$")
RE_ENTER_REL = re.compile(
    rf"^Are there any ((?:{ATTR} ?)+)that enter the scene (before|after) the ((?:{ATTR} ?)+?)enters the scene\?$"
)
RE_EXIT_ANY = re.compile(rf"^Are there any ((?:{ATTR} ?)+)that exit the scene\?$")
RE_EXIT_REL = re.compile(
    rf"^Are there any ((?:{ATTR} ?)+)that exit the scene (before|after) the ((?:{ATTR} ?)+?)enters the scene\?$"
)
RE_MOVING = re.compile(rf"^Are there any (moving|stationary) ((?:{ATTR} ?)*?)when the video (begins|ends)\?$")
RE_MOVING_EVENT = re.compile(
    rf"^Are there any (moving|stationary) ((?:{ATTR} ?)*?)when the ((?:{ATTR} ?)+?)enters the scene\?$"
)
RE_MOVING_PLAIN = re.compile(rf"^Are there any (moving|stationary) ((?:{ATTR} ?)+)\?$")
RE_PRESENT_TIME = re.compile(
    rf"^Are there any ((?:{ATTR} ?)+?)when the video (begins|ends)\?$"
)
RE_PRESENT_EVENT = re.compile(
    rf"^Are there any ((?:{ATTR} ?)+?)when the ((?:{ATTR} ?)+?)enters the scene\?$"
)
RE_COLLISIONS_REL = re.compile(
    r"^Are there collisions (before|after) the ((?:{ATTR} ?)+?)enters the scene\?$"
)


def class_tokens(label: str) -> set[str]:
    words = set(label.strip().split())
    tokens = {w.rstrip("s") for w in words} - {"that"}
    # "objects"/"object" is the generic noun; scene names carry concrete shapes.
    return tokens - {"object"}


def matches(name_tokens: set[str], tokens: set[str]) -> bool:
    return tokens <= name_tokens


class Scene:
    def __init__(self, summary: dict):
        self.objects = summary.get("objects") or []
        self.tokens = [{w for w in o["name"].split()} for o in self.objects]
        self.first = [PHASES.get(o.get("first_seen"), -1) for o in self.objects]
        self.last = [PHASES.get(o.get("last_seen"), -1) for o in self.objects]
        self.edges = []
        for edge in summary.get("entry_order") or []:
            if " before " in edge:
                left, right = edge.split(" before ", 1)
                self.edges.append((left.strip(), right.strip()))
        self.enterers = {name for pair in self.edges for name in pair}
        self.collisions = summary.get("collisions") or []
        self.present_start = set(summary.get("present_at_start") or [])
        self.present_end = set(summary.get("present_at_end") or [])

    def members(self, tokens: set[str]) -> list[int]:
        return [i for i, t in enumerate(self.tokens) if matches(t, tokens)]

    def entry_rank(self, index: int) -> tuple[int, int] | None:
        name = self.objects[index]["name"]
        before = sum(1 for a, b in self.edges if b == name)
        after = sum(1 for a, b in self.edges if a == name)
        return (self.first[index], after - before) if self.first[index] >= 0 else None

    def anchor_index(self, tokens: set[str]) -> int | None:
        found = self.members(tokens)
        return found[0] if len(found) == 1 else None


def execute(instruction: str, scene: Scene) -> str | None:
    text = " ".join(instruction.split())
    match = RE_ENTER_REL.match(text)
    if match:
        members = scene.members(class_tokens(match.group(1)))
        anchor = scene.anchor_index(class_tokens(match.group(3)))
        if anchor is None:
            return None
        anchor_rank = scene.entry_rank(anchor)
        if anchor_rank is None:
            return None
        direction = match.group(2)
        for m in members:
            rank = scene.entry_rank(m)
            if rank is None or scene.first[m] == 0:
                continue
            if (direction == "after" and rank > anchor_rank) or (
                direction == "before" and rank < anchor_rank
            ):
                return "yes"
        return "no"
    match = RE_ENTER_ANY.match(text)
    if match:
        members = scene.members(class_tokens(match.group(1)))
        return "yes" if any(scene.first[m] > 0 for m in members) else "no"
    match = RE_EXIT_REL.match(text)
    if match:
        members = scene.members(class_tokens(match.group(1)))
        anchor = scene.anchor_index(class_tokens(match.group(3)))
        if anchor is None or scene.first[anchor] < 0:
            return None
        direction = match.group(2)
        for m in members:
            if scene.last[m] < 0 or scene.last[m] == 4:
                continue
            if (direction == "after" and scene.last[m] >= scene.first[anchor]) or (
                direction == "before" and scene.last[m] <= scene.first[anchor]
            ):
                return "yes"
        return "no"
    match = RE_EXIT_ANY.match(text)
    if match:
        members = scene.members(class_tokens(match.group(1)))
        return "yes" if any(scene.last[m] < 4 for m in members) else "no"
    if RE_MOVING_PLAIN.match(text) or RE_MOVING.match(text) or RE_MOVING_EVENT.match(text):
        return None  # motion outside scene schema -> fallback
    match = RE_PRESENT_EVENT.match(text)
    if match:
        members = scene.members(class_tokens(match.group(1)))
        anchor = scene.anchor_index(class_tokens(match.group(2)))
        if anchor is None or scene.first[anchor] < 0:
            return None
        phase = scene.first[anchor]
        for m in members:
            if scene.first[m] <= phase <= scene.last[m]:
                return "yes"
        return "no"
    match = RE_PRESENT_TIME.match(text)
    if match:
        members = scene.members(class_tokens(match.group(1)))
        boundary = match.group(2)
        if boundary == "begins":
            return "yes" if any(scene.first[m] == 0 for m in members) else "no"
        return "yes" if any(scene.last[m] == 4 for m in members) else "no"
    match = RE_COLLISIONS_REL.match(text)
    if match:
        anchor = scene.anchor_index(class_tokens(match.group(2)))
        if anchor is None or scene.first[anchor] < 0 or not scene.collisions:
            return None
        phase = scene.first[anchor]
        direction = match.group(1)
        for hit in scene.collisions:
            when = PHASES.get(hit.get("when"), -1)
            if when < 0:
                continue
            if (direction == "before" and when < phase) or (direction == "after" and when > phase):
                return "yes"
        return "no"
    return None


def main() -> int:
    if len(sys.argv) != 5:
        raise SystemExit("usage: ns_cascade.py QUESTIONS SCENES D3P_RECORDS OUTPUT")
    questions_path, scenes_path, d3p_path, out_path = (Path(a) for a in sys.argv[1:])
    questions = [json.loads(l) for l in questions_path.read_text().splitlines() if l.strip()]
    scenes = {}
    for line in scenes_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scenes[row["id"]] = row
    d3p = {}
    for line in d3p_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("pass", 0) == 0:
            d3p[row["id"]] = row
    stats = {
        "questions": len(questions),
        "scene_available": 0,
        "scene_valid": 0,
        "parsed": 0,
        "resolved": 0,
        "motion_fallback": 0,
        "d3p_correct": 0,
        "executor_correct": 0,
        "executor_wrong": 0,
        "cascade_correct": 0,
        "oracle_correct": 0,
    }
    per_template = {}
    for question in questions:
        gold = question["target"]["value"]
        row = scenes.get(question["id"])
        answer = None
        if row and row.get("summary_valid") and row.get("summary"):
            stats["scene_available"] += 1
            stats["scene_valid"] += 1
            scene = Scene(row["summary"])
            answer = execute(question["instruction"], scene)
        if answer is None and row and RE_MOVING_PLAIN.match(" ".join(question["instruction"].split())):
            stats["motion_fallback"] += 1
        direct = d3p.get(question["id"], {})
        direct_correct = direct.get("prediction") == gold
        stats["d3p_correct"] += int(direct_correct)
        if answer is not None:
            stats["resolved"] += 1
            key = type(answer).__name__
            correct = answer == gold
            stats["executor_correct" if correct else "executor_wrong"] += 1
            stats["cascade_correct"] += int(correct)
            template = question["instruction"].split(" that ")[0][:30]
            bucket = per_template.setdefault(template, [0, 0])
            bucket[0] += int(correct)
            bucket[1] += 1
        else:
            stats["cascade_correct"] += int(direct_correct)
    summary = {
        **stats,
        "d3p_accuracy": round(stats["d3p_correct"] / len(questions), 4),
        "cascade_accuracy": round(stats["cascade_correct"] / len(questions), 4),
        "executor_coverage": round(stats["resolved"] / len(questions), 4),
        "executor_accuracy_on_resolved": round(
            stats["executor_correct"] / max(1, stats["resolved"]), 4
        ),
        "per_template": {k: [v[0], v[1]] for k, v in per_template.items()},
    }
    Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("d3p_accuracy", "cascade_accuracy", "executor_coverage", "executor_accuracy_on_resolved")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
