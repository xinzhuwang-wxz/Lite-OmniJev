"""Read-only exact-text/opposite-gold TRAIN capacity; no dataset or model execution."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import time
from pathlib import Path

from data.schema import Sample
from scripts.s0_media_dev import source_hash_for_video

path = Path(__file__).with_name("inspect-temporal-order-contrast.py")
spec = importlib.util.spec_from_file_location("matched_official_source", path)
assert spec and spec.loader
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
sha = source.sha


def require(ok, why):
    if not ok:
        raise ValueError(why)


def syntax(program):
    if any(t in program for t in ["filter_before", "filter_after"]):
        return "native_event_relation"
    if "get_frame" in program and "events" in program:
        return "event_conditioned_motion"
    if "get_frame" in program:
        return "endpoint_motion"
    return "other_exist"


def capacity(rows):
    """Exact text AND exact postfix; unique families make bucket capacities disjoint."""
    require(
        len(rows) == len({r["family"] for r in rows}), "one source question per family required"
    )
    text_groups = collections.defaultdict(list)
    groups = collections.defaultdict(list)
    for row in rows:
        require(row["gold"] in {"yes", "no"}, "official binary gold")
        text_groups[row["text"]].append(row)
        groups[(row["text"], tuple(row["program"]))].append(row)
    counts = collections.Counter()
    pair_count = 0
    matched_questions = 0
    strict_families = set()
    for values in groups.values():
        yes = [r for r in values if r["gold"] == "yes"]
        no = [r for r in values if r["gold"] == "no"]
        if not yes or not no:
            continue
        # Matching distinct SHA is a tiny bipartite matching problem, not a cartesian-pair count.
        matches = {}

        def augment(i, visited, no=no, yes=yes, matches=matches):
            for j, b in enumerate(no):
                if yes[i]["media"] == b["media"] or j in visited:
                    continue
                visited.add(j)
                if j not in matches or augment(matches[j], visited):
                    matches[j] = i
                    return True
            return False

        n = sum(augment(i, set()) for i in range(len(yes)))
        if n:
            matched_questions += 1
            pair_count += n
            counts[syntax(values[0]["program"])] += n
            strict_families.update(r["family"] for r in values)
    opposite_text = [v for v in text_groups.values() if {r["gold"] for r in v} == {"yes", "no"}]
    return {
        "source_rows": len(rows),
        "gold_counts": dict(collections.Counter(r["gold"] for r in rows)),
        "exact_text_groups": len(text_groups),
        "opposite_gold_exact_text_groups": len(opposite_text),
        "opposite_text_groups_with_multiple_programs": sum(
            len({tuple(r["program"]) for r in v}) > 1 for v in opposite_text
        ),
        "exact_text_and_program_opposite_groups": matched_questions,
        "max_disjoint_distinct_media_pairs": pair_count,
        "usable_families_in_opposite_groups": len(strict_families),
        "pair_capacity_by_syntax": dict(counts),
        "feasible30pairs60unique_families": pair_count >= 30,
    }


def parent_families(root):
    lineage = root / "reports/s4-ce0-video-exposure-001.json"
    require(
        sha(lineage) == "cb9574e06fc9950198f703ab9107d6d2d42824a5d2399ded2a26a146dd88cbfa",
        "lineage receipt",
    )
    report = json.loads(lineage.read_bytes())
    for stage in report["stages"]:
        require(
            sha(stage["report"]) == stage["report_sha256"]
            and sha(stage["data"]) == stage["data_sha256"],
            "lineage inputs",
        )
    first = root / "artifacts/s4-temporal-coverage-pilot-001"
    require(
        sha(first / "report.json")
        == "1e7f877e04ea311d29ebc1fe1711f31aa2cc91034f8794b666850bcf5f2fff19",
        "first receipt",
    )
    receipt = json.loads((first / "report.json").read_bytes())
    require(sha(first / "train.jsonl") == receipt["output_sha256"]["train.jsonl"], "first data")
    samples = [
        Sample.model_validate_json(line)
        for line in (first / "train.jsonl").read_text().splitlines()
    ]
    result = set(report["lineage_unique_video_families"]) | {
        s.meta.family_id for s in samples if s.meta.modality == ["video"]
    }
    require(len(result) == 408, "parent408")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    require(not args.output.exists(), "output exists")
    pool = args.root / "artifacts/s3-pools-video-001/run-a/train.jsonl"
    require(sha(pool) == source.POOL_SHA, "admitted8299 TRAIN SHA")
    questions = args.official / "questions/train.json"
    require(
        sha(questions) == source.QUESTIONS_SHA
        and sha(args.official / "README.txt") == source.README_SHA,
        "official source",
    )
    samples = [
        Sample.model_validate_json(line) for line in pool.read_text().splitlines() if line.strip()
    ]
    require(len(samples) == 8299 and all(s.meta.pool == "train" for s in samples), "TRAIN-only8299")
    excluded = parent_families(args.root)
    scenes = source.selected_scenes(
        questions, {int(s.meta.family_id.split(":")[1]) for s in samples}
    )
    rows = []
    for sample in samples:
        index = int(sample.meta.family_id.split(":")[1])
        q = next(
            q
            for q in scenes[index]["questions"]
            if q["question_id"] == int(sample.id.rsplit("-", 1)[-1])
        )
        require(
            q["question"] == sample.questions[0].instruction
            and q["answer"].casefold() == sample.questions[0].target.value,
            "original text/gold",
        )
        require(
            source_hash_for_video({"scene_index": index}, q, sample.prefix.media[0].sha256)
            == sample.meta.source_sha256,
            "official program/media binding",
        )
        rows.append(
            {
                "family": sample.meta.family_id,
                "text": q["question"],
                "gold": q["answer"].casefold(),
                "program": q["program"],
                "media": sample.prefix.media[0].sha256,
            }
        )
    result = {
        "status": "completed",
        "exit_code": 0,
        "gpu_used": False,
        "cost": None,
        "normalization": "none; exact original instruction bytes/codepoints, no semantic rewriting",
        "all_train": capacity(rows),
        "unseen_to_working_f2db_parent": capacity([r for r in rows if r["family"] not in excluded]),
        "excluded_parent_families": 408,
        "unseen_scope": "only relative to f2db lineage, not all historical experiments",
        "training_authorized": False,
        "new_pool_created": False,
        "bindings": {
            "data_sha256": source.POOL_SHA,
            "questions_sha256": source.QUESTIONS_SHA,
            "readme_sha256": source.README_SHA,
            "helper_sha256": sha(Path(__file__)),
            "source_helper_sha256": sha(path),
        },
        "duration_seconds": time.monotonic() - started,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
