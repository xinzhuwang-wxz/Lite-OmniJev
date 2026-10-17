"""Build four-type CLEVRER evaluation samples in the product contract (CPU).

Official question types map onto Jev primitives without changing their metric:
descriptive exist -> noul; count / query_* -> choice; explanatory, predictive and
counterfactual -> one noul per listed choice ("is this candidate correct?"), so the
official per-option accuracy is per-sample accuracy and per-question accuracy is
"all options of one question correct" (group = id prefix before "-c"). Official
types go to a sidecar OUT.types.jsonl because the sample contract forbids extras.

Usage: --questions train.json --media-root DIR --exclude-families FILE
       --videos-from N --videos-count K --seed S --split NAME --output OUT.jsonl
Only videos absent from --exclude-families (one "<index> <pool>" per line) are
eligible; the selection is a seeded shuffle, so debug and judge splits are taken
as disjoint ranges of the same order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from data.schema import Sample

COUNT_OPTIONS = ["0", "1", "2", "3", "4", "5"]
QUERY_OPTIONS = {
    "query_color": ["gray", "red", "blue", "green", "brown", "cyan", "purple", "yellow"],
    "query_shape": ["sphere", "cube", "cylinder"],
    "query_material": ["metal", "rubber"],
}
MC_TYPES = ("explanatory", "predictive", "counterfactual")
STATE = "Answer the question using the attached video."


def question_rows(question: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    """Return (suffix, product question, official subtype) rows for one official question."""
    kind = question["question_type"]
    text = question["question"]
    if kind == "descriptive":
        subtype = question["question_subtype"]
        answer = str(question["answer"])
        if subtype == "exist":
            spec = {"type": "noul", "options": None, "value": answer}
        elif subtype == "count":
            spec = {"type": "choice", "options": COUNT_OPTIONS, "value": answer}
        else:
            spec = {"type": "choice", "options": QUERY_OPTIONS[subtype], "value": answer}
        if spec["type"] == "choice":
            options = spec["options"]
            lettered = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
            instruction = f"{text}\nOptions:\n{lettered}"
            value = chr(65 + options.index(spec["value"]))
        else:
            instruction, value = text, spec["value"]
        return [("", _question(spec["type"], instruction, spec["options"], value), subtype)]
    rows = []
    for choice in question["choices"]:
        instruction = (
            f"Question: {text}\nCandidate answer: {choice['choice']}\n"
            "Is this candidate answer correct?"
        )
        value = "yes" if choice["answer"] == "correct" else "no"
        rows.append((f"-c{choice['choice_id']}", _question("noul", instruction, None, value), kind))
    return rows


def _question(kind: str, instruction: str, options: list[str] | None, value: str) -> dict[str, Any]:
    return {
        "qid": kind,
        "type": kind,
        "instruction": instruction,
        "options": options,
        "score_levels": None,
        "score_descriptions": None,
        "target": {"kind": "hard", "source": "clevrer_official_answer", "value": value},
        "pair_of": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--exclude-families", type=Path, required=True)
    parser.add_argument("--videos-from", type=int, required=True)
    parser.add_argument("--videos-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.questions.read_bytes()
    questions_sha = hashlib.sha256(raw).hexdigest()
    videos = json.loads(raw)
    excluded = {
        int(line.split()[0])
        for line in args.exclude_families.read_text().splitlines()
        if line.strip()
    }
    eligible = sorted(v["scene_index"] for v in videos if v["scene_index"] not in excluded)
    random.Random(args.seed).shuffle(eligible)
    chosen = set(eligible[args.videos_from : args.videos_from + args.videos_count])
    by_index = {v["scene_index"]: v for v in videos}
    counts: dict[str, int] = {}
    sidecar = args.output.with_suffix(".types.jsonl")
    with (
        args.output.open("x", encoding="utf-8") as stream,
        sidecar.open("x", encoding="utf-8") as types,
    ):
        for index in sorted(chosen):
            uri = f"video/video_{index:05d}.mp4"
            if not (args.media_root / uri).is_file():
                counts["skipped_missing_media"] = counts.get("skipped_missing_media", 0) + 1
                continue
            media_sha = hashlib.sha256((args.media_root / uri).read_bytes()).hexdigest()
            for official in by_index[index]["questions"]:
                for suffix, question, subtype in question_rows(official):
                    sample = Sample.model_validate(
                        {
                            "id": f"clevrer-train-{index:05d}-{official['question_id']}{suffix}",
                            "prefix": {
                                "state": STATE,
                                "media": [{"type": "video", "uri": uri, "sha256": media_sha}],
                            },
                            "questions": [question],
                            "meta": {
                                "modality": ["video"],
                                "source_dataset": "CLEVRER",
                                "source_revision": f"official-train-json-sha{questions_sha}",
                                "source_sha256": questions_sha,
                                "license": "CC0-1.0",
                                "usage": "commercial",
                                "pool": "dev",
                                "family_id": f"clevrer:{index}",
                                "data_version": f"clevrer-fourtype-{args.split}-v1",
                            },
                        }
                    )
                    stream.write(
                        json.dumps(sample.model_dump(mode="json"), ensure_ascii=False) + "\n"
                    )
                    types.write(
                        json.dumps(
                            {
                                "id": sample.id,
                                "question_type": official["question_type"],
                                "subtype": subtype,
                            }
                        )
                        + "\n"
                    )
                    counts[subtype] = counts.get(subtype, 0) + 1
    print(
        json.dumps(
            {"split": args.split, "videos": len(chosen), "rows_by_subtype": counts}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
