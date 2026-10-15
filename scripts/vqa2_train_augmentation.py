"""Build conservative VQA2 Choice gold on already allocated train images only.

Normalization is Unicode casefold, whitespace collapse, and terminal .?! removal;
it deliberately does not merge synonyms, number words, or compound colors.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts.s2_media_pools import (
    canonical_family_hash,
    canonical_json,
    canonical_sample_sha,
    sha256_bytes,
    sha256_file,
)

VERSION = "vqa2-existing-train-choice-v1"
LICENSE = "QA:CC-BY-4.0;media:CC-BY-2.0"
COLORS = (
    "black",
    "white",
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
)


def normalize(value: str) -> str:
    return " ".join(value.casefold().split()).rstrip(".?!").strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_archive(path: Path, member: str, key: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist().count(member) != 1:
            raise ValueError("missing or duplicate expected JSON member")
        payload = json.loads(archive.read(member))
    if payload.get("license", {}).get("url", "").rstrip("/") not in {
        "http://creativecommons.org/licenses/by/4.0",
        "https://creativecommons.org/licenses/by/4.0",
    }:
        raise ValueError("archive license is not CC-BY-4.0")
    return [row for row in payload[key]]


def build(
    *,
    parent_train: Path,
    parent_ledger: Path,
    train_questions: Path,
    train_annotations: Path,
    val_questions: Path,
    val_annotations: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("refusing to overwrite output directory")
    inputs = {
        "parent_train": parent_train,
        "parent_ledger": parent_ledger,
        "train_questions": train_questions,
        "train_annotations": train_annotations,
        "val_questions": val_questions,
        "val_annotations": val_annotations,
    }
    hashes = {name: sha256_file(path) for name, path in inputs.items()}
    ledger = read_jsonl(parent_ledger)
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reserved: set[str] = set()
    for row in ledger:
        by_item[row["item_id"]].append(row)
        if row["pool"] != "train":
            reserved.add(canonical_family_hash(row["family_id"]))
            if "family_hash" in row:
                reserved.add(row["family_hash"])
    parents: dict[int, Sample] = {}
    old_questions: set[tuple[int, str]] = set()
    for raw in read_jsonl(parent_train):
        sample = Sample.model_validate(raw)
        if sample.meta.pool != "train":
            raise ValueError("parent sample must be train")
        if not re.fullmatch(r"coco:(0|[1-9][0-9]*)", sample.meta.family_id):
            continue
        family = sample.meta.family_id
        records = by_item[sample.id]
        if (
            len(records) != 1
            or records[0]["pool"] != "train"
            or records[0]["family_id"] != family
            or records[0].get("derived_sample_sha256") != canonical_sample_sha(sample)
            or canonical_family_hash(family) in reserved
        ):
            raise ValueError("parent train ledger membership or binding invalid")
        if (
            sample.meta.usage != "commercial"
            or sample.meta.license.split(";media:")[-1] != "CC-BY-2.0"
            or len(sample.prefix.media) != 1
            or sample.prefix.media[0].type != "image"
        ):
            raise ValueError("parent image not approved commercial CC-BY-2.0")
        image_id = int(family.split(":")[1])
        if image_id in parents and parents[image_id].prefix.media != sample.prefix.media:
            raise ValueError("inconsistent parent family media")
        parents.setdefault(image_id, sample)
        old_questions.update((image_id, normalize(q.instruction)) for q in sample.questions)
    if not parents:
        raise ValueError("no approved parent images")
    excluded: Counter[str] = Counter()
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[int] = set()
    for split in ("train", "val"):
        questions = read_archive(
            inputs[f"{split}_questions"],
            f"v2_OpenEnded_mscoco_{split}2014_questions.json",
            "questions",
        )
        annotations = read_archive(
            inputs[f"{split}_annotations"], f"v2_mscoco_{split}2014_annotations.json", "annotations"
        )
        allowed_annotations: dict[int, dict[str, Any]] = {}
        for ann in annotations:
            if ann.get("image_id") not in parents:
                continue
            qid = ann["question_id"]
            if qid in allowed_annotations:
                raise ValueError("duplicate allowed annotation ID")
            allowed_annotations[qid] = ann
        for question in questions:
            image_id = question["image_id"]
            if image_id not in parents:
                excluded["outside_parent_train"] += 1
                continue
            qid = question["question_id"]
            if qid in seen_ids:
                raise ValueError("duplicate allowed question ID")
            seen_ids.add(qid)
            matched = allowed_annotations.get(qid)
            if matched is None or matched["image_id"] != image_id:
                raise ValueError("allowed question annotation mismatch")
            ann = matched
            text = normalize(question["question"])
            answers = ann.get("answers", [])
            normalized = [normalize(a["answer"]) for a in answers]
            gold = normalized[0] if normalized else ""
            valid = (
                len(normalized) == 10
                and len(set(normalized)) == 1
                and gold == normalize(ann["multiple_choice_answer"])
                and bool(gold)
            )
            grouped[(image_id, text)].append(
                {
                    "question": question,
                    "annotation": ann,
                    "gold": gold,
                    "valid": valid,
                    "split": split,
                }
            )
    samples: list[Sample] = []
    provenance: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    for (image_id, text), rows in sorted(grouped.items()):
        if (image_id, text) in old_questions:
            excluded["parent_duplicate"] += len(rows)
            continue
        if len({r["gold"] for r in rows}) > 1:
            excluded["conflicting_duplicate"] += len(rows)
            continue
        if not all(r["valid"] for r in rows):
            excluded["consensus_disagreement"] += len(rows)
            continue
        excluded["duplicate_question"] += len(rows) - 1
        row = min(rows, key=lambda r: r["question"]["question_id"])
        gold = row["gold"]
        qid = row["question"]["question_id"]
        rng = random.Random(f"42:{image_id}:{qid}")
        if gold in {"yes", "no"}:
            category, options = "yes_no", ["yes", "no"]
        elif re.fullmatch(r"(?:[0-9]|1[0-9]|20)", gold):
            category = "integer"
            options = [gold] + rng.sample([str(i) for i in range(21) if str(i) != gold], 3)
        elif gold in COLORS and re.search(
            r"\bcolou?r\b", text + " " + normalize(row["annotation"].get("question_type", ""))
        ):
            category = "color"
            options = [gold] + rng.sample([c for c in COLORS if c != gold], 3)
        else:
            excluded["unsupported_answer"] += 1
            continue
        rng.shuffle(options)
        parent = parents[image_id]
        record = {
            "id": f"vqa2-train-{qid}",
            "question_id": qid,
            "image_id": image_id,
            "category": category,
            "answers": row["annotation"]["answers"],
            "normalized_gold": gold,
            "normalization": "casefold+collapse-whitespace+strip-terminal-.?!",
            "duplicate_question_ids": sorted(r["question"]["question_id"] for r in rows),
            "question_archive_sha256": hashes[f"{row['split']}_questions"],
            "annotation_archive_sha256": hashes[f"{row['split']}_annotations"],
            "license": LICENSE,
            "distractors": "synthesized-domain-options",
            "parent_sample_id": parent.id,
            "parent_sample_sha256": canonical_sample_sha(parent),
        }
        payload = parent.model_dump(mode="json")
        payload["id"] = record["id"]
        payload["questions"] = [
            {
                "qid": "choice",
                "type": "choice",
                "instruction": row["question"]["question"],
                "options": options,
                "target": {
                    "kind": "hard",
                    "value": chr(65 + options.index(gold)),
                    "source": "vqa2_unanimous_10_human_answers",
                },
            }
        ]
        payload["meta"].update(
            source_dataset="VQA-v2",
            source_revision="2.0-2017-04-26",
            source_sha256=sha256_bytes(canonical_json(record).encode()),
            license=LICENSE,
            data_version=VERSION,
        )
        samples.append(Sample.model_validate(payload))
        provenance.append(record)
        categories[category] += 1
    output_dir.mkdir(parents=True)
    outputs: dict[str, str] = {}
    allocation = [
        {
            "item_id": s.id,
            "pool": "train",
            "family_id": s.meta.family_id,
            "family_hash": canonical_family_hash(s.meta.family_id),
            "derived_sample_sha256": canonical_sample_sha(s),
            "parent_ledger_sha256": hashes["parent_ledger"],
            "source_dataset": s.meta.source_dataset,
            "source_revision": s.meta.source_revision,
            "license": LICENSE,
        }
        for s in samples
    ]
    for name, values in {
        "train.jsonl": [s.model_dump(mode="json") for s in samples],
        "allocation-ledger.jsonl": allocation,
        "provenance.jsonl": provenance,
    }.items():
        path = output_dir / name
        path.write_text("".join(canonical_json(v) + "\n" for v in values))
        outputs[name] = sha256_file(path)
    report = {
        "data_version": VERSION,
        "seed": 42,
        "input_sha256": hashes,
        "output_sha256": outputs,
        "sample_count": len(samples),
        "parent_image_count": len(parents),
        "category_counts": dict(categories),
        "excluded_counts": dict(excluded),
        "distractors": "synthesized-domain-options",
        "license": LICENSE,
        "script_sha256": sha256_file(Path(__file__)),
    }
    (output_dir / "report.json").write_text(canonical_json(report) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-train",
        "parent-ledger",
        "train-questions",
        "train-annotations",
        "val-questions",
        "val-annotations",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build(**vars(args))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile):
        parser.exit(
            2, "VQA2 augmentation rejected: invalid input, binding, license, or output path\n"
        )
    print(canonical_json(report))


if __name__ == "__main__":
    main()
