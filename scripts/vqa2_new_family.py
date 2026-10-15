"""Plan and build unanimous VQA2 Choice samples on previously unused COCO families."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
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
from scripts.vqa2_train_augmentation import COLORS, LICENSE, normalize, read_archive, read_jsonl
from train.multimodal import resolve_media

VERSION = "vqa2-new-family-choice-v1"
MEDIA_LICENSE = "http://creativecommons.org/licenses/by/2.0/"


def _write(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))
    return sha256_file(path)


def _exclusions(paths: list[Path]) -> tuple[set[str], set[str]]:
    families: set[str] = set()
    media: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            if row.get("family_id"):
                families.add(canonical_family_hash(row["family_id"]))
            if row.get("family_hash"):
                families.add(row["family_hash"])
            for key in ("media_sha256", "media_sha256s"):
                value = row.get(key, [])
                media.update([value] if isinstance(value, str) else value)
    return families, media


def plan(
    *,
    coco_metadata: Path,
    aok_train: Path,
    train_questions: Path,
    train_annotations: Path,
    output_dir: Path,
    allocation_ledger: list[Path] | None = None,
    seed: int = 42,
    color: int = 160,
    integer: int = 160,
    yes_no: int = 80,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("refusing to overwrite output directory")
    quotas = {"color": color, "integer": integer, "yes_no": yes_no}
    if min(quotas.values()) < 0 or not 1 <= sum(quotas.values()) <= 600:
        raise ValueError("sample quota must be between 1 and 600")
    paths = {
        "coco_metadata": coco_metadata,
        "aok_train": aok_train,
        "train_questions": train_questions,
        "train_annotations": train_annotations,
    }
    paths.update({f"ledger_{i}": p for i, p in enumerate(allocation_ledger or [])})
    hashes = {k: sha256_file(p) for k, p in paths.items()}
    families, media = _exclusions(allocation_ledger or [])
    # Only image identifiers are inspected from the original AOK source.
    aok_ids = {row["image_id"] for row in json.loads(aok_train.read_text())}
    metadata = json.loads(coco_metadata.read_text())
    licenses = [r for r in metadata["licenses"] if r["id"] == 4]
    if len(licenses) != 1 or licenses[0]["url"] != MEDIA_LICENSE:
        raise ValueError("COCO license 4 must be exact CC-BY-2.0")
    images: dict[int, dict[str, Any]] = {}
    for row in metadata["images"]:
        iid = row["id"]
        if iid in aok_ids or canonical_family_hash(f"coco:{iid}") in families:
            continue
        if row["license"] != 4:
            continue
        expected = f"{iid:012d}.jpg"
        if row["file_name"] != expected or row.get("coco_url") not in {
            f"http://images.cocodataset.org/train2017/{expected}",
            f"https://images.cocodataset.org/train2017/{expected}",
        }:
            continue
        if iid in images:
            raise ValueError("duplicate image metadata")
        images[iid] = row
    # Establish exclusion membership before accessing question/answer fields.
    annotations = {}
    for row in read_archive(
        train_annotations, "v2_mscoco_train2014_annotations.json", "annotations"
    ):
        if row["image_id"] not in images:
            continue
        if row["question_id"] in annotations:
            raise ValueError("duplicate annotation")
        annotations[row["question_id"]] = row
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[int] = set()
    for q in read_archive(
        train_questions, "v2_OpenEnded_mscoco_train2014_questions.json", "questions"
    ):
        iid = q["image_id"]
        if iid not in images:
            continue
        qid = q["question_id"]
        if qid in seen:
            raise ValueError("duplicate question")
        seen.add(qid)
        ann = annotations.get(qid)
        if ann is None or ann["image_id"] != iid:
            raise ValueError("question annotation mismatch")
        answers = [normalize(a["answer"]) for a in ann["answers"]]
        gold = normalize(ann["multiple_choice_answer"])
        valid = len(answers) == 10 and set(answers) == {gold} and bool(gold)
        text = normalize(q["question"])
        rng = random.Random(f"{seed}:{iid}:{qid}")
        category, options = "unsupported", []
        if gold in {"yes", "no"}:
            category, options = "yes_no", ["yes", "no"]
        elif re.fullmatch(r"(?:[0-9]|1[0-9]|20)", gold):
            category, options = (
                "integer",
                [gold] + rng.sample([str(i) for i in range(21) if str(i) != gold], 3),
            )
        elif gold in COLORS and re.search(
            r"\bcolou?r\b", text + " " + normalize(ann.get("question_type", ""))
        ):
            category, options = "color", [gold] + rng.sample([c for c in COLORS if c != gold], 3)
        rng.shuffle(options)
        grouped[(iid, text)].append(
            {
                "image_id": iid,
                "question_id": qid,
                "question": q["question"],
                "answers": ann["answers"],
                "gold": gold,
                "valid": valid,
                "category": category,
                "options": options,
            }
        )
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in grouped.values():
        if not all(r["valid"] for r in rows) or len({r["gold"] for r in rows}) != 1:
            continue
        row = min(rows, key=lambda r: r["question_id"])
        if row["category"] in quotas:
            buckets[row["category"]].append(row)
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for category, quota in quotas.items():
        ordered = sorted(
            buckets[category],
            key=lambda r: sha256_bytes(f"{seed}:{r['image_id']}:{r['question_id']}".encode()),
        )
        count = 0
        for row in ordered:
            if count >= quota:
                break
            if row["image_id"] in used:
                continue
            used.add(row["image_id"])
            count += 1
            meta = images[row["image_id"]]
            selected.append(
                {
                    **row,
                    "family_id": f"coco:{row['image_id']}",
                    "url": meta["coco_url"],
                    "file_name": meta["file_name"],
                    "attribution": {
                        "flickr_url": meta.get("flickr_url"),
                        "license_url": MEDIA_LICENSE,
                    },
                    "source_metadata_sha256": sha256_bytes(canonical_json(meta).encode()),
                    "license": LICENSE,
                    "distractors": "synthesized-domain-options",
                }
            )
    if not selected:
        raise ValueError("no eligible new families")
    output_dir.mkdir(parents=True)
    candidate_sha = _write(output_dir / "candidates.jsonl", selected)
    report = {
        "data_version": VERSION,
        "seed": seed,
        "input_sha256": hashes,
        "input_paths": {k: str(p.resolve()) for k, p in paths.items()},
        "candidates_sha256": candidate_sha,
        "sample_count": len(selected),
        "category_counts": dict(Counter(r["category"] for r in selected)),
        "requested_counts": quotas,
        "excluded_media_sha256": sorted(media),
        "excluded_family_hashes": sorted(families),
        "script_sha256": sha256_file(Path(__file__)),
    }
    (output_dir / "plan.json").write_text(canonical_json(report) + "\n")
    return report


def build(
    *, plan_dir: Path, download_manifest: Path, output_dir: Path, media_root: Path
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("refusing to overwrite output directory")
    report = json.loads((plan_dir / "plan.json").read_text())
    if (
        report["data_version"] != VERSION
        or sha256_file(plan_dir / "candidates.jsonl") != report["candidates_sha256"]
    ):
        raise ValueError("plan binding mismatch")
    for key, source in report["input_paths"].items():
        if sha256_file(Path(source)) != report["input_sha256"][key]:
            raise ValueError("input hash mismatch")
    image_module = importlib.import_module("PIL.Image")
    downloads = {}
    for row in read_jsonl(download_manifest):
        if row["image_id"] in downloads:
            raise ValueError("duplicate download image")
        downloads[row["image_id"]] = row
    samples, provenance, ledger = [], [], []
    seen_media = set(report["excluded_media_sha256"])
    seen_families = set(report["excluded_family_hashes"])
    for row in read_jsonl(plan_dir / "candidates.jsonl"):
        media = downloads[row["image_id"]]
        path = Path(media["path"])
        digest = sha256_file(path)
        uri = str(path.absolute().relative_to(media_root.absolute()))
        resolve_media(media_root.absolute(), uri, digest, max_bytes=100_000_000)
        family_hash = canonical_family_hash(row["family_id"])
        if digest != media["sha256"] or media["url"] != row["url"]:
            raise ValueError("download hash or URL mismatch")
        if digest in seen_media or family_hash in seen_families:
            raise ValueError("existing or duplicate media/family")
        seen_media.add(digest)
        seen_families.add(family_hash)
        with image_module.open(path) as image:
            image.verify()
        with image_module.open(path) as image:
            image.load()
        source = {**row, "media_sha256": digest, "plan_sha256": sha256_file(plan_dir / "plan.json")}
        sample = Sample.model_validate(
            {
                "id": f"vqa2-new-{row['question_id']}",
                "prefix": {
                    "state": "Answer the multiple-choice question using the attached image.",
                    "media": [{"type": "image", "uri": uri, "sha256": digest}],
                },
                "questions": [
                    {
                        "qid": "choice",
                        "type": "choice",
                        "instruction": row["question"],
                        "options": row["options"],
                        "target": {
                            "kind": "hard",
                            "value": chr(65 + row["options"].index(row["gold"])),
                            "source": "vqa2_unanimous_10_human_answers",
                        },
                    }
                ],
                "meta": {
                    "modality": ["image"],
                    "source_dataset": "VQA-v2",
                    "source_revision": "v2-annotations-sha256:"
                    + report["input_sha256"]["train_annotations"],
                    "source_sha256": sha256_bytes(canonical_json(source).encode()),
                    "license": LICENSE,
                    "usage": "commercial",
                    "pool": "train",
                    "family_id": row["family_id"],
                    "data_version": VERSION,
                },
            }
        )
        samples.append(sample.model_dump(mode="json"))
        provenance.append(source)
        ledger.append(
            {
                "item_id": sample.id,
                "pool": "train",
                "family_id": row["family_id"],
                "family_hash": family_hash,
                "derived_sample_sha256": canonical_sample_sha(sample),
                "source_dataset": "VQA-v2",
                "source_revision": sample.meta.source_revision,
                "license": LICENSE,
                "media_sha256": digest,
            }
        )
    output_dir.mkdir(parents=True)
    outputs = {
        name: _write(output_dir / name, rows)
        for name, rows in {
            "train.jsonl": samples,
            "allocation-ledger.jsonl": ledger,
            "provenance.jsonl": provenance,
        }.items()
    }
    result = {
        **report,
        "output_sha256": outputs,
        "download_manifest_sha256": sha256_file(download_manifest),
    }
    (output_dir / "report.json").write_text(canonical_json(result) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    for name in (
        "coco-metadata",
        "aok-train",
        "train-questions",
        "train-annotations",
        "output-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--allocation-ledger", type=Path, action="append", default=[])
    for name, default in (("seed", 42), ("color", 160), ("integer", 160), ("yes-no", 80)):
        p.add_argument("--" + name, type=int, default=default)
    p = sub.add_parser("build")
    for name in ("plan-dir", "download-manifest", "output-dir", "media-root"):
        p.add_argument("--" + name, type=Path, required=True)
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    try:
        result = (plan if mode == "plan" else build)(**args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"VQA2 new-family rejected: {exc}\n")
    print(
        canonical_json(
            {
                key: result[key]
                for key in (
                    "data_version",
                    "sample_count",
                    "category_counts",
                    "candidates_sha256",
                    "output_sha256",
                )
                if key in result
            }
        )
    )


if __name__ == "__main__":
    main()
