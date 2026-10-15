"""Private aggregate-only VQA2 capacity CLI; no downloads, selection or pool writes.

Run ``python -m scripts.vqa2_gold_capacity --config ABS`` inside the private
launcher. Config contains {path, sha256} for each PIN key and media_roots (absolute
COCO train2017 directories). Synthetic fixtures require --synthetic-canary.
"""

from __future__ import annotations

import contextlib
import io
import json
import platform
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from data.schema import Sample
from scripts.gold_inventory import (
    Artifact,
    ComponentRecord,
    Manifest,
    PrivateParser,
    audit_manifest,
    bound_lines,
    digest,
    extract,
    link_components,
)
from scripts.s1_text_pools import canonical_family_hash
from scripts.vqa2_new_family import MEDIA_LICENSE
from scripts.vqa2_train_augmentation import COLORS, normalize

PINS = {
    "inventory": "0d583307d050423ab40c34ba129aefd7a7345355992367e0518f07083e6eba6a",
    "quarantine": "8901483c170a7b1b1d04cf339eb4fbc6fc9e9722b8c2beb9cde0dc56b437ba7f",
    "coco_metadata": "4b62086319480e0739ef390d04084515defb9c213ff13605a036061e33314317",
    "aok_train": "94820c4452d29102365647598c54608d4f48f56bee01b514c996a09cc0972029",
    "train_annotations": "fb101bcefe91422c543c2bb6d70af11eb3119d0ff745ae283d09acdf66250853",
    "train_questions": "05a64b6e2582d06d7585f5429674a9a33851878be1bff9f8668cdcf792df611e",
}
CATEGORIES = ("yes_no", "integer", "color")


def bound(spec: dict[str, Any]) -> bytes:
    artifact = Artifact.model_validate(spec)
    path = Path(artifact.path)
    if path.is_symlink():
        raise ValueError("input")
    content = path.read_bytes()
    if digest(content) != artifact.sha256:
        raise ValueError("digest")
    return content


def source_id(value: Any) -> int:
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError("identity")
    return int(value)


def archive(content: bytes, member: str, key: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(content)) as handle:
        if handle.namelist().count(member) != 1:
            raise ValueError("archive")
        result: list[dict[str, Any]] = json.loads(handle.read(member))[key]
        return result


def measure(config: dict[str, Any], *, synthetic_canary: bool = False) -> dict[str, Any]:
    if set(config) != {*PINS, "media_roots"}:
        raise ValueError("config")
    if not synthetic_canary and any(config[k]["sha256"] != v for k, v in PINS.items()):
        raise ValueError("pins")
    inputs = {key: bound(config[key]) for key in PINS}
    manifest = Manifest.model_validate_json(inputs["inventory"])
    receipt = audit_manifest(Path(config["inventory"]["path"]))
    if receipt.outcome != "isolated" or receipt.manifest_sha256 != digest(inputs["inventory"]):
        raise ValueError("inventory")
    samples, rows = [], []
    for artifact in manifest.samples:
        for line in bound_lines(artifact):
            sample = Sample.model_validate_json(line)
            samples.append(sample)
            rows.append(extract(sample, artifact))
    records = [
        ComponentRecord.model_validate_json(line)
        for ledger in manifest.component_ledgers
        for line in bound_lines(ledger)
    ]
    link_components(samples, rows, records)
    families = {value for row in rows for value in row.original_family_sha256}
    media = {value for row in rows for value in row.media_sha256}
    for record in records:
        for component in (record.image, record.audio):
            families.update((component.family_hash, canonical_family_hash(component.family_id)))
            media.add(component.media_sha256)
    quarantine = json.loads(inputs["quarantine"])
    if set(quarantine) != {"families", "media"} or any(
        not isinstance(quarantine[k], list)
        or any(
            not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in quarantine[k]
        )
        for k in quarantine
    ):
        raise ValueError("quarantine")
    families.update(quarantine["families"])
    media.update(quarantine["media"])
    roots = [Path(p) for p in config["media_roots"]]
    if not all(p.is_absolute() and p.is_dir() for p in roots):
        raise ValueError("roots")
    aok = {source_id(r["image_id"]) for r in json.loads(inputs["aok_train"])}
    metadata = json.loads(inputs["coco_metadata"])
    licenses = [r for r in metadata["licenses"] if source_id(r["id"]) == 4]
    if len(licenses) != 1 or licenses[0]["url"] != MEDIA_LICENSE:
        raise ValueError("license")
    counts = dict.fromkeys(
        (
            "excluded_families",
            "excluded_license_or_metadata",
            "history_media_collisions",
            "conflicting_or_nonunanimous_questions",
            "unsupported_questions",
            "duplicate_questions",
        ),
        0,
    )
    images: dict[int, dict[str, Any]] = {}
    seen_images: set[int] = set()
    for row in metadata["images"]:
        iid = source_id(row["id"])
        if iid in seen_images:
            raise ValueError("duplicate_image")
        seen_images.add(iid)
        if iid in aok or canonical_family_hash(f"coco:{iid}") in families:
            counts["excluded_families"] += 1
            continue
        filename = f"{iid:012d}.jpg"
        if (
            source_id(row["license"]) != 4
            or row["file_name"] != filename
            or row.get("coco_url")
            not in {
                f"http://images.cocodataset.org/train2017/{filename}",
                f"https://images.cocodataset.org/train2017/{filename}",
            }
        ):
            counts["excluded_license_or_metadata"] += 1
            continue
        images[iid] = row
    annotations = {}
    for row in archive(
        inputs["train_annotations"], "v2_mscoco_train2014_annotations.json", "annotations"
    ):
        if source_id(row["image_id"]) not in images:
            continue
        qid = source_id(row["question_id"])
        if qid in annotations:
            raise ValueError("duplicate_annotation")
        annotations[qid] = row
    grouped: dict[tuple[int, str], list[tuple[str, bool, str]]] = defaultdict(list)
    seen: set[int] = set()
    for q in archive(
        inputs["train_questions"], "v2_OpenEnded_mscoco_train2014_questions.json", "questions"
    ):
        iid = source_id(q["image_id"])
        if iid not in images:
            continue
        qid = source_id(q["question_id"])
        if qid in seen:
            raise ValueError("duplicate_question")
        seen.add(qid)
        ann = annotations.get(qid)
        if ann is None or source_id(ann["image_id"]) != iid:
            raise ValueError("join")
        gold = normalize(ann["multiple_choice_answer"])
        answers = [normalize(a["answer"]) for a in ann["answers"]]
        question = normalize(q["question"])
        valid = bool(question and gold) and len(answers) == 10 and set(answers) == {gold}
        category = "unsupported"
        if gold in {"yes", "no"}:
            category = "yes_no"
        elif re.fullmatch(r"(?:[0-9]|1[0-9]|20)", gold):
            category = "integer"
        elif gold in COLORS and re.search(
            r"\bcolou?r\b", question + " " + normalize(ann.get("question_type", ""))
        ):
            category = "color"
        grouped[(iid, question)].append((gold, valid, category))
    eligible: dict[int, list[str]] = defaultdict(list)
    for (iid, _), values in grouped.items():
        if not all(v[1] for v in values) or len({v[0] for v in values}) != 1:
            counts["conflicting_or_nonunanimous_questions"] += 1
            continue
        # Different annotation question types cannot manufacture category eligibility.
        categories = {v[2] for v in values}
        if len(categories) != 1 or values[0][2] not in CATEGORIES:
            counts["unsupported_questions"] += 1
            continue
        counts["duplicate_questions"] += len(values) - 1
        eligible[iid].append(values[0][2])
    available: dict[int, str] = {}
    for iid in list(eligible):
        hashes = {
            digest(p.read_bytes())
            for root in roots
            if (p := root / images[iid]["file_name"]).is_file()
        }
        if len(hashes) > 1:
            raise ValueError("media_conflict")
        if hashes & media:
            counts["history_media_collisions"] += 1
            del eligible[iid]
        elif hashes:
            available[iid] = next(iter(hashes))
    category_counts = {
        c: {
            "originals": sum(v.count(c) for v in eligible.values()),
            "families": sum(c in v for v in eligible.values()),
        }
        for c in CATEGORIES
    }
    return {
        "counts": {
            **counts,
            "eligible_originals": sum(map(len, eligible.values())),
            "eligible_families": len(eligible),
            "local_media_families": len(available),
            "missing_media_families": len(eligible) - len(available),
            "local_unique_media": len(set(available.values())),
        },
        "categories": category_counts,
        "metadata_family_capacity_3166": len(eligible) >= 3166,
        "metadata_family_capacity_4000": len(eligible) >= 4000,
        "local_byte_distinct_family_capacity_4000": len(set(available.values())) >= 4000,
        "allocation_feasibility": "pending_media_decode_and_lineage",
        "historical_decoded_rgb_aliases": "not_available_from_inventory",
        "input_sha256": {k: digest(v) for k, v in inputs.items()},
        "historical_artifact_sha256": receipt.verified_artifact_sha256,
        "local_media_inventory_sha256": digest("\n".join(sorted(available.values())).encode()),
    }


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": "vqa2-gold-capacity-v1",
        "status": "rejected",
        "failure": "job_failed",
        "exit_code": 1,
        "gpu_used": False,
        "network_bytes": 0,
        "pool_assignment_written": False,
        "gold_admission": "not_validated",
        "permissions": "not_validated",
        "inventory_completeness": "not_validated",
        "python": platform.python_version(),
        "cpu_machine": platform.machine(),
        "operating_system": platform.system(),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "seed": 42,
    }
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            parser = PrivateParser(add_help=False)
            parser.add_argument("--config", required=True, type=Path)
            parser.add_argument("--synthetic-canary", action="store_true")
            args = parser.parse_args(argv)
            result["synthetic_canary"] = args.synthetic_canary
            raw = args.config.read_bytes()
            result["config_sha256"] = digest(raw)
            result.update(measure(json.loads(raw), synthetic_canary=args.synthetic_canary))
            result.update(status="measured", failure=None, exit_code=0)
    except Exception:
        pass
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, sort_keys=True))
    return int(result["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
