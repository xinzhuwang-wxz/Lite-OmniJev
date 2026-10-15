"""Private deterministic VQA gold planner/materializer; performs no network I/O.

Plan: --config ABS --output-dir ABS --report ABS. Materialize: --plan ABS
--plan-sha256 SHA --download-manifest ABS --download-sha256 SHA --media-root ABS
--output-dir ABS --report ABS. Download JSONL rows bind image_id, path, url, sha256.
Only aggregate receipts leave the private job; canaries never admit real gold.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from data.schema import Sample
from scripts.clevrer_gold_pools import encode, write_private
from scripts.gold_inventory import Manifest, PrivateParser, bound_lines, digest
from scripts.s1_text_pools import canonical_family_hash
from scripts.vqa2_gold_capacity import PINS, archive, bound, measure, source_id
from scripts.vqa2_new_family import MEDIA_LICENSE
from scripts.vqa2_train_augmentation import COLORS, LICENSE, normalize

VERSION = "vqa2-gold-pools-v1"
QUOTAS = {"calibration": 3000, "holdout": 1000, "dev": 500}


def safe_path(path: Path) -> None:
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("unsafe_path")


def official_url(iid: int) -> str:
    return f"https://images.cocodataset.org/train2017/{iid:012d}.jpg"


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    write_private(path, b"".join(encode(row) for row in rows))
    return digest(path.read_bytes())


def exclusions(config: dict[str, Any]) -> tuple[set[str], set[str], list[tuple[str, str]]]:
    inventory = Manifest.model_validate_json(bound(config["inventory"]))
    families, media, historical = set(), set(), []
    for artifact in inventory.samples:
        for line in bound_lines(artifact):
            sample = Sample.model_validate_json(line)
            families.add(canonical_family_hash(sample.meta.family_id))
            for item in sample.prefix.media:
                media.add(item.sha256)
                if item.type == "image":
                    historical.append((item.uri, item.sha256))
    for component_artifact in inventory.component_ledgers:
        for line in bound_lines(component_artifact):
            row = json.loads(line)
            for key in ("image", "audio"):
                families.update(
                    (row[key]["family_hash"], canonical_family_hash(row[key]["family_id"]))
                )
                media.add(row[key]["media_sha256"])
    quarantine = json.loads(bound(config["quarantine"]))
    families.update(quarantine["families"])
    media.update(quarantine["media"])
    return families, media, historical


def choose(
    config: dict[str, Any], quotas: dict[str, int], canary: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for key in PINS:
        safe_path(Path(config[key]["path"]))
    for root in config["media_roots"]:
        safe_path(Path(root))
    measured = measure(config, synthetic_canary=canary)
    families, media, _ = exclusions(config)
    aok = {source_id(r["image_id"]) for r in json.loads(bound(config["aok_train"]))}
    metadata = json.loads(bound(config["coco_metadata"]))
    images = {
        source_id(r["id"]): r
        for r in metadata["images"]
        if source_id(r["id"]) not in aok
        and canonical_family_hash(f"coco:{source_id(r['id'])}") not in families
        and source_id(r["license"]) == 4
        and r["file_name"] == f"{source_id(r['id']):012d}.jpg"
        and r.get("coco_url")
        in {
            official_url(source_id(r["id"])),
            official_url(source_id(r["id"])).replace("https:", "http:"),
        }
    }
    for iid in list(images):
        if any(
            (p := Path(root) / images[iid]["file_name"]).is_file()
            and digest(p.read_bytes()) in media
            for root in config["media_roots"]
        ):
            del images[iid]
    annotations = {
        source_id(r["question_id"]): r
        for r in archive(
            bound(config["train_annotations"]),
            "v2_mscoco_train2014_annotations.json",
            "annotations",
        )
        if source_id(r["image_id"]) in images
    }
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for q in archive(
        bound(config["train_questions"]),
        "v2_OpenEnded_mscoco_train2014_questions.json",
        "questions",
    ):
        iid, qid = source_id(q["image_id"]), source_id(q["question_id"])
        if iid not in images:
            continue
        ann = annotations[qid]
        gold, question = normalize(ann["multiple_choice_answer"]), normalize(q["question"])
        votes = [normalize(a["answer"]) for a in ann["answers"]]
        category = "unsupported"
        if gold in {"yes", "no"}:
            category = "yes_no"
        elif re.fullmatch(r"(?:[0-9]|1[0-9]|20)", gold):
            category = "integer"
        elif gold in COLORS and re.search(
            r"\bcolou?r\b", question + " " + normalize(ann.get("question_type", ""))
        ):
            category = "color"
        grouped[(iid, question)].append(
            {
                "image_id": iid,
                "question_id": qid,
                "question": q["question"],
                "gold": gold,
                "answers": ann["answers"],
                "category": category,
                "valid": bool(question and gold) and len(votes) == 10 and set(votes) == {gold},
            }
        )
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in grouped.values():
        if (
            all(r["valid"] for r in rows)
            and len({r["gold"] for r in rows}) == len({r["category"] for r in rows}) == 1
        ):
            row = min(rows, key=lambda r: r["question_id"])
            buckets[row["category"]].append(row)
    selected, used = [], set()
    # Allocate scarce categories globally before common yes/no families.
    for category, numerator in (("color", 1), ("integer", 1), ("yes_no", 3)):
        ordered = sorted(
            buckets[category],
            key=lambda r: digest(f"42:{r['image_id']}:{r['question_id']}".encode()),
        )
        for pool in QUOTAS:
            total = quotas[pool]
            needed = total * numerator // 5
            chosen = [r for r in ordered if r["image_id"] not in used][:needed]
            if len(chosen) != needed:
                raise ValueError("insufficient_capacity")
            for row in chosen:
                iid, qid, gold = row["image_id"], row["question_id"], row["gold"]
                used.add(iid)
                rng = random.Random(f"42:{iid}:{qid}")
                domain = list(COLORS) if category == "color" else [str(i) for i in range(21)]
                options = (
                    ["yes", "no"]
                    if category == "yes_no"
                    else [gold] + rng.sample([x for x in domain if x != gold], 3)
                )
                rng.shuffle(options)
                selected.append(
                    {
                        **row,
                        "pool": pool,
                        "role": "supplemental_dev" if pool == "dev" else "gold_reserve",
                        "options": options,
                        "url": official_url(iid),
                        "family_id": f"coco:{iid}",
                        "attribution": images[iid],
                        "source_gold_sha256": digest(encode(annotations[qid])),
                    }
                )
    return selected, measured


def plan(
    config: dict[str, Any],
    output: Path,
    *,
    canary: bool = False,
    quotas: dict[str, int] | None = None,
) -> dict[str, Any]:
    quotas = QUOTAS if quotas is None else quotas
    if (
        set(quotas) != set(QUOTAS)
        or any(type(n) is not int or n < 5 or n % 5 for n in quotas.values())
        or (not canary and quotas != QUOTAS)
    ):
        raise ValueError("invalid_quota")
    safe_path(output)
    if output.exists():
        raise ValueError("output_exists")
    selected, measured = choose(config, quotas, canary)
    output.mkdir(mode=0o700)
    candidates_sha = write_rows(output / "candidates.jsonl", selected)
    requests_sha = write_rows(
        output / "download-requests.jsonl",
        [
            {"image_id": r["image_id"], "url": r["url"], "file_name": f"{r['image_id']:012d}.jpg"}
            for r in selected
        ],
    )
    private = {
        "schema_version": VERSION,
        "config": config,
        "canary": canary,
        "quotas": quotas,
        "candidates_sha256": candidates_sha,
        "requests_sha256": requests_sha,
        "historical_artifact_sha256": measured["historical_artifact_sha256"],
    }
    write_private(output / "plan.json", encode(private))
    return {
        "counts": dict(Counter(r["pool"] for r in selected)),
        "category_counts": dict(Counter(r["category"] for r in selected)),
        "input_sha256": measured["input_sha256"],
        "plan_sha256": digest((output / "plan.json").read_bytes()),
    }


def rgb(path: Path, expected: str, *, historical: bool = False) -> str:
    Image = importlib.import_module("PIL.Image")

    safe_path(path)
    if digest(path.read_bytes()) != expected:
        raise ValueError("media_hash")
    with Image.open(path) as image:
        if image.format not in ({"JPEG", "PNG"} if historical else {"JPEG"}):
            raise ValueError("media_format")
        image.load()
        decoded = image.convert("RGB")
        return digest(encode(list(decoded.size)) + decoded.tobytes())


def materialize(
    plan_path: Path,
    plan_sha: str,
    downloads_path: Path,
    downloads_sha: str,
    media_root: Path,
    output: Path,
) -> dict[str, Any]:
    for path in (plan_path, downloads_path, media_root, output):
        safe_path(path)
    if output.exists() or not output.is_relative_to(media_root) or not media_root.is_dir():
        raise ValueError("output_path")
    private = json.loads(bound({"path": str(plan_path), "sha256": plan_sha}))
    if private["schema_version"] != VERSION:
        raise ValueError("plan_version")
    rows, measured = choose(private["config"], private["quotas"], private["canary"])
    if (
        measured["historical_artifact_sha256"] != private["historical_artifact_sha256"]
        or digest(b"".join(encode(r) for r in rows)) != private["candidates_sha256"]
    ):
        raise ValueError("plan_mutation")
    bound(
        {"path": str(plan_path.parent / "candidates.jsonl"), "sha256": private["candidates_sha256"]}
    )
    downloads = [
        json.loads(line)
        for line in bound({"path": str(downloads_path), "sha256": downloads_sha}).splitlines()
    ]
    by_id = {source_id(r["image_id"]): r for r in downloads}
    if len(by_id) != len(downloads) or set(by_id) != {r["image_id"] for r in rows}:
        raise ValueError("download_membership")
    _, seen_raw, historical = exclusions(private["config"])
    seen_rgb: set[str] = set()
    roots = [Path(p) for p in private["config"]["media_roots"]]
    for uri, sha in set(historical):
        parsed = urlparse(uri)
        direct = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(uri)
        paths = [direct] if direct.is_absolute() else [root / direct for root in roots]
        matches = [p for p in paths if p.is_file() and digest(p.read_bytes()) == sha]
        if not matches:
            raise ValueError("historical_rgb_unavailable")
        seen_rgb.add(rgb(matches[0], sha, historical=True))
    verified = []
    failures: Counter[str] = Counter()
    for row in rows:
        try:
            download = by_id[row["image_id"]]
            if (
                download["url"] != official_url(row["image_id"])
                or download.get("final_url", download["url"]) != download["url"]
            ):
                raise ValueError("download_url")
            path, sha = Path(download["path"]), download["sha256"]
            decoded = rgb(path, sha)
            if sha in seen_raw or decoded in seen_rgb:
                raise ValueError("media_duplicate")
            seen_raw.add(sha)
            seen_rgb.add(decoded)
            verified.append((row, path, sha, decoded))
        except Exception:
            failures["media_validation_failed"] += 1
    if failures:
        return {"status": "rejected", "counts": dict(failures), "exit_code": 1}
    output.mkdir(mode=0o700)
    (output / "media").mkdir(mode=0o700)
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    aliases = []
    for row, path, sha, decoded in verified:
        destination = output / "media" / f"{row['image_id']:012d}.jpg"
        payload = path.read_bytes()
        if digest(payload) != sha:
            raise ValueError("media_changed")
        write_private(destination, payload)
        source = {
            **row,
            "media_sha256": sha,
            "rgb_sha256": decoded,
            "plan_sha256": plan_sha,
            "source_revision": private["config"]["train_annotations"]["sha256"],
            "license": LICENSE,
            "media_license": MEDIA_LICENSE,
        }
        sample = Sample.model_validate(
            {
                "id": f"vqa2-gold-{row['question_id']}",
                "prefix": {
                    "state": "Answer the multiple-choice question using the attached image.",
                    "media": [
                        {
                            "type": "image",
                            "uri": str(destination.relative_to(media_root)),
                            "sha256": sha,
                        }
                    ],
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
                    "source_revision": "v2-annotations-sha256:" + source["source_revision"],
                    "source_sha256": digest(encode(source)),
                    "license": LICENSE,
                    "usage": "research" if private["canary"] else "commercial",
                    "pool": row["pool"],
                    "family_id": row["family_id"],
                    "data_version": VERSION + ("-canary" if private["canary"] else ""),
                },
            }
        )
        samples[row["pool"]].append(sample.model_dump(mode="json"))
        aliases.append(
            {
                **source,
                "item_id": sample.id,
                "family_hash": canonical_family_hash(row["family_id"]),
                "original_question_sha256": digest(row["question"].encode()),
            }
        )
    inventory = json.loads(bound(private["config"]["inventory"]))
    outputs = {}
    for pool, values in samples.items():
        path = output / f"{pool}.jsonl"
        sha = write_rows(path, values)
        outputs[pool] = sha
        inventory["samples"].append(
            {
                "path": str(path),
                "sha256": sha,
                "cohort": "image",
                "exposure": "current",
                "expected_pool": pool,
            }
        )
    outputs["aliases"] = write_rows(output / "alias-ledger.jsonl", aliases)
    write_private(output / "inventory-manifest.json", encode(inventory))
    outputs["inventory"] = digest((output / "inventory-manifest.json").read_bytes())
    return {
        "synthetic_canary": private["canary"],
        "counts": {k: len(v) for k, v in samples.items()},
        "output_sha256": outputs,
        "download_manifest_sha256": downloads_sha,
    }


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": VERSION,
        "status": "rejected",
        "failure": "job_failed",
        "exit_code": 1,
        "full_gold_gate": "not_validated",
        "private_job_permissions": "not_validated",
        "inventory_completeness": "not_validated",
        "seed": 42,
        "network_bytes": 0,
        "cost_usd": None,
    }
    args = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            parser = PrivateParser(add_help=False)
            parser.add_argument("mode", choices=("plan", "materialize"))
            for name in (
                "config",
                "plan",
                "download-manifest",
                "media-root",
                "output-dir",
                "report",
            ):
                parser.add_argument(
                    "--" + name, type=Path, required=name in {"output-dir", "report"}
                )
            for name in ("plan-sha256", "download-sha256"):
                parser.add_argument("--" + name)
            parser.add_argument("--synthetic-canary", action="store_true")
            for pool, count in QUOTAS.items():
                parser.add_argument(f"--{pool}-count", type=int, default=count)
            args = parser.parse_args(argv)
            safe_path(args.report)
            if args.report.exists():
                raise ValueError("report_exists")
            if args.mode == "plan":
                safe_path(args.config)
                result.update(
                    plan(
                        json.loads(args.config.read_bytes()),
                        args.output_dir,
                        canary=args.synthetic_canary,
                        quotas={pool: getattr(args, pool + "_count") for pool in QUOTAS},
                    )
                )
                result["synthetic_canary"] = args.synthetic_canary
            else:
                result.update(
                    materialize(
                        args.plan,
                        args.plan_sha256,
                        args.download_manifest,
                        args.download_sha256,
                        args.media_root,
                        args.output_dir,
                    )
                )
            if result.get("counts", {}).get("media_validation_failed", 0) == 0:
                result.update(
                    status="planned" if args.mode == "plan" else "materialized",
                    failure=None,
                    exit_code=0,
                )
    except Exception:
        pass
    result["elapsed_seconds"] = round(time.monotonic() - started, 6)
    if args is not None:
        try:
            write_private(args.report, encode(result))
        except Exception:
            result.update(status="rejected", failure="report_io", exit_code=1)
    print(json.dumps(result, sort_keys=True))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
