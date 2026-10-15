"""Private CPU CLEVRER validation reserve builder; export aggregate receipts only.

Run with --manifest ABS --output-dir ABS --report ABS. Manifest binds questions,
videos, readme, inventory and quarantine via {path, sha256}. Canary mode accepts
synthetic sources and explicit small counts; its receipt never admits real gold.
Permissions here are defense in depth, not an independent execution boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field

from data.schema import Contract, Sample, Sha256
from scripts.gold_inventory import (
    Artifact,
    ComponentRecord,
    Manifest,
    PrivateParser,
    Rejected,
    SampleArtifact,
    bound_lines,
    digest,
    extract,
)
from scripts.s0_media_dev import clevrer_video_member
from scripts.s1_text_pools import canonical_family_hash

PINS = {
    "questions": "fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d",
    "videos": "07f6d48bd708c80428684fcf17a7874f6c01fbb82a59b2f79ede199a63db7aae",
    "readme": "a4838cc68b79ba049b54f95def1714cdbbd1380193d7d2a5d1a85440ac862e7b",
}


class BuildManifest(Contract):
    schema_version: Literal["clevrer-gold-build-v1"]
    questions: Artifact
    videos: Artifact
    readme: Artifact
    inventory: Artifact
    quarantine: Artifact


class Quarantine(Contract):
    families: list[Sha256]
    media: list[Sha256]


class Receipt(Contract):
    schema_version: Literal["clevrer-gold-receipt-v1"] = "clevrer-gold-receipt-v1"
    outcome: Literal["built", "rejected"] = "rejected"
    mode: Literal["real", "canary"] = "real"
    real_source_verified: bool = False
    full_gold_gate: Literal["not_validated"] = "not_validated"
    inventory_completeness: Literal["not_validated"] = "not_validated"
    private_job_permissions: Literal["not_validated"] = "not_validated"
    category: str = "job_failed"
    seed: Literal[42] = 42
    source_license: Literal["CC0-1.0"] = "CC0-1.0"
    source_revision: str = "official-validation-sha256:" + PINS["questions"]
    input_sha256: dict[str, Sha256] = Field(default_factory=dict)
    output_sha256: dict[str, Sha256] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    code_sha256: str = ""
    code_commit: str = "unavailable"
    runtime: str = ""
    elapsed_seconds: float = 0.0
    cost_usd: Literal[0] = 0
    exit_code: int = 1


class BuildError(Exception):
    """Only static categories may leave the job."""


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(artifact: Artifact) -> None:
    if file_digest(Path(artifact.path)) != artifact.sha256:
        raise BuildError("hash_mismatch")


def bound_json(artifact: Artifact) -> Any:
    content = Path(artifact.path).read_bytes()
    if digest(content) != artifact.sha256:
        raise BuildError("hash_mismatch")
    return json.loads(content)


def exclusions(manifest: BuildManifest, receipt: Receipt) -> tuple[set[str], set[str]]:
    inventory = Manifest.model_validate(bound_json(manifest.inventory))
    families: set[str] = set()
    media: set[str] = set()
    for index, artifact in enumerate(inventory.samples):
        for line in bound_lines(artifact):
            sample = Sample.model_validate_json(line)
            extract(sample, artifact)  # Validate declared pools/modalities, union ALL versions.
            families.add(canonical_family_hash(sample.meta.family_id))
            media.update(item.sha256 for item in sample.prefix.media)
        receipt.input_sha256[f"inventory_sample_{index}"] = artifact.sha256
    for index, component_artifact in enumerate(inventory.component_ledgers):
        for line in bound_lines(component_artifact):
            record = ComponentRecord.model_validate_json(line)
            for component in (record.image, record.audio):
                if component.family_hash != canonical_family_hash(component.family_id):
                    raise BuildError("invalid_component")
                families.add(component.family_hash)
                media.add(component.media_sha256)
            if record.crossmodal_family_hash != canonical_family_hash(record.crossmodal_family_id):
                raise BuildError("invalid_component")
            families.add(record.crossmodal_family_hash)
        receipt.input_sha256[f"inventory_components_{index}"] = component_artifact.sha256
    quarantine = Quarantine.model_validate(bound_json(manifest.quarantine))
    families.update(quarantine.families)
    media.update(quarantine.media)
    return families, media


def rank(namespace: str, identity: str) -> str:
    return digest(f"42:{namespace}:{identity}".encode())


def candidates(rows: Any, real: bool, counts: Counter[str]) -> list[tuple[int, dict[str, Any]]]:
    if not isinstance(rows, list) or (real and len(rows) != 5000):
        raise BuildError("invalid_source")
    seen: set[int] = set()
    result: list[tuple[int, dict[str, Any]]] = []
    for scene in rows:
        index = scene["scene_index"]
        if type(index) is not int or not 10000 <= index < 15000 or index in seen:
            raise BuildError("invalid_scene_identity")
        seen.add(index)
        eligible = []
        question_ids: set[int] = set()
        for question in scene["questions"]:
            qid = question["question_id"]
            if type(qid) is not int or qid < 0 or qid in question_ids:
                raise BuildError("invalid_question_identity")
            question_ids.add(qid)
            if (
                question.get("question_type") != "descriptive"
                or question.get("question_subtype") != "exist"
            ):
                counts["non_exist_questions"] += 1
                continue
            if (
                question.get("answer") not in ("yes", "no")
                or not isinstance(question.get("question"), str)
                or not question["question"].strip()
            ):
                counts["invalid_gold_questions"] += 1
                continue
            eligible.append(question)
        if eligible:
            result.append(
                (
                    index,
                    min(eligible, key=lambda q: rank("question", f"{index}:{q['question_id']}")),
                )
            )
        else:
            counts["families_without_gold"] += 1
    if real and seen != set(range(10000, 15000)):
        raise BuildError("invalid_scene_identity")
    counts["source_families"] = len(seen)
    return sorted(result, key=lambda pair: rank("family", f"clevrer:{pair[0]}"))


def zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    basenames: set[str] = set()
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in info.filename
            or stat.S_ISLNK(mode)
            or info.flag_bits & 1
            or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))
        ):
            raise BuildError("unsafe_archive")
        if info.is_dir():
            continue
        if info.filename in members or path.name in basenames:
            raise BuildError("duplicate_archive_member")
        members[info.filename] = info
        basenames.add(path.name)
    return members


def write_private(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)


def encode(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode()


def build(args: argparse.Namespace, receipt: Receipt) -> None:
    content = args.manifest.read_bytes()
    receipt.input_sha256["build_manifest"] = digest(content)
    manifest = BuildManifest.model_validate_json(content)
    real = not args.canary
    receipt.mode = "real" if real else "canary"
    if (
        args.calibration_count < 1
        or args.holdout_count < 1
        or (real and (args.calibration_count, args.holdout_count) != (3000, 1000))
    ):
        raise BuildError("invalid_quota")
    for name in ("questions", "videos", "readme", "inventory", "quarantine"):
        artifact = getattr(manifest, name)
        if real and name in PINS and artifact.sha256 != PINS[name]:
            raise BuildError("source_not_pinned")
        verify(artifact)
        receipt.input_sha256[name] = artifact.sha256
    receipt.real_source_verified = real
    receipt.source_revision = (
        "official-validation-sha256:" if real else "synthetic-canary:"
    ) + manifest.questions.sha256
    families, media = exclusions(manifest, receipt)
    counts: Counter[str] = Counter()
    selected: list[tuple[int, dict[str, Any], str]] = []
    rows = candidates(bound_json(manifest.questions), real, counts)
    with zipfile.ZipFile(manifest.videos.path) as archive:
        members = zip_members(archive)
        for index, question in rows:
            if canonical_family_hash(f"clevrer:{index}") in families:
                counts["excluded_family"] += 1
                continue
            member = members.get(clevrer_video_member(index))
            if member is None:
                raise BuildError("missing_media")
            # Hash every eligible candidate before pool allocation; reading to EOF checks CRC.
            with archive.open(member) as handle:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
                sha = hasher.hexdigest()
            if sha in media:
                counts["excluded_media_duplicate"] += 1
                continue
            media.add(sha)
            selected.append((index, question, sha))
        counts["eligible_distinct_families"] = len(selected)
        receipt.counts = dict(counts)
        total = args.calibration_count + args.holdout_count
        if len(selected) < total:
            raise BuildError("insufficient_capacity")
        args.output_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        media_dir = args.output_dir / "media"
        media_dir.mkdir(mode=0o700)
        samples_by_pool: dict[str, list[Sample]] = {"calibration": [], "holdout": []}
        ledger: list[dict[str, Any]] = []
        for offset, (index, question, sha) in enumerate(selected[:total]):
            pool = "calibration" if offset < args.calibration_count else "holdout"
            path = media_dir / f"video_{index:05d}.mp4"
            payload = archive.read(members[clevrer_video_member(index)])
            if digest(payload) != sha:
                raise BuildError("media_changed")
            write_private(path, payload)
            family = f"clevrer:{index}"
            sample = Sample.model_validate(
                {
                    "id": f"clevrer-validation-{index:05d}-{question['question_id']}",
                    "prefix": {
                        "state": "Answer the yes/no question using the attached video.",
                        "media": [{"type": "video", "uri": path.as_uri(), "sha256": sha}],
                    },
                    "questions": [
                        {
                            "qid": "noul",
                            "type": "noul",
                            "instruction": question["question"],
                            "target": {
                                "kind": "hard",
                                "value": question["answer"],
                                "source": "clevrer_descriptive_exist_answer",
                            },
                        }
                    ],
                    "meta": {
                        "modality": ["video"],
                        "source_dataset": "CLEVRER",
                        "source_revision": receipt.source_revision,
                        "source_sha256": digest(
                            encode(
                                {
                                    "question": question,
                                    "scene_index": index,
                                    "questions_sha256": manifest.questions.sha256,
                                    "media_sha256": sha,
                                }
                            )
                        ),
                        "license": "CC0-1.0",
                        "usage": "commercial" if real else "research",
                        "pool": pool,
                        "family_id": family,
                        "data_version": "clevrer-validation-gold-v1"
                        if real
                        else "clevrer-canary-not-real-gold-v1",
                    },
                }
            )
            samples_by_pool[pool].append(sample)
            ledger.append(
                {
                    "item_id": sample.id,
                    "family_id": family,
                    "family_hash": canonical_family_hash(family),
                    "media_sha256": sha,
                    "pool": pool,
                    "question_id": question["question_id"],
                }
            )
        output_artifacts = []
        for pool, samples in samples_by_pool.items():
            path = args.output_dir / f"{pool}.jsonl"
            write_private(
                path, b"".join((sample.model_dump_json() + "\n").encode() for sample in samples)
            )
            sha = file_digest(path)
            receipt.output_sha256[pool] = sha
            receipt.counts[pool] = len(samples)
            output_artifacts.append(
                SampleArtifact.model_validate(
                    {
                        "path": str(path),
                        "sha256": sha,
                        "cohort": "video",
                        "exposure": "current",
                        "expected_pool": pool,
                    }
                )
            )
        write_private(args.output_dir / "allocation.jsonl", b"".join(encode(row) for row in ledger))
        output_manifest = Manifest(
            schema_version="gold-inventory-manifest-v1",
            samples=output_artifacts,
            component_ledgers=[],
        )
        write_private(
            args.output_dir / "inventory-manifest.json", output_manifest.model_dump_json().encode()
        )
        for name in ("allocation.jsonl", "inventory-manifest.json"):
            receipt.output_sha256[name] = file_digest(args.output_dir / name)
    receipt.outcome, receipt.category, receipt.exit_code = "built", "none", 0


def main(argv: list[str] | None = None) -> int:
    start = time.monotonic()
    receipt = Receipt(runtime=f"Python {platform.python_version()} / {platform.system()}")
    args = None
    try:
        receipt.code_sha256 = file_digest(Path(__file__))
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode == 0:
            receipt.code_commit = commit.stdout.strip()
        parser = PrivateParser(description=__doc__)
        for name in ("manifest", "output-dir", "report"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--canary", action="store_true")
        parser.add_argument("--calibration-count", type=int, default=3000)
        parser.add_argument("--holdout-count", type=int, default=1000)
        args = parser.parse_args(argv)
        receipt.mode = "canary" if args.canary else "real"
        if any(not path.is_absolute() for path in (args.manifest, args.output_dir, args.report)):
            raise BuildError("invalid_arguments")
        if (
            args.output_dir.exists()
            or args.output_dir.is_symlink()
            or args.report.exists()
            or args.report.is_symlink()
        ):
            raise BuildError("output_exists")
        build(args, receipt)
    except BuildError as error:
        receipt.category = str(error)
    except Rejected as error:
        receipt.category = error.category
    except zipfile.BadZipFile:
        receipt.category = "invalid_archive"
    except (ValueError, KeyError, TypeError):
        receipt.category = "invalid_metadata"
    except OSError:
        receipt.category = "io_error"
    except Exception:
        receipt.category = "job_failed"
    receipt.elapsed_seconds = round(time.monotonic() - start, 6)
    if args is not None and args.report.is_absolute():
        try:
            write_private(args.report, receipt.model_dump_json().encode() + b"\n")
        except Exception:
            receipt.outcome, receipt.category, receipt.exit_code = "rejected", "report_io", 1
    print(receipt.model_dump_json())
    return receipt.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
