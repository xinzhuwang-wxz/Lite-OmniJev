"""Build the first public image/video development cohorts from TRAIN media sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from data.schema import Sample
from eval.sealed_gate import DevelopmentRecord
from train.multimodal import load_media

AOK_DATASET = "A-OKVQA"
AOK_REVISION = "42e8c44ff40d13cf272776a3197365f582137772"
AOK_QA_LICENSE = "Apache-2.0"
AOK_MEDIA_LICENSE = "CC-BY-2.0"
AOK_TAR_SHA256 = "3992b488babc0c1147f0def18c7a55274aeeb37ab668cf80226b8f62ee35a8e1"
AOK_MANIFEST_SHA256 = "e20747c1da29c44f93ff6fe9f1e300745234ab7926916773beb6e1ab7642a1dd"
AOK_TRAIN_MEMBER = "aokvqa_v1p0_train.json"

CLEVRER_DATASET = "CLEVRER"
CLEVRER_REVISION = (
    "official-train-json-sha11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
)
CLEVRER_LICENSE = "CC0-1.0"
CLEVRER_QUESTIONS_SHA256 = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
CLEVRER_VIDEO_ZIP_SHA256 = "c7401bc86f7ff230bda3b193619f41a79828bed1f0890948b16dd57c89dfdd3f"

SEED = 42
DEV_COUNT = 500
IMAGE_DATA_VERSION = "aokvqa-train-image-choice-dev-v1"
VIDEO_DATA_VERSION = "clevrer-train-video-noul-dev-v1"
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_VIDEO_BYTES = 64 * 1024 * 1024
MAX_VIDEO_SECONDS = 60.0
VIDEO_FRAMES = 16


class MediaDevError(ValueError):
    """User-correctable input or output contract failure."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def script_sha256() -> str:
    return sha256_file(Path(__file__))


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def allocation_prefix(mode: Literal["image", "video"]) -> str:
    return f"s0-{mode}-dev-v1:{SEED}:"


def allocation_key(mode: Literal["image", "video"], family_id: str) -> str:
    return sha256_bytes(f"{allocation_prefix(mode)}{family_id}".encode())


def row_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def validate_output_paths(paths: list[Path], media_out_dir: Path) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise MediaDevError("output paths must be unique")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise MediaDevError("refusing to overwrite existing outputs")
    if media_out_dir.exists() or media_out_dir.is_symlink():
        raise MediaDevError("refusing to overwrite media output directory")


def validate_dev_count(*, expected_count: int, dev_count: int) -> None:
    if expected_count <= 1:
        raise MediaDevError("expected_count must leave non-dev families")
    if not 1 <= dev_count <= expected_count - 1:
        raise MediaDevError("dev_count must be between 1 and expected_count - 1")


def safe_relative(parts: str) -> Path:
    relative = Path(parts)
    if ":" in parts or relative.is_absolute() or ".." in relative.parts or "\x00" in parts:
        raise MediaDevError("unsafe relative media path")
    if any(part != PurePosixPath(part).name for part in relative.parts):
        raise MediaDevError("unsafe relative media path")
    return relative


def safe_copy(src: Path, root: Path, dst_root: Path, relative: Path) -> Path:
    if root.is_symlink() or root.absolute() != root.resolve() or not root.is_dir():
        raise MediaDevError("asset root must be a real directory")
    current = root
    for part in src.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise MediaDevError("media symlinks are forbidden")
    if not src.is_file() or not src.resolve().is_relative_to(root.resolve()):
        raise MediaDevError("media must be a regular file inside asset root")
    out = dst_root / relative
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.is_symlink():
        raise MediaDevError("duplicate output media path")
    shutil.copyfile(src, out)
    return out


def decode_image(path: Path) -> dict[str, Any]:
    image_module = import_module("PIL.Image")
    try:
        with image_module.open(path) as source:
            source.load()
            width, height = source.size
            fmt = source.format
            rgb_sha = sha256_bytes(source.convert("RGB").tobytes())
    except (OSError, ValueError) as exc:
        raise MediaDevError("image decode failed") from exc
    if width <= 0 or height <= 0 or width * height > 40_000_000:
        raise MediaDevError("image dimensions invalid or too large")
    return {"width": width, "height": height, "format": fmt, "rgb_sha256": rgb_sha}


def decode_video(path: Path) -> dict[str, Any]:
    decoded = load_media(path, "video", num_frames=VIDEO_FRAMES, max_seconds=MAX_VIDEO_SECONDS)
    metadata = dict(decoded["metadata"])
    if len(decoded["frames"]) != VIDEO_FRAMES:
        raise MediaDevError("decoded video does not contain 16 frames")
    duration = float(metadata["duration"])
    if not 0 < duration <= MAX_VIDEO_SECONDS:
        raise MediaDevError("video duration invalid or exceeds 60 seconds")
    return {
        "duration_seconds": duration,
        "fps": metadata["fps"],
        "total_num_frames": metadata["total_num_frames"],
        "sampled_frame_indices": metadata["frames_indices"],
        "decoded_frames": len(decoded["frames"]),
        "decoded_width": metadata["width"],
        "decoded_height": metadata["height"],
        "decoded_frame_rgb_sha256": [sha256_bytes(frame.tobytes()) for frame in decoded["frames"]],
    }


def load_aok_manifest(path: Path, *, expected_sha256: str) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise MediaDevError("AOKVQA image manifest sha256 mismatch")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if raw.get("split") != "train" or raw.get("coco_split") != "train":
                continue
            if raw.get("license_id") != 4:
                raise MediaDevError("AOKVQA allowed manifest contains non-CC-BY license")
            qid = str(raw.get("question_id", ""))
            file_name = str(raw.get("file_name", ""))
            if not qid or PurePosixPath(file_name).name != file_name:
                raise MediaDevError(f"invalid AOKVQA manifest row {line_no}")
            if qid in rows:
                raise MediaDevError("duplicate AOKVQA manifest question_id")
            rows[qid] = raw
    return rows


def load_aok_train(path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise MediaDevError("AOKVQA train tar sha256 mismatch")
    with tarfile.open(path, "r:gz") as archive:
        try:
            member = archive.getmember(AOK_TRAIN_MEMBER)
        except KeyError as exc:
            raise MediaDevError("AOKVQA train member missing") from exc
        stream = archive.extractfile(member)
        if stream is None:
            raise MediaDevError("AOKVQA train member unreadable")
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise MediaDevError("AOKVQA train JSON must be a list")
    return rows


def aok_candidates(
    *,
    train_rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    image_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    by_family_all: dict[str, dict[str, Any]] = {}
    for raw in train_rows:
        if raw.get("split") != "train":
            raise MediaDevError("AOKVQA tar contains non-train row in train member")
        qid = str(raw.get("question_id", ""))
        family = f"coco:{raw.get('image_id')}"
        entry = by_family_all.setdefault(
            family,
            {
                "family_id": family,
                "image_id": raw.get("image_id"),
                "pool": "train_unavailable",
                "train_qa_count": 0,
                "allowed_qa_count": 0,
                "source_row_hashes": [],
            },
        )
        entry["train_qa_count"] += 1
        entry["source_row_hashes"].append(row_hash({"question_id": raw.get("question_id")}))
        allowed = manifest.get(qid)
        if allowed is None:
            exclusions["missing_allowed_license_or_media"] += 1
            continue
        choices = raw.get("choices")
        answer_index = raw.get("correct_choice_idx")
        if (
            not isinstance(choices, list)
            or len(choices) != 4
            or len({str(choice) for choice in choices}) != 4
        ):
            exclusions["invalid_choice_options"] += 1
            continue
        if type(answer_index) is not int or not 0 <= answer_index < len(choices):
            exclusions["invalid_correct_choice_idx"] += 1
            continue
        media_path = image_root / "train" / str(allowed["file_name"])
        if not media_path.is_file():
            raise MediaDevError("AOKVQA allowed image file missing")
        if media_path.stat().st_size > MAX_IMAGE_BYTES:
            raise MediaDevError("AOKVQA image exceeds byte limit")
        image_sha = sha256_file(media_path)
        download_sha = allowed.get("sha256")
        if isinstance(download_sha, str) and download_sha and image_sha != download_sha:
            raise MediaDevError("AOKVQA image sha256 disagrees with manifest")
        attribution = (
            f"COCO license_id=4 CC-BY-2.0; coco_url={allowed.get('coco_url')}; "
            f"flickr_url={allowed.get('flickr_url')}"
        )
        entry["pool"] = "train_available"
        entry["allowed_qa_count"] += 1
        candidates.append(
            {
                "raw": raw,
                "manifest": {**allowed, "sha256": image_sha, "attribution": attribution},
                "family_id": family,
                "media_path": media_path,
            }
        )
    ledger = []
    for entry in by_family_all.values():
        ledger.append(
            {
                "family_id": entry["family_id"],
                "image_id": entry["image_id"],
                "pool": entry["pool"],
                "train_qa_count": entry["train_qa_count"],
                "allowed_qa_count": entry["allowed_qa_count"],
                "source_row_hashes_sha256": sha256_bytes(
                    canonical_json(sorted(entry["source_row_hashes"])).encode()
                ),
            }
        )
    return candidates, sorted(ledger, key=lambda row: str(row["family_id"])), dict(exclusions)


def load_clevrer_questions(path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise MediaDevError("CLEVRER train question sha256 mismatch")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 10_000:
        raise MediaDevError("CLEVRER train questions must contain 10000 scenes")
    return rows


def clevrer_video_member(scene_index: int) -> str:
    bucket_start = (scene_index // 1000) * 1000
    bucket_end = bucket_start + 1000
    return f"video_{bucket_start:05d}-{bucket_end:05d}/video_{scene_index:05d}.mp4"


def clevrer_candidates(
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    ledger: list[dict[str, Any]] = []
    seen_scenes: set[int] = set()
    for scene in scenes:
        scene_index = scene.get("scene_index")
        if type(scene_index) is not int or not 0 <= scene_index < 10_000:
            raise MediaDevError("invalid CLEVRER scene_index")
        if scene_index in seen_scenes:
            raise MediaDevError("duplicate CLEVRER scene_index")
        seen_scenes.add(scene_index)
        family = f"clevrer:{scene_index}"
        exist: list[dict[str, Any]] = []
        total_qa = 0
        for question in scene.get("questions", []):
            total_qa += 1
            if (
                question.get("question_type") == "descriptive"
                and question.get("question_subtype") == "exist"
            ):
                answer = str(question.get("answer", "")).casefold()
                if answer not in {"yes", "no"}:
                    exclusions["exist_answer_not_yes_no"] += 1
                    continue
                exist.append(question)
        if not exist:
            exclusions["scene_without_exist"] += 1
        else:
            candidates.append(
                {
                    "scene_index": scene_index,
                    "family_id": family,
                    "questions": sorted(
                        exist,
                        key=lambda row: (
                            sha256_bytes(str(row.get("question", "")).encode()),
                            int(row.get("question_id", -1)),
                        ),
                    ),
                    "qa_count": total_qa,
                    "exist_count": len(exist),
                }
            )
        ledger.append(
            {
                "family_id": family,
                "scene_index": scene_index,
                "pool": "train_available",
                "train_qa_count": total_qa,
                "eligible_exist_count": len(exist),
            }
        )
    if len(seen_scenes) != 10_000:
        raise MediaDevError("CLEVRER scene count mismatch")
    return candidates, ledger, dict(exclusions)


def choose_one_per_family(
    mode: Literal["image", "video"], candidates: list[dict[str, Any]], *, dev_count: int
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_family[str(candidate["family_id"])].append(candidate)
    validate_dev_count(expected_count=len(by_family), dev_count=dev_count)
    selected_families = sorted(
        by_family, key=lambda family: (allocation_key(mode, family), family)
    )[:dev_count]
    selected: list[dict[str, Any]] = []
    for family in selected_families:
        selected.append(
            sorted(
                by_family[family],
                key=lambda item: (
                    row_hash(item.get("raw", item.get("questions", []))),
                    str(item.get("family_id")),
                ),
            )[0]
        )
    return selected


def source_hash_for_image(candidate: dict[str, Any]) -> str:
    raw = candidate["raw"]
    manifest = candidate["manifest"]
    return sha256_bytes(
        canonical_json(
            {
                "question_id": raw["question_id"],
                "image_id": raw["image_id"],
                "question_sha256": sha256_bytes(str(raw["question"]).encode()),
                "choices_sha256": row_hash([str(choice) for choice in raw["choices"]]),
                "media_sha256": manifest["sha256"],
                "license_id": manifest["license_id"],
                "revision": AOK_REVISION,
            }
        ).encode()
    )


def image_sample(candidate: dict[str, Any], media_uri: str) -> Sample:
    raw = candidate["raw"]
    source_hash = source_hash_for_image(candidate)
    target = chr(ord("A") + int(raw["correct_choice_idx"]))
    return Sample.model_validate(
        {
            "id": f"aokvqa-train-{raw['question_id']}",
            "prefix": {
                "state": "Answer the multiple-choice question using the attached image.",
                "media": [
                    {
                        "type": "image",
                        "uri": media_uri,
                        "sha256": candidate["manifest"]["sha256"],
                    }
                ],
            },
            "questions": [
                {
                    "qid": "choice",
                    "type": "choice",
                    "instruction": str(raw["question"]),
                    "options": [str(choice) for choice in raw["choices"]],
                    "target": {
                        "kind": "hard",
                        "value": target,
                        "source": "aokvqa_correct_choice_idx",
                    },
                }
            ],
            "meta": {
                "modality": ["image"],
                "source_dataset": AOK_DATASET,
                "source_revision": AOK_REVISION,
                "source_sha256": source_hash,
                "license": f"QA:{AOK_QA_LICENSE};media:{AOK_MEDIA_LICENSE}",
                "usage": "commercial",
                "pool": "dev",
                "family_id": candidate["family_id"],
                "data_version": IMAGE_DATA_VERSION,
            },
        }
    )


def source_hash_for_video(
    candidate: dict[str, Any], question: dict[str, Any], media_sha: str
) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "scene_index": candidate["scene_index"],
                "question_id": question["question_id"],
                "question_sha256": sha256_bytes(str(question["question"]).encode()),
                "program_sha256": row_hash(question.get("program", [])),
                "media_sha256": media_sha,
                "revision": CLEVRER_REVISION,
            }
        ).encode()
    )


def video_sample(
    candidate: dict[str, Any], question: dict[str, Any], *, media_uri: str, media_sha: str
) -> Sample:
    source_hash = source_hash_for_video(candidate, question, media_sha)
    return Sample.model_validate(
        {
            "id": f"clevrer-train-{candidate['scene_index']:05d}-{question['question_id']}",
            "prefix": {
                "state": "Answer the yes/no question using the attached video.",
                "media": [{"type": "video", "uri": media_uri, "sha256": media_sha}],
            },
            "questions": [
                {
                    "qid": "noul",
                    "type": "noul",
                    "instruction": str(question["question"]),
                    "target": {
                        "kind": "hard",
                        "value": str(question["answer"]).casefold(),
                        "source": "clevrer_descriptive_exist_answer",
                    },
                }
            ],
            "meta": {
                "modality": ["video"],
                "source_dataset": CLEVRER_DATASET,
                "source_revision": CLEVRER_REVISION,
                "source_sha256": source_hash,
                "license": CLEVRER_LICENSE,
                "usage": "commercial",
                "pool": "dev",
                "family_id": candidate["family_id"],
                "data_version": VIDEO_DATA_VERSION,
            },
        }
    )


def development_record_for(
    sample: Sample, modality: Literal["image", "video"]
) -> DevelopmentRecord:
    return DevelopmentRecord(
        item_id=sample.id,
        modality=modality,
        family_hash=sha256_bytes(sample.meta.family_id.encode()),
        source_hash=sample.meta.source_sha256,
        source_dataset=sample.meta.source_dataset,
        source_revision=sample.meta.source_revision,
        license=sample.meta.license,
        label_origin="hard_gold",
    )


def write_jsonl(
    path: Path, rows: list[Sample] | list[DevelopmentRecord] | list[dict[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        if isinstance(row, Sample | DevelopmentRecord):
            lines.append(row.model_dump_json())
        else:
            lines.append(canonical_json(row))
    payload = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(payload, encoding="utf-8")
    return sha256_bytes(payload.encode())


def mark_selected(
    ledger: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    families = {str(item["family_id"]) for item in selected}
    marked: list[dict[str, Any]] = []
    for row in ledger:
        pool = "dev_excluded" if row["family_id"] in families else row["pool"]
        marked.append({**row, "pool": pool})
    return sorted(marked, key=lambda row: str(row["family_id"]))


def extract_zip_member(
    archive: zipfile.ZipFile, member: str, dst_root: Path, relative: Path
) -> tuple[Path, bytes]:
    info = archive.getinfo(member)
    if info.is_dir() or info.file_size <= 0 or info.file_size > MAX_VIDEO_BYTES:
        raise MediaDevError("CLEVRER video member size invalid")
    data = archive.read(info)
    out = dst_root / relative
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.is_symlink():
        raise MediaDevError("duplicate output media path")
    out.write_bytes(data)
    return out, data


def receipt_video_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for file_info in payload.get("files", []):
        if file_info.get("relative_path") == "videos/train/video_train.zip":
            sha = file_info.get("local_sha256")
            if sha != CLEVRER_VIDEO_ZIP_SHA256:
                raise MediaDevError("CLEVRER receipt video zip sha256 mismatch")
            return str(sha)
    raise MediaDevError("CLEVRER receipt missing train video zip")


def build_image(args: argparse.Namespace, *, started_at: float) -> tuple[dict[str, Any], int]:
    manifest = load_aok_manifest(
        args.aok_manifest, expected_sha256=args.expected_aok_manifest_sha256
    )
    train_rows = load_aok_train(args.aok_tar, expected_sha256=args.expected_aok_tar_sha256)
    candidates, ledger, exclusions = aok_candidates(
        train_rows=train_rows, manifest=manifest, image_root=args.image_root
    )
    selected = choose_one_per_family("image", candidates, dev_count=args.dev_count)
    samples: list[Sample] = []
    media_rows: list[dict[str, Any]] = []
    for candidate in selected:
        relative = safe_relative(f"image/{candidate['manifest']['file_name']}")
        copied = safe_copy(candidate["media_path"], args.image_root, args.media_out_dir, relative)
        if sha256_file(copied) != candidate["manifest"]["sha256"]:
            raise MediaDevError("copied image sha256 mismatch")
        decoded = decode_image(copied)
        sample = image_sample(candidate, relative.as_posix())
        samples.append(sample)
        media_rows.append(
            {
                "item_id": sample.id,
                "family_id": sample.meta.family_id,
                "image_id": candidate["raw"]["image_id"],
                "question_id": candidate["raw"]["question_id"],
                "uri": relative.as_posix(),
                "source_path": str(candidate["manifest"]["path"]),
                "sha256": candidate["manifest"]["sha256"],
                "bytes": copied.stat().st_size,
                "license": AOK_MEDIA_LICENSE,
                "license_id": candidate["manifest"]["license_id"],
                "attribution": candidate["manifest"]["attribution"],
                "coco_url": candidate["manifest"].get("coco_url"),
                "flickr_url": candidate["manifest"].get("flickr_url"),
                **decoded,
            }
        )
    outputs = write_outputs(args, samples, mark_selected(ledger, selected), media_rows)
    report = base_report(args, "image", started_at)
    report.update(
        status="passed",
        exit_code=0,
        source={
            "dataset": AOK_DATASET,
            "revision": AOK_REVISION,
            "train_tar_sha256": args.expected_aok_tar_sha256,
            "manifest_sha256": args.expected_aok_manifest_sha256,
            "train_rows": len(train_rows),
            "allowed_manifest_rows": len(manifest),
            "qa_license": AOK_QA_LICENSE,
            "media_license": AOK_MEDIA_LICENSE,
        },
        split={
            "dev_items": len(samples),
            "candidate_qa": len(candidates),
            "candidate_families": len({item["family_id"] for item in candidates}),
            "all_train_families": len(ledger),
            "selected_families": len({sample.meta.family_id for sample in samples}),
            "excluded_counts": exclusions,
        },
        outputs=outputs,
    )
    return report, 0


def build_video(args: argparse.Namespace, *, started_at: float) -> tuple[dict[str, Any], int]:
    receipt_sha = receipt_video_sha(args.clevrer_receipt)
    scenes = load_clevrer_questions(
        args.clevrer_questions, expected_sha256=args.expected_clevrer_questions_sha256
    )
    candidates, ledger, exclusions = clevrer_candidates(scenes)
    selected = choose_one_per_family("video", candidates, dev_count=args.dev_count)
    samples: list[Sample] = []
    media_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(args.video_zip) as archive:
        for candidate in selected:
            question = candidate["questions"][0]
            member = clevrer_video_member(int(candidate["scene_index"]))
            relative = safe_relative(f"video/video_{candidate['scene_index']:05d}.mp4")
            extracted, data = extract_zip_member(archive, member, args.media_out_dir, relative)
            media_sha = sha256_bytes(data)
            decoded = decode_video(extracted)
            sample = video_sample(
                candidate, question, media_uri=relative.as_posix(), media_sha=media_sha
            )
            samples.append(sample)
            media_rows.append(
                {
                    "item_id": sample.id,
                    "family_id": sample.meta.family_id,
                    "scene_index": candidate["scene_index"],
                    "question_id": question["question_id"],
                    "uri": relative.as_posix(),
                    "zip_member": member,
                    "zip_member_crc": archive.getinfo(member).CRC,
                    "sha256": media_sha,
                    "bytes": len(data),
                    "license": CLEVRER_LICENSE,
                    **decoded,
                }
            )
    outputs = write_outputs(args, samples, mark_selected(ledger, selected), media_rows)
    report = base_report(args, "video", started_at)
    report.update(
        status="passed",
        exit_code=0,
        source={
            "dataset": CLEVRER_DATASET,
            "revision": CLEVRER_REVISION,
            "questions_sha256": args.expected_clevrer_questions_sha256,
            "video_zip_sha256_reused_from_receipt": receipt_sha,
            "video_zip_full_hash_recomputed": False,
            "license": CLEVRER_LICENSE,
            "train_scenes": len(scenes),
        },
        split={
            "dev_items": len(samples),
            "candidate_qa": sum(item["exist_count"] for item in candidates),
            "candidate_families": len({item["family_id"] for item in candidates}),
            "all_train_families": len(ledger),
            "selected_families": len({sample.meta.family_id for sample in samples}),
            "excluded_counts": exclusions,
            "fixed_video_frames": VIDEO_FRAMES,
            "max_video_seconds": MAX_VIDEO_SECONDS,
        },
        outputs=outputs,
    )
    return report, 0


def write_outputs(
    args: argparse.Namespace,
    samples: list[Sample],
    exclusion: list[dict[str, Any]],
    media_rows: list[dict[str, Any]],
) -> dict[str, str]:
    dev_hash = write_jsonl(args.dev_out, samples)
    dev_ledger_hash = write_jsonl(
        args.dev_ledger_out,
        [development_record_for(sample, args.mode) for sample in samples],
    )
    exclusion_hash = write_jsonl(args.exclusion_ledger_out, exclusion)
    media_hash = write_jsonl(args.media_manifest_out, media_rows)
    return {
        "dev_jsonl_sha256": dev_hash,
        "development_ledger_sha256": dev_ledger_hash,
        "exclusion_ledger_sha256": exclusion_hash,
        "media_manifest_sha256": media_hash,
    }


def base_report(args: argparse.Namespace, mode: str, started_at: float) -> dict[str, Any]:
    finished_at = time.time()
    return {
        "status": "blocked",
        "exit_code": 1,
        "blocker": None,
        "command": [sys.executable, "-m", "scripts.s0_media_dev", *sys.argv[1:]],
        "protocol": f"s0-{mode}-dev-v1",
        "config": {
            "mode": mode,
            "seed": SEED,
            "allocation_prefix": allocation_prefix(mode),  # type: ignore[arg-type]
            "dev_count": args.dev_count,
            "config_sha256": sha256_bytes(
                canonical_json({"mode": mode, "seed": SEED, "dev_count": args.dev_count}).encode()
            ),
        },
        "primitive_scope": {
            "image_choice_only": mode == "image",
            "video_noul_only": mode == "video",
            "score_covered": False,
            "complete_distribution_or_quality": False,
        },
        "training_admitted": False,
        "quality_gate": "not_run",
        "gpu_used": False,
        "holdout_accessed": False,
        "cost": None,
        "real_human_kappa_verified": False,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_head": git_head(),
            "script_sha256": script_sha256(),
        },
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": finished_at - started_at,
        },
    }


def run(args: argparse.Namespace) -> int:
    started_at = time.time()
    outputs = [
        args.dev_out,
        args.dev_ledger_out,
        args.exclusion_ledger_out,
        args.media_manifest_out,
    ]
    if args.report.exists() or args.report.is_symlink():
        return 1
    if args.report.resolve(strict=False) in {path.resolve(strict=False) for path in outputs}:
        return 1
    try:
        validate_output_paths([*outputs, args.report], args.media_out_dir)
        if args.mode == "image":
            report, code = build_image(args, started_at=started_at)
        else:
            report, code = build_video(args, started_at=started_at)
    except (
        OSError,
        KeyError,
        zipfile.BadZipFile,
        tarfile.TarError,
        MediaDevError,
        ValueError,
    ) as exc:
        report = base_report(args, args.mode, started_at)
        report["blocker"] = str(exc) if isinstance(exc, MediaDevError) else type(exc).__name__
        code = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["image", "video"], required=True)
    parser.add_argument("--dev-out", type=Path, required=True)
    parser.add_argument("--dev-ledger-out", type=Path, required=True)
    parser.add_argument("--exclusion-ledger-out", type=Path, required=True)
    parser.add_argument("--media-manifest-out", type=Path, required=True)
    parser.add_argument("--media-out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dev-count", type=int, default=DEV_COUNT)
    parser.add_argument("--aok-tar", type=Path)
    parser.add_argument("--aok-manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--expected-aok-tar-sha256", default=AOK_TAR_SHA256)
    parser.add_argument("--expected-aok-manifest-sha256", default=AOK_MANIFEST_SHA256)
    parser.add_argument("--clevrer-questions", type=Path)
    parser.add_argument("--video-zip", type=Path)
    parser.add_argument("--clevrer-receipt", type=Path)
    parser.add_argument("--expected-clevrer-questions-sha256", default=CLEVRER_QUESTIONS_SHA256)
    args = parser.parse_args(argv)
    if args.mode == "image" and not (args.aok_tar and args.aok_manifest and args.image_root):
        parser.error("image mode requires --aok-tar, --aok-manifest and --image-root")
    if args.mode == "video" and not (args.clevrer_questions and args.video_zip):
        parser.error("video mode requires --clevrer-questions and --video-zip")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
