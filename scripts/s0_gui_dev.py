"""Build the first GUIOdyssey screenshot target-location development cohort."""

from __future__ import annotations

import argparse
import binascii
import concurrent.futures
import hashlib
import importlib
import io
import json
import platform
import random
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from data.schema import Sample
from eval.sealed_gate import DevelopmentRecord
from scripts.gui_odyssey_media import REPO_ID, REVISION

DATASET = REPO_ID
DATA_VERSION = "gui-odyssey-train-screenshot-choice4-dev-v1"
SEED = 42
DEV_COUNT = 500
MAX_ANNOTATION_DOWNLOADS = 630
MAX_DOWNLOAD_WORKERS = 8
DOWNLOAD_TIMEOUT_SECONDS = 20
ALLOCATION_PREFIX = f"s0-gui-dev-v1:{SEED}:"
ANNOTATION_LICENSE = "CC-BY-4.0"
ANNOTATION_URL_PREFIX = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/annotations"


class GuiDevError(ValueError):
    """User-correctable input or output contract failure."""


Box = tuple[int, int, int, int]
Point = tuple[int, int]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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


def allocation_key(family_id: str) -> str:
    return sha256_bytes(f"{ALLOCATION_PREFIX}{family_id}".encode())


def step_key(episode_id: str, step: int) -> str:
    return sha256_bytes(f"s0-gui-step-v1:{SEED}:{episode_id}:{step}".encode())


def safe_episode_id(value: Any) -> str:
    episode_id = str(value)
    if not episode_id.isdigit() or len(episode_id) != 16:
        raise GuiDevError("unsafe episode_id")
    return episode_id


def safe_file_name(value: Any) -> str:
    name = str(value)
    if PurePosixPath(name).name != name or "\\" in name or "\x00" in name:
        raise GuiDevError("unsafe screenshot file name")
    return name


def validate_output_paths(paths: list[Path]) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise GuiDevError("output paths must be unique")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise GuiDevError("refusing to overwrite existing outputs")


def load_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("partition") != "train":
        raise GuiDevError("GUI media plan must be train-only")
    episodes = payload.get("selected_episodes")
    if not isinstance(episodes, list) or not episodes:
        raise GuiDevError("GUI media plan missing selected train episodes")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in episodes:
        if not isinstance(raw, dict):
            raise GuiDevError("invalid GUI media plan episode")
        episode_id = safe_episode_id(raw.get("episode_id"))
        if episode_id in seen:
            raise GuiDevError("duplicate episode in GUI media plan")
        seen.add(episode_id)
        if raw.get("partition") != "train":
            raise GuiDevError("non-train episode in GUI media plan")
        result.append(raw)
    return result


def selected_episode_ids(plan_rows: list[dict[str, Any]], *, needed: int, limit: int) -> list[str]:
    if limit < needed:
        raise GuiDevError("annotation download limit is below dev_count")
    ordered = sorted(
        (safe_episode_id(row["episode_id"]) for row in plan_rows),
        key=lambda episode_id: (allocation_key(episode_id), episode_id),
    )
    return ordered[:limit]


def annotation_url(episode_id: str) -> str:
    return f"{ANNOTATION_URL_PREFIX}/{episode_id}.json"


def download_one_annotation(episode_id: str, annotation_root: Path) -> dict[str, Any]:
    url = annotation_url(episode_id)
    out = annotation_root / f"{episode_id}.json"
    receipt: dict[str, Any] = {
        "episode_id": episode_id,
        "url": url,
        "revision": REVISION,
        "result": "pending",
    }
    if out.exists() and not out.is_symlink():
        payload = out.read_bytes()
        receipt.update(
            result="present",
            bytes=len(payload),
            sha256=sha256_bytes(payload),
        )
        return receipt
    request = urllib.request.Request(url, headers={"User-Agent": "lite-omnijev-gui-dev"})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = response.read()
        parsed = json.loads(payload.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise GuiDevError("annotation payload is not an object")
        annotation_root.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        receipt.update(result="downloaded", bytes=len(payload), sha256=sha256_bytes(payload))
    except (
        OSError,
        TimeoutError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        GuiDevError,
    ) as exc:
        receipt.update(result="failed", error_type=type(exc).__name__)
    return receipt


def ensure_annotations(
    episode_ids: list[str], annotation_root: Path, *, download_missing: bool
) -> list[dict[str, Any]]:
    annotation_root.mkdir(parents=True, exist_ok=True)
    if not download_missing:
        receipts = []
        for episode_id in episode_ids:
            path = annotation_root / f"{episode_id}.json"
            if path.exists() and not path.is_symlink():
                payload = path.read_bytes()
                result = "present"
                receipt: dict[str, Any] = {
                    "episode_id": episode_id,
                    "url": annotation_url(episode_id),
                    "revision": REVISION,
                    "result": result,
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            else:
                receipt = {
                    "episode_id": episode_id,
                    "url": annotation_url(episode_id),
                    "revision": REVISION,
                    "result": "missing",
                }
            receipts.append(receipt)
        return receipts
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as pool:
        futures = [
            pool.submit(download_one_annotation, episode_id, annotation_root)
            for episode_id in episode_ids
        ]
        return [future.result() for future in futures]


def load_annotation(annotation_root: Path, episode_id: str) -> dict[str, Any]:
    path = annotation_root / f"{episode_id}.json"
    if path.is_symlink() or not path.is_file():
        raise GuiDevError("annotation missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GuiDevError("annotation must be an object")
    if safe_episode_id(payload.get("episode_id")) != episode_id:
        raise GuiDevError("annotation episode_id mismatch")
    return payload


def png_header_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(33)
    if len(header) != 33 or not header.startswith(b"\x89PNG\r\n\x1a\n") or header[12:16] != b"IHDR":
        raise GuiDevError("PNG IHDR required")
    expected_crc = struct.unpack_from(">I", header, 29)[0]
    actual_crc = binascii.crc32(header[12:29]) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise GuiDevError("PNG IHDR CRC mismatch")
    width, height = struct.unpack_from(">II", header, 16)
    if width <= 0 or height <= 0:
        raise GuiDevError("invalid screenshot dimensions")
    return int(width), int(height)


def image_header_info(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise GuiDevError("regular screenshot required")
    width, height = png_header_dimensions(path)
    return {"width": width, "height": height}


def decode_png_rgb(payload: bytes) -> tuple[int, int, str]:
    try:
        image_module = importlib.import_module("PIL.Image")
        with image_module.open(io.BytesIO(payload)) as image:
            if image.format != "PNG":
                raise GuiDevError("PNG screenshot required")
            image.load()
            rgb = image.convert("RGB")
            return int(rgb.width), int(rgb.height), sha256_bytes(rgb.tobytes())
    except ImportError as exc:
        raise GuiDevError("Pillow is required for full PNG decode") from exc


def image_info(path: Path) -> dict[str, Any]:
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise GuiDevError("regular screenshot required")
    payload = path.read_bytes()
    width, height, decoded_sha256 = decode_png_rgb(payload)
    if width <= 0 or height <= 0:
        raise GuiDevError("invalid screenshot dimensions")
    return {
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "width": width,
        "height": height,
        "decoded_rgb_sha256": decoded_sha256,
    }


def parse_box(value: Any) -> Box:
    if not isinstance(value, list) or len(value) != 4:
        raise GuiDevError("invalid bbox")
    box = tuple(value)
    if not all(type(item) is int for item in box):
        raise GuiDevError("invalid bbox")
    return box


def parse_click_point(value: Any) -> Point:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, list) and len(item) == 2 for item in value)
    ):
        raise GuiDevError("invalid click info")
    first = tuple(value[0])
    second = tuple(value[1])
    if first != second or not all(type(item) is int for item in first):
        raise GuiDevError("invalid click point")
    return first


def box_is_legal(box: Box, *, width: int, height: int) -> bool:
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def contains(box: Box, point: Point) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def intersects(a: Box, b: Box) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def candidate_negative_boxes(true_box: Box, point: Point, *, width: int, height: int) -> list[Box]:
    box_width = true_box[2] - true_box[0]
    box_height = true_box[3] - true_box[1]
    if box_width <= 0 or box_height <= 0 or box_width > width or box_height > height:
        return []
    xs = list(range(0, width - box_width + 1, max(1, box_width)))
    ys = list(range(0, height - box_height + 1, max(1, box_height)))
    if xs[-1] != width - box_width:
        xs.append(width - box_width)
    if ys[-1] != height - box_height:
        ys.append(height - box_height)
    boxes: list[Box] = []
    for y in ys:
        for x in xs:
            box = (x, y, x + box_width, y + box_height)
            if contains(box, point) or intersects(box, true_box):
                continue
            if any(intersects(box, existing) for existing in boxes):
                continue
            boxes.append(box)
            if len(boxes) == 3:
                return boxes
    return boxes


def option_order(episode_id: str, step: int, boxes: list[Box]) -> tuple[list[Box], str]:
    labels = ["A", "B", "C", "D"]
    ordered = list(boxes)
    seed = int(step_key(episode_id, step)[:16], 16)
    random.Random(seed).shuffle(ordered)
    return ordered, labels[ordered.index(boxes[0])]


def format_box(box: Box) -> str:
    x1, y1, x2, y2 = box
    return f"x1={x1}, y1={y1}, x2={x2}, y2={y2}"


def validate_episode_steps(annotation: dict[str, Any], episode_id: str, media_root: Path) -> None:
    steps = annotation.get("steps")
    if not isinstance(steps, list) or not steps:
        raise GuiDevError("annotation missing steps")
    if annotation.get("step_length") != len(steps):
        raise GuiDevError("step_length mismatch")
    seen_steps: set[int] = set()
    for raw in steps:
        if not isinstance(raw, dict) or type(raw.get("step")) is not int:
            raise GuiDevError("invalid step record")
        step_no = int(raw["step"])
        if step_no in seen_steps:
            raise GuiDevError("duplicate step")
        seen_steps.add(step_no)
        expected = f"{episode_id}_{step_no}.png"
        if safe_file_name(raw.get("screenshot")) != expected:
            raise GuiDevError("screenshot identity mismatch")
        path = media_root / "screenshots" / expected
        if path.is_symlink() or not path.is_file():
            raise GuiDevError("episode screenshot missing")
    if seen_steps != set(range(len(steps))):
        raise GuiDevError("step numbers are not contiguous")


def eligible_step(
    annotation: dict[str, Any],
    *,
    episode_id: str,
    media_root: Path,
    counters: Counter[str],
) -> dict[str, Any] | None:
    validate_episode_steps(annotation, episode_id, media_root)
    steps = annotation["steps"]
    raw_click_steps = [
        raw
        for raw in steps
        if isinstance(raw, dict) and raw.get("action") == "CLICK" and type(raw.get("step")) is int
    ]
    ordered_click_steps = sorted(
        raw_click_steps,
        key=lambda raw: (step_key(episode_id, int(raw["step"])), int(raw["step"])),
    )
    for raw in ordered_click_steps:
        step = int(raw["step"])
        instruction = str(raw.get("low_level_instruction", "")).strip()
        if not instruction:
            counters["empty_low_level_instruction"] += 1
            continue
        screenshot_name = safe_file_name(raw.get("screenshot"))
        media_path = media_root / "screenshots" / screenshot_name
        info = image_header_info(media_path)
        box = parse_box(raw.get("sam2_bbox"))
        if not box_is_legal(box, width=info["width"], height=info["height"]):
            counters["illegal_bbox"] += 1
            continue
        point = parse_click_point(raw.get("info"))
        if not contains(box, point):
            counters["click_outside_bbox"] += 1
            continue
        negatives = candidate_negative_boxes(box, point, width=info["width"], height=info["height"])
        if len(negatives) < 3:
            counters["insufficient_negative_boxes"] += 1
            continue
        return {
            "episode_id": episode_id,
            "step": step,
            "instruction": instruction,
            "screenshot": screenshot_name,
            "media_path": media_path,
            "true_box": box,
            "point": point,
            "negative_boxes": negatives[:3],
            "image": info,
            "raw_step_sha256": sha256_bytes(canonical_json(raw).encode()),
        }
    counters["non_click_step"] += len(steps) - len(raw_click_steps)
    return None


def source_hash_for(candidate: dict[str, Any]) -> str:
    payload = {
        "episode_id": candidate["episode_id"],
        "step": candidate["step"],
        "screenshot": candidate["screenshot"],
        "screenshot_sha256": candidate["image"]["sha256"],
        "raw_step_sha256": candidate["raw_step_sha256"],
        "revision": REVISION,
    }
    return sha256_bytes(canonical_json(payload).encode())


def sample_for(candidate: dict[str, Any]) -> tuple[Sample, list[Box], str]:
    boxes = [candidate["true_box"], *candidate["negative_boxes"]]
    ordered_boxes, target_token = option_order(candidate["episode_id"], candidate["step"], boxes)
    options = [format_box(box) for box in ordered_boxes]
    source_hash = source_hash_for(candidate)
    sample = Sample.model_validate(
        {
            "id": (
                f"gui-odyssey-train-{candidate['episode_id']}-"
                f"{candidate['step']:04d}-{source_hash[:12]}"
            ),
            "prefix": {
                "state": (
                    "Select the screenshot region that matches the atomic GUI subtask. "
                    "This development item tests bbox target localization only, "
                    "not full GUI action "
                    "generation or open-parameter execution."
                ),
                "media": [
                    {
                        "type": "image",
                        "uri": candidate["screenshot"],
                        "sha256": candidate["image"]["sha256"],
                    }
                ],
            },
            "questions": [
                {
                    "qid": "target-region",
                    "type": "choice",
                    "instruction": str(candidate["instruction"]),
                    "options": options,
                    "target": {
                        "kind": "hard",
                        "value": target_token,
                        "source": "gui_odyssey_sam2_bbox_program_truth",
                    },
                }
            ],
            "meta": {
                "modality": ["image"],
                "source_dataset": DATASET,
                "source_revision": REVISION,
                "source_sha256": source_hash,
                "license": ANNOTATION_LICENSE,
                "usage": "commercial",
                "pool": "dev",
                "family_id": f"gui:{candidate['episode_id']}",
                "data_version": DATA_VERSION,
            },
        }
    )
    return sample, ordered_boxes, target_token


def development_record_for(sample: Sample) -> DevelopmentRecord:
    return DevelopmentRecord(
        item_id=sample.id,
        modality="screenshot",
        family_hash=sha256_bytes(sample.meta.family_id.encode()),
        source_hash=sample.meta.source_sha256,
        source_dataset=sample.meta.source_dataset,
        source_revision=sample.meta.source_revision,
        license=sample.meta.license,
        label_origin="program_truth",
    )


def media_manifest_row(candidate: dict[str, Any], sample: Sample) -> dict[str, Any]:
    return {
        "item_id": sample.id,
        "episode_id": candidate["episode_id"],
        "step": candidate["step"],
        "screenshot": candidate["screenshot"],
        "sha256": candidate["image"]["sha256"],
        "bytes": candidate["image"]["bytes"],
        "width": candidate["image"]["width"],
        "height": candidate["image"]["height"],
        "decoded_rgb_sha256": candidate["image"]["decoded_rgb_sha256"],
        "license": ANNOTATION_LICENSE,
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "attribution": "GUIOdyssey annotation and screenshot source",
    }


def provenance_row(
    candidate: dict[str, Any],
    sample: Sample,
    ordered_boxes: list[Box],
    target_token: str,
) -> dict[str, Any]:
    return {
        "item_id": sample.id,
        "episode_id": candidate["episode_id"],
        "family_id": sample.meta.family_id,
        "step": candidate["step"],
        "screenshot": candidate["screenshot"],
        "click_point": list(candidate["point"]),
        "gold_bbox": list(candidate["true_box"]),
        "option_bboxes": [list(box) for box in ordered_boxes],
        "target_token": target_token,
        "source_hash": sample.meta.source_sha256,
        "label_origin": "program_truth",
    }


def exclusion_ledger_rows(
    plan_rows: list[dict[str, Any]], selected: list[Sample]
) -> list[dict[str, Any]]:
    selected_families = {sample.meta.family_id for sample in selected}
    rows = []
    for row in plan_rows:
        episode_id = safe_episode_id(row["episode_id"])
        family_id = f"gui:{episode_id}"
        rows.append(
            {
                "family_id": family_id,
                "episode_id": episode_id,
                "pool": "dev_excluded" if family_id in selected_families else "train_available",
                "screenshot_count": row.get("screenshot_count"),
                "partition": row.get("partition"),
                "source_dataset": DATASET,
                "source_revision": REVISION,
                "allocation_key_sha256": allocation_key(episode_id),
            }
        )
    return sorted(rows, key=lambda item: str(item["episode_id"]))


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


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode())


def build_report(
    *,
    status: str,
    exit_code: int,
    command: list[str],
    started_at: float,
    blocker: str | None = None,
    source: dict[str, Any] | None = None,
    split: dict[str, Any] | None = None,
    outputs: dict[str, str] | None = None,
    config: dict[str, Any],
) -> dict[str, Any]:
    finished_at = time.time()
    return {
        "status": status,
        "exit_code": exit_code,
        "blocker": blocker,
        "command": command,
        "protocol": "s0-gui-dev-v1",
        "config": {**config, "config_sha256": sha256_bytes(canonical_json(config).encode())},
        "primitive_scope": {
            "choice4_target_localization_only": True,
            "score_covered": False,
            "noul_covered": False,
            "full_gui_action_generation_covered": False,
        },
        "training_admitted": False,
        "quality_gate": "not_run",
        "gpu_used": False,
        "holdout_accessed": False,
        "cost": None,
        "source": source,
        "split": split,
        "outputs": outputs,
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


def config_payload(*, dev_count: int, max_annotation_downloads: int) -> dict[str, Any]:
    return {
        "protocol": "s0-gui-dev-v1",
        "seed": SEED,
        "allocation_prefix": ALLOCATION_PREFIX,
        "dev_count": dev_count,
        "max_annotation_downloads": max_annotation_downloads,
        "max_download_workers": MAX_DOWNLOAD_WORKERS,
        "download_timeout_seconds": DOWNLOAD_TIMEOUT_SECONDS,
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "data_version": DATA_VERSION,
        "license": ANNOTATION_LICENSE,
    }


def run(
    *,
    plan_path: Path,
    annotation_root: Path,
    media_root: Path,
    dev_out: Path,
    dev_ledger_out: Path,
    exclusion_ledger_out: Path,
    media_manifest_out: Path,
    provenance_out: Path,
    receipts_out: Path,
    report_out: Path,
    dev_count: int,
    max_annotation_downloads: int,
    download_missing: bool,
    command: list[str],
) -> int:
    started_at = time.time()
    config = config_payload(dev_count=dev_count, max_annotation_downloads=max_annotation_downloads)
    output_paths = [
        dev_out,
        dev_ledger_out,
        exclusion_ledger_out,
        media_manifest_out,
        provenance_out,
        receipts_out,
    ]
    if report_out.exists() or report_out.is_symlink():
        return 1
    if report_out.resolve(strict=False) in {path.resolve(strict=False) for path in output_paths}:
        return 1
    try:
        if not 1 <= dev_count < max_annotation_downloads <= MAX_ANNOTATION_DOWNLOADS:
            raise GuiDevError("dev_count/max_annotation_downloads outside bounded contract")
        validate_output_paths([*output_paths, report_out])
        plan_rows = load_plan(plan_path)
        episode_ids = selected_episode_ids(
            plan_rows, needed=dev_count, limit=max_annotation_downloads
        )
        receipts = ensure_annotations(
            episode_ids, annotation_root, download_missing=download_missing
        )
        receipts_hash = write_json(receipts_out, receipts)
        samples: list[Sample] = []
        media_rows: list[dict[str, Any]] = []
        provenance_rows: list[dict[str, Any]] = []
        exclusions: Counter[str] = Counter()
        candidate_episodes = 0
        for episode_id in episode_ids:
            if len(samples) >= dev_count:
                break
            try:
                annotation = load_annotation(annotation_root, episode_id)
                candidate = eligible_step(
                    annotation, episode_id=episode_id, media_root=media_root, counters=exclusions
                )
            except (GuiDevError, OSError, ValueError, json.JSONDecodeError) as exc:
                exclusions[str(exc) if isinstance(exc, GuiDevError) else type(exc).__name__] += 1
                continue
            if candidate is None:
                exclusions["episode_without_eligible_click"] += 1
                continue
            candidate_episodes += 1
            candidate["image"] = image_info(candidate["media_path"])
            sample, ordered_boxes, target_token = sample_for(candidate)
            samples.append(sample)
            media_rows.append(media_manifest_row(candidate, sample))
            provenance_rows.append(provenance_row(candidate, sample, ordered_boxes, target_token))
        if len(samples) != dev_count:
            raise GuiDevError(f"eligible GUI dev items {len(samples)} below requested {dev_count}")
        if len({sample.meta.family_id for sample in samples}) != len(samples):
            raise GuiDevError("selected GUI families are not unique")
        dev_hash = write_jsonl(dev_out, samples)
        dev_ledger_hash = write_jsonl(
            dev_ledger_out, [development_record_for(sample) for sample in samples]
        )
        exclusion_hash = write_jsonl(
            exclusion_ledger_out, exclusion_ledger_rows(plan_rows, samples)
        )
        media_hash = write_jsonl(media_manifest_out, media_rows)
        provenance_hash = write_jsonl(provenance_out, provenance_rows)
        outputs = {
            "dev_jsonl_sha256": dev_hash,
            "development_ledger_sha256": dev_ledger_hash,
            "exclusion_ledger_sha256": exclusion_hash,
            "media_manifest_sha256": media_hash,
            "provenance_jsonl_sha256": provenance_hash,
            "download_receipts_sha256": receipts_hash,
        }
        source = {
            "media_plan_sha256": sha256_file(plan_path),
            "annotation_receipts": dict(
                sorted(Counter(str(row["result"]) for row in receipts).items())
            ),
            "annotation_url_prefix": ANNOTATION_URL_PREFIX,
            "source_dataset": DATASET,
            "source_revision": REVISION,
            "license": ANNOTATION_LICENSE,
            "attribution": "hflqf88888/GUIOdyssey annotations and screenshots",
        }
        split = {
            "dev_items": len(samples),
            "selected_gui_families": len({sample.meta.family_id for sample in samples}),
            "all_train_gui_families": len(plan_rows),
            "candidate_episodes_with_eligible_click": candidate_episodes,
            "scanned_episode_limit": len(episode_ids),
            "excluded_counts": dict(sorted(exclusions.items())),
            "target_token_counts": dict(
                sorted(Counter(row["target_token"] for row in provenance_rows).items())
            ),
        }
        report = build_report(
            status="passed",
            exit_code=0,
            command=command,
            started_at=started_at,
            source=source,
            split=split,
            outputs=outputs,
            config=config,
        )
        code = 0
    except (OSError, GuiDevError, ValueError) as exc:
        report = build_report(
            status="blocked",
            exit_code=1,
            command=command,
            started_at=started_at,
            blocker=str(exc) if isinstance(exc, GuiDevError) else type(exc).__name__,
            config=config,
        )
        code = 1
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--dev-out", type=Path, required=True)
    parser.add_argument("--dev-ledger-out", type=Path, required=True)
    parser.add_argument("--exclusion-ledger-out", type=Path, required=True)
    parser.add_argument("--media-manifest-out", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path, required=True)
    parser.add_argument("--receipts-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dev-count", type=int, default=DEV_COUNT)
    parser.add_argument("--max-annotation-downloads", type=int, default=MAX_ANNOTATION_DOWNLOADS)
    parser.add_argument("--download-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(
        plan_path=args.plan,
        annotation_root=args.annotation_root,
        media_root=args.media_root,
        dev_out=args.dev_out,
        dev_ledger_out=args.dev_ledger_out,
        exclusion_ledger_out=args.exclusion_ledger_out,
        media_manifest_out=args.media_manifest_out,
        provenance_out=args.provenance_out,
        receipts_out=args.receipts_out,
        report_out=args.report,
        dev_count=args.dev_count,
        max_annotation_downloads=args.max_annotation_downloads,
        download_missing=bool(args.download_missing),
        command=[sys.executable, "-m", "scripts.s0_gui_dev", *(argv or sys.argv[1:])],
    )


if __name__ == "__main__":
    raise SystemExit(main())
