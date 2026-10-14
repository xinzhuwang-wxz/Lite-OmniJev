"""Freeze public MVBench frame identities without questions, labels, or models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.public_eval_media import checked_zip_members, sha256

REVISION = "230a2d4fac8900333c61754641c7a13e069ac9c6"
REFERENCE_SHA = "4530b25b34e464958d26d3f4b01abd73a4ebe53aa287702c5206048524e43efb"
TASKS = (
    "Action Sequence",
    "Action Prediction",
    "Unexpected Action",
    "Object Interaction",
    "Object Shuffle",
    "Action Localization",
    "Action Count",
    "State Change",
    "Character Order",
)
ARCHIVES = {
    "star": "star.zip",
    "perception": "perception.zip",
    "FunQA_test": "FunQA_test.zip",
    "sta": "sta.zip",
}
MAX_TEMP = 128 * 1024**2
SUPPLEMENTAL_SHA256 = {
    "EDXBD.mp4": "bcde2775a4c060d47956b190c40d0df2a4f596d356ef6f2b2de6e088f588b59a",
    "K47J5.mp4": "86e41185686f978e84d6ad360950fe472ac47a7182e8c0c0665fd2e0533d9e91",
    "9MNZ5.mp4": "9a5d0dfb03adea950c91057e2b90f871b1d22477611dbee020b1837b318af59c",
    "QXT9W.mp4": "494c527685ebfbaadb6cb0175c2597ea30cf9f0bfd7b7ebd99979eba4e8ef576",
    "ABHC6.mp4": "4147ff99d684515527c005497a234a47f27f862b998e18513ca8045198fa8e6a",
    "ALXUC.mp4": "fddc61a91533ec216dc34919368c7b0a64fba658b0b5109afe0fc05c1737915c",
    "BAUQE.mp4": "d7596c6841c0064d35f77d7bbd7b38e92521114de6e24080f2e14c5caccb51bf",
    "PHH6B.mp4": "879dca0e48881751d045ff29ea54eddbf81f725b16eb60ed3647d0a8a1f09cfc",
    "MNC10.mp4": "362947b747fc81dc6cf76afad20ac9fc0e7b3c31a7501bbb6c305d61c6724d03",
    "W7CR5.mp4": "fa70172648c2671de7e885285bfb1d53c301895b37938aadaef0d54baf0b7380",
    "Q8UJ8.mp4": "7d27402ab5453200bb803b341572ec2aed9f1fa260ba346f9e07bc5ee8b546f8",
    "X9WTR.mp4": "9e54b43feecf28543fc325dd2e8c30c677a35e9c733a3f80964dc01da4bfc64a",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def frame_indices(count: int, fps: float, bound: tuple[float, float] | None) -> list[int]:
    """Official MVBench 16-bin midpoint formula, Python/NumPy ties-to-even round."""
    if count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid_video_metadata")
    start, end = bound if bound is not None else (-100000.0, 100000.0)
    if not all(math.isfinite(x) for x in (start, end)) or start > end:
        raise ValueError("invalid_clip_window")
    lo, hi = max(0, round(start * fps)), min(round(end * fps), count - 1)
    if lo > hi:
        raise ValueError("clip_outside_video")
    segment = float(hi - lo) / 16
    return [int(lo + segment / 2 + round(segment * index)) for index in range(16)]


def references(config: dict[str, Any], metadata: Path) -> list[dict[str, Any]]:
    """Whitelist original public media reference fields; discard question/answer data."""
    result = []
    selected = [t for t in config["mvbench"]["tasks"] if t["task"] in TASKS]
    if tuple(t["task"] for t in selected) != TASKS:
        raise ValueError("task_order_mismatch")
    for task in selected:
        path = metadata / ("mvbench-json-" + Path(task["file"]).name)
        if sha256(path) != task["sha256"]:
            raise ValueError("public_json_hash_mismatch")
        rows = json.loads(path.read_text())
        if len(rows) != 200:
            raise ValueError("public_row_count_mismatch")
        for index, row in enumerate(rows):
            item = {"task_json_sha256": task["sha256"], "row_index": index, "video": row["video"]}
            if task["clip_bounds"]:
                item.update(start=row["start"], end=row["end"])
            result.append(item)
    if len(result) != 1800 or hashlib.sha256(canonical(result)).hexdigest() != REFERENCE_SHA:
        raise ValueError("public_reference_manifest_hash_mismatch")
    return result


def probe(path: Path) -> dict[str, Any]:
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames,duration,codec_name",
                "-of",
                "json",
                str(path),
            ],
            timeout=60,
        )
    )
    stream = info["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    frames = int(stream["nb_frames"])
    width, height = int(stream["width"]), int(stream["height"])
    if not (
        0 < width * height * 3 <= 64 * 1024**2 and frames > 0 and math.isfinite(fps) and fps > 0
    ):
        raise ValueError("video_metadata_limit")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "fps_rational": stream["avg_frame_rate"],
        "frames": frames,
        "codec": stream["codec_name"],
        "duration": float(stream["duration"]),
    }


def decode_frames(
    path: Path, indices: list[int], metadata: dict[str, Any], *, png_cache: Path | None = None
) -> dict[int, str]:
    """One bounded RGB frame in RAM; decode sequentially through final selected frame."""
    wanted = sorted(set(indices))
    if not wanted or wanted[0] < 0 or wanted[-1] >= metadata["frames"]:
        raise ValueError("frame_index_outside_video")
    expression = "+".join(f"eq(n\\,{index})" for index in wanted)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-threads",
        "1",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        f"select={expression}",
        "-vsync",
        "0",
        "-filter_threads",
        "1",
        "-threads",
        "1",
        "-frames:v",
        str(len(wanted)),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    if png_cache is not None:
        command.extend(
            [
                "-n",
                "-map",
                "0:v:0",
                "-vf",
                f"select={expression}",
                "-vsync",
                "0",
                "-filter_threads",
                "1",
                "-threads",
                "1",
                "-frames:v",
                str(len(wanted)),
                "-pix_fmt",
                "rgb24",
                "-c:v",
                "png",
                "-pred",
                "mixed",
                "-compression_level",
                "6",
                "-start_number",
                "0",
                str(png_cache / "frame-%05d.png"),
            ]
        )
    size = metadata["width"] * metadata["height"] * 3
    hashes = {}
    # stderr is an owned disk file, avoiding a pipe deadlock; subprocess has a hard timeout.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        assert process.stdout is not None
        try:
            # Selector deadline also covers a decoder that produces no bytes.
            import selectors

            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + 900
                for index in wanted:
                    frame = bytearray()
                    while len(frame) < size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise TimeoutError("frame_decode_timeout")
                        block = os.read(process.stdout.fileno(), min(65536, size - len(frame)))
                        if not block:
                            raise ValueError("short_rgb_frame")
                        frame.extend(block)
                    hashes[index] = hashlib.sha256(frame).hexdigest()
            if process.wait(timeout=10) != 0:
                raise ValueError("ffmpeg_frame_decode_failed")
            if process.stdout.read(1):
                raise ValueError("unexpected_extra_frame")
            if png_cache is not None:
                for position, index in enumerate(wanted):
                    cached = png_cache / f"frame-{position:05d}.png"
                    target = png_cache / f"{index}.png"
                    if target.exists() or not cached.is_file():
                        raise ValueError("invalid_png_cache_output")
                    cached.rename(target)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
    return hashes


def verify_png_frames(png_cache: Path, expected: dict[int, str], metadata: dict[str, Any]) -> None:
    """Decode every lossless cache asset and reject any RGB identity change."""
    ordered = sorted(expected)
    with tempfile.TemporaryDirectory(prefix="verify-", dir=png_cache) as temp:
        sequence = Path(temp)
        for position, index in enumerate(ordered):
            os.link(png_cache / f"{index}.png", sequence / f"{position:05d}.png")
        info = dict(metadata, frames=len(ordered))
        actual = decode_frames(sequence / "%05d.png", list(range(len(ordered))), info)
        if any(actual[position] != expected[index] for position, index in enumerate(ordered)):
            raise ValueError("png_RGB_hash_mismatch")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def run(
    project: Path, config_path: Path, output: Path, *, plan_only: bool = False
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    if config["mvbench"]["revision"] != REVISION:
        raise ValueError("revision_mismatch")
    source = project / "artifacts/data/research-eval/mvbench" / REVISION
    metadata = project / "artifacts/data/research-eval/video-eval-source-metadata"
    rows = references(config, metadata)
    tasks = [t for t in config["mvbench"]["tasks"] if t["task"] in TASKS]
    groups: dict[str, list[int]] = defaultdict(list)
    sources = {}
    source_stats = {}
    archives = {}
    members = {}
    try:
        for name, filename in ARCHIVES.items():
            archive_path = source / filename
            observed = archive_path.stat()
            source_stats[filename] = [observed.st_size, observed.st_mtime_ns, observed.st_ino]
            archive = zipfile.ZipFile(archive_path)
            archives[name] = archive
            members[name] = checked_zip_members(archive, max_expanded_bytes=16 * 1024**3)
        supplemental = source / "supplemental-charades-2I0cX8bvsoqLbtsk6mi6Wrb0d0uV1H8E"
        for number, row in enumerate(rows):
            task = tasks[number // 200]
            prefix = task["source_prefix"].removeprefix("your_data_path/")
            member = prefix + row["video"]
            category = prefix.split("/")[0]
            if member in members[category]:
                zip_info = members[category][member]
                if zip_info.file_size > MAX_TEMP:
                    raise ValueError("media_temp_limit")
                location = {
                    "archive": ARCHIVES[category],
                    "member": member,
                    "bytes": zip_info.file_size,
                    "crc32": f"{zip_info.CRC:08x}",
                }
            else:
                path = supplemental / row["video"]
                if category != "star" or not path.is_file() or path.stat().st_size > MAX_TEMP:
                    raise ValueError("missing_public_media")
                location = {"supplemental_path": str(path), "bytes": path.stat().st_size}
            key = member
            if "supplemental_path" in location:
                observed = Path(location["supplemental_path"]).stat()
                source_stats[key] = [observed.st_size, observed.st_mtime_ns, observed.st_ino]
            groups[key].append(number)
            sources[key] = (category, location)
        report: dict[str, Any] = {
            "status": "running",
            "started_utc": datetime.now(UTC).isoformat(),
            "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
            if Path("/sys/fs/cgroup/cpu.max").exists()
            else None,
            "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip()
            if Path("/sys/fs/cgroup/memory.max").exists()
            else None,
            "reference_sha256": REFERENCE_SHA,
            "rows": len(rows),
            "tasks": list(TASKS),
            "unique_media": len(groups),
            "completed_media": 0,
            "source_stats": source_stats,
            "source_archive_provenance": {
                a["path"]: {k: a[k] for k in ("bytes", "sha256")}
                for a in config["mvbench"]["archives"]
                if Path(a["path"]).name in ARCHIVES.values()
            },
            "helper_sha256": sha256(
                Path(__file__).resolve().parents[1] / "scripts/public_eval_media.py"
            ),
            "code_sha256": sha256(Path(__file__)),
            "ffmpeg_binary_sha256": sha256(Path(shutil.which("ffmpeg") or "ffmpeg")),
            "ffprobe_binary_sha256": sha256(Path(shutil.which("ffprobe") or "ffprobe")),
            "config_sha256": sha256(config_path),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "cache_strategy": (
                "manifest_only_bounded_current_media_RGB; "
                "regenerate_and_verify_each_hash_before_forward"
            ),
            "max_temporary_media_bytes": MAX_TEMP,
            "decoder_threads": 1,
            "gpu_seconds": 0,
            "teacher_forward": "pending",
            "quality_gates": "not_measured",
            "training_admitted": False,
            "media": [],
            "exit_code": None,
        }
        decoder_binding = {
            k: report[k]
            for k in ("ffmpeg_binary_sha256", "ffprobe_binary_sha256", "ffmpeg", "helper_sha256")
        }
        started = time.monotonic()
        checkpoint = output / "checkpoints"
        checkpoint.mkdir(exist_ok=True)
        plan_rows = [dict(row, task=TASKS[i // 200]) for i, row in enumerate(rows)]
        for key, numbers in groups.items():
            category, location = sources[key]
            cache = checkpoint / (hashlib.sha256(key.encode()).hexdigest() + ".json")
            existing = json.loads(cache.read_text()) if cache.exists() else None
            if existing and (
                existing["code_sha256"] != report["code_sha256"]
                or existing["reference_sha256"] != REFERENCE_SHA
                or existing["source_stats"] != source_stats
                or existing["decoder_binding"] != decoder_binding
            ):
                raise ValueError("checkpoint_protocol_mismatch")
            if existing and (plan_only or "frame_hashes" in existing):
                item = existing
            else:
                with tempfile.TemporaryDirectory(prefix="public-video-", dir=output) as temp:
                    video = Path(temp) / "source.mp4"
                    if "archive" in location:
                        with (
                            archives[category].open(location["member"]) as inp,
                            video.open("xb") as out,
                        ):
                            shutil.copyfileobj(inp, out, 1024**2)
                    else:
                        shutil.copyfile(location["supplemental_path"], video)
                    media_started = time.monotonic()
                    media_sha = sha256(video)
                    if (
                        "supplemental_path" in location
                        and media_sha != SUPPLEMENTAL_SHA256[Path(key).name]
                    ):
                        raise ValueError("supplemental_source_hash_mismatch")
                    info = probe(video)
                    sampled = {}
                    for number in numbers:
                        row = rows[number]
                        bound = (row["start"], row["end"]) if "start" in row else None
                        sampled[str(number)] = frame_indices(info["frames"], info["fps"], bound)
                    union = sorted({i for indices in sampled.values() for i in indices})
                    item = {
                        "media": key,
                        "source": location,
                        "source_sha256": media_sha,
                        "metadata": info,
                        "row_indices": sampled,
                        "unique_selected_frames": len(union),
                        "code_sha256": report["code_sha256"],
                        "reference_sha256": REFERENCE_SHA,
                        "source_stats": source_stats,
                        "decoder_binding": decoder_binding,
                    }
                    if not plan_only:
                        item["frame_hashes"] = {
                            str(i): digest
                            for i, digest in decode_frames(video, union, info).items()
                        }
                    item["processing_seconds"] = time.monotonic() - media_started
                    write_json(cache, item)
            report["media"].append(item)
            report["completed_media"] += 1
            report["elapsed_seconds"] = time.monotonic() - started
            for number in numbers:
                plan_rows[number].update(
                    media=key,
                    indices=item["row_indices"][str(number)],
                    width=item["metadata"]["width"],
                    height=item["metadata"]["height"],
                )
                if "frame_hashes" in item:
                    plan_rows[number]["rgb24_sha256"] = [
                        item["frame_hashes"][str(i)] for i in plan_rows[number]["indices"]
                    ]
            write_json(output / "progress.json", {k: v for k, v in report.items() if k != "media"})
        report["unique_selected_frames"] = sum(m["unique_selected_frames"] for m in report["media"])
        report["raw_RGB_bytes_if_persisted"] = sum(
            m["unique_selected_frames"] * m["metadata"]["width"] * m["metadata"]["height"] * 3
            for m in report["media"]
        )
        report.update(
            status="planned" if plan_only else "all_1800_rows_16_RGB_frames_verified", exit_code=0
        )
        report["completed_utc"] = datetime.now(UTC).isoformat()
        report["manifest_sha256"] = hashlib.sha256(canonical(plan_rows)).hexdigest()
        manifest = output / ("plan-manifest.json" if plan_only else "frame-manifest.json")
        write_json(manifest, plan_rows)
        manifest.chmod(0o444)
        write_json(output / ("plan-report.json" if plan_only else "report.json"), report)
        return report
    finally:
        for archive in archives.values():
            archive.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    try:
        run(args.project, args.config, args.output, plan_only=args.plan_only)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output / "failure.json",
            {"exit_code": 1, "error_type": type(error).__name__, "reason": str(error)[:200]},
        )
        raise


if __name__ == "__main__":
    main()
