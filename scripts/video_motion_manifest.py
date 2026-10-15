"""CPU manifest for one explicitly SHA-bound local video; never changes media/loaders."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

from train.multimodal import _DECODE_INPUT_OPTIONS, file_hash, resolve_media
from train.video_sampling import motion_coverage_indices

MAX_VIDEO_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 7200
THUMBNAIL_BYTES = 64 * 64
MAX_RAW_BYTES = (MAX_FRAMES + 1) * THUMBNAIL_BYTES


def build_manifest(video: Path, expected_sha256: str) -> dict[str, Any]:
    """Decode bounded 64x64 gray thumbnails; expose only metadata, scores and indices."""
    started = time.monotonic()
    path = resolve_media(
        video.absolute().parent, video.name, expected_sha256, max_bytes=MAX_VIDEO_BYTES
    )
    with tempfile.TemporaryDirectory(prefix="omnijev-motion-") as directory:
        metadata_path = Path(directory) / "metadata.json"
        with metadata_path.open("xb") as stream:
            subprocess.run(
                [
                    "ffprobe",
                    *_DECODE_INPUT_OPTIONS,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "format=duration:stream=codec_type,avg_frame_rate,nb_frames,width,height",
                    "-of",
                    "json",
                    str(path),
                ],
                check=True,
                stdout=stream,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=30,
            )
        if metadata_path.stat().st_size > 65536:
            raise ValueError("video metadata exceeds byte limit")
        metadata = json.loads(metadata_path.read_bytes())
        duration = float(metadata["format"]["duration"])
        stream = next(s for s in metadata["streams"] if s["codec_type"] == "video")
        fps = float(Fraction(stream["avg_frame_rate"]))
        count = int(stream["nb_frames"])
        if not math.isfinite(duration) or not 0 < duration <= 60:
            raise ValueError("video duration must be finite and within60seconds")
        if not math.isfinite(fps) or fps <= 0 or not 16 <= count <= MAX_FRAMES:
            raise ValueError("video requires positive finite FPS and16..7200known frames")
        if not 0 < int(stream["width"]) * int(stream["height"]) <= 40_000_000:
            raise ValueError("video frame resolution exceeds pixel limit")
        raw = Path(directory) / "thumbnails.gray"
        decode_started = time.monotonic()
        with raw.open("xb") as pixels:
            subprocess.run(
                [
                    "ffmpeg",
                    *_DECODE_INPUT_OPTIONS,
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-filter_threads",
                    "1",
                    "-filter_complex_threads",
                    "1",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-sn",
                    "-dn",
                    "-vf",
                    "scale=64:64",
                    "-vsync",
                    "0",
                    "-frames:v",
                    str(MAX_FRAMES + 1),
                    "-threads",
                    "1",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "gray",
                    "pipe:1",
                ],
                check=True,
                stdout=pixels,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                timeout=60,
            )
        decode_seconds = time.monotonic() - decode_started
        size = raw.stat().st_size
        if size > MAX_RAW_BYTES or size != count * THUMBNAIL_BYTES:
            raise ValueError("decoded frame bytes differ from bounded video metadata")
        np = importlib.import_module("numpy")
        # At most29.5MB of uint8 data; per-frame signed difference uses only two thumbnails.
        frames = np.fromfile(raw, dtype=np.uint8).reshape(count, 64, 64)
        scores = [0.0] + [
            float(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean())
            for i in range(1, count)
        ]
    if file_hash(path) != expected_sha256:
        raise ValueError("video changed during CPU preparation")
    uniform = [round(i * (count - 1) / 15) for i in range(16)]
    adaptive = motion_coverage_indices(scores)
    return {
        "status": "prepared",
        "exit_code": 0,
        "gpu_used": False,
        "scope": "one-video experimental CPU sampling manifest; no quality evidence",
        "video_path": str(path),
        "video_sha256": expected_sha256,
        "fps": fps,
        "total_num_frames": count,
        "duration": duration,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "num_frames": 16,
        "uniform_indices": uniform,
        "adaptive_indices": adaptive,
        "timestamps_basis": "frame index / avg_frame_rate; nominal, not decoded PTS",
        "uniform_timestamps_seconds": [index / fps for index in uniform],
        "adaptive_timestamps_seconds": [index / fps for index in adaptive],
        "per_frame_motion": scores,
        "motion_score": "mean absolute adjacent64x64gray difference; frame0=0",
        "segment_rule": "14contiguous nonempty floor-boundary interior segments; earliest maximum",
        "limits": {
            "video_bytes": MAX_VIDEO_BYTES,
            "frames": MAX_FRAMES,
            "raw_bytes_including_one_overflow_frame": MAX_RAW_BYTES,
            "duration_seconds": 60,
            "decode_seconds": 60,
        },
        "decode_seconds": decode_seconds,
        "duration_seconds": time.monotonic() - started,
        "code_sha256": {
            name: file_hash(Path(name))
            for name in (
                "scripts/video_motion_manifest.py",
                "train/video_sampling.py",
                "train/multimodal.py",
            )
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.output.open("x", encoding="utf-8") as output:
        try:
            report = build_manifest(args.video, args.expected_sha256)
        except Exception as exc:
            report = {
                "status": "failed",
                "exit_code": 1,
                "gpu_used": False,
                "error": f"{type(exc).__name__}: {exc}",
                "duration_seconds": time.monotonic() - started,
            }
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
