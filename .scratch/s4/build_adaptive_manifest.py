"""Adaptive frame manifest: 8 uniform + 8 frame-difference peak indices per video.

Pure-pixel deterministic selection (no learned components, no CoT). For each
video the bounded 64x64 gray decode from the proven motion manifest builder
supplies adjacent-frame difference energies; peaks are chosen with a minimum
separation and merged with the 8-point uniform grid into strictly increasing
explicit frame indices compatible with train.multimodal.load_media.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


def load_scores(video: Path, sha: str) -> tuple[int, list[float]]:
    """Reuse the bounded thumbnail decode to obtain frame count and diff scores."""
    import subprocess
    import tempfile
    from fractions import Fraction

    from train.multimodal import _DECODE_INPUT_OPTIONS, file_hash, resolve_media

    with tempfile.TemporaryDirectory(prefix="omnijev-adaptive-") as directory:
        path = resolve_media(video.absolute().parent, video.name, sha, max_bytes=256 * 1024 * 1024)
        meta = subprocess.run(
            [
                "ffprobe", *_DECODE_INPUT_OPTIONS, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,nb_frames", "-of", "json", str(path),
            ],
            check=True, capture_output=True, text=True, timeout=30,
        )
        stream = json.loads(meta.stdout)["streams"][0]
        count = int(stream["nb_frames"])
        raw = Path(directory) / "gray"
        with raw.open("wb") as out:
            subprocess.run(
                [
                    "ffmpeg", *_DECODE_INPUT_OPTIONS, "-v", "error", "-threads", "1",
                    "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
                    "-vf", "scale=64:64", "-vsync", "0", "-f", "rawvideo",
                    "-pix_fmt", "gray", "pipe:1",
                ],
                check=True, stdout=out, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=60,
            )
        if file_hash(path) != sha:
            raise SystemExit("video changed during preparation")
        np = importlib.import_module("numpy")
        frames = np.fromfile(raw, dtype=np.uint8).reshape(count, 64, 64)
        scores = [0.0] + [
            float(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean())
            for i in range(1, count)
        ]
    return count, scores


def adaptive_indices(count: int, scores: list[float]) -> list[int]:
    uniform = sorted({round(i * (count - 1) / 7) for i in range(8)})
    min_gap = max(1, count // 32)
    ranked = sorted(range(1, count - 1), key=lambda i: (-scores[i], i))
    peaks: list[int] = []
    for candidate in ranked:
        if len(peaks) >= 8:
            break
        if any(abs(candidate - chosen) < min_gap for chosen in [*peaks, *uniform]):
            continue
        peaks.append(candidate)
    merged = sorted({*uniform, *peaks})
    if len(merged) != 16 or any(b <= a for a, b in zip(merged, merged[1:])):
        raise SystemExit(f"adaptive selection invalid for count={count}: {merged}")
    return merged


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: build_adaptive_manifest.py MEDIA_ROOT DATA_JSONL OUTPUT")
    media_root, data_path, output = (Path(a) for a in sys.argv[1:])
    import os

    sys.path.insert(0, os.getcwd())
    rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    manifest = {}
    for index, row in enumerate(rows):
        media = row["prefix"]["media"][0]
        video = media_root / media["uri"]
        count, scores = load_scores(video, media["sha256"])
        manifest[media["sha256"]] = {
            "total_num_frames": count,
            "adaptive_indices": adaptive_indices(count, scores),
        }
        if (index + 1) % 25 == 0:
            print(json.dumps({"progress": index + 1, "total": len(rows)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"videos": len(manifest), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
