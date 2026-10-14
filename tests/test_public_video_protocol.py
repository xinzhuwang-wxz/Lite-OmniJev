"""Observable public sampling and actual CPU RGB decoding behavior."""

import shutil
import subprocess
from pathlib import Path

import pytest

from eval.public_video_protocol import decode_frames, frame_indices, probe, verify_png_frames


def test_official_midpoints_and_clamped_bounds() -> None:
    assert frame_indices(33, 1.0, None) == list(range(1, 33, 2))
    assert frame_indices(33, 1.0, (0, 100)) == list(range(1, 33, 2))
    assert frame_indices(33, 1.0, (0.5, 16.5)) == list(range(16))
    assert frame_indices(1, 25, None) == [0] * 16
    assert frame_indices(33, 1, (32, 32)) == [32] * 16


@pytest.mark.parametrize(
    "count,fps,bound",
    [(0, 1, None), (10, 0, None), (10, float("nan"), None), (10, 1, (20, 30)), (10, 1, (3, 2))],
)
def test_invalid_windows_are_rejected(
    count: int, fps: float, bound: tuple[float, float] | None
) -> None:
    with pytest.raises(ValueError):
        frame_indices(count, fps, bound)


def test_real_cpu_decode_is_repeatable_and_does_not_change_input(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.fail("FFmpeg and ffprobe required for CPU protocol acceptance")
    video = tmp_path / "owned.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=16:duration=2",
            "-threads",
            "1",
            str(video),
        ],
        check=True,
    )
    original = video.read_bytes()
    info = probe(video)
    indices = frame_indices(info["frames"], info["fps"], None)
    png_cache = tmp_path / "rgb-cache"
    png_cache.mkdir()
    first = decode_frames(video, indices, info, png_cache=png_cache)
    verify_png_frames(png_cache, first, info)
    changed = dict(first)
    changed[indices[0]] = "0" * 64
    with pytest.raises(ValueError, match="png_RGB_hash_mismatch"):
        verify_png_frames(png_cache, changed, info)
    assert len(first) == 16
    assert len(set(first.values())) > 1
    assert decode_frames(video, indices, info) == first
    assert video.read_bytes() == original
    with pytest.raises(ValueError, match="frame_index_outside_video"):
        decode_frames(video, [info["frames"]], info)
    video.write_bytes(b"owned corrupt media")
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        decode_frames(video, indices, info)
