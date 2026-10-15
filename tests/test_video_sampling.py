"""CPU sampling mathematics and real synthetic-container decode evidence."""

import json
import shutil
import subprocess

import pytest

from scripts.video_motion_manifest import build_manifest, main
from train.multimodal import file_hash
from train.video_sampling import motion_coverage_indices


def test_segment_maxima_and_earliest_tie_preserve_coverage():
    # Interior segments are1..4 and5..8; global top2 would incorrectly choose1,2.
    assert motion_coverage_indices([0, 9, 8, 0, 0, 0, 0, 0, 0, 0], 4) == [0, 1, 5, 9]
    assert motion_coverage_indices([0] * 30) == [0, *range(1, 29, 2), 29]
    assert motion_coverage_indices([0] * 16) == list(range(16))
    assert motion_coverage_indices([0, 1, 2, 3], 2) == [0, 3]


@pytest.mark.parametrize("count", [16, 17, 29, 31, 150, 7200])
def test_every_segment_has_exactly_one_unique_interior_selection(count):
    scores = [(index * 17) % 29 for index in range(count)]
    indices = motion_coverage_indices(scores)
    assert len(indices) == len(set(indices)) == 16
    assert indices == sorted(indices) and indices[0] == 0 and indices[-1] == count - 1
    for segment, selected in enumerate(indices[1:-1]):
        start = 1 + segment * (count - 2) // 14
        stop = 1 + (segment + 1) * (count - 2) // 14
        assert start <= selected < stop
        assert selected == max(range(start, stop), key=lambda index: scores[index])


@pytest.mark.parametrize(
    "scores,number",
    [
        ([0] * 15, 16),
        ([0] * 16, 1),
        ([0] * 16, True),
        ([0] * 16, 16.0),
        ([0, float("nan")] + [0] * 14, 16),
        ([0, float("inf")] + [0] * 14, 16),
        ([0, -1] + [0] * 14, 16),
        ([0, True] + [0] * 14, 16),
        ([0, "2"] + [0] * 14, 16),
    ],
)
def test_invalid_counts_and_scores_rejected(scores, number):
    with pytest.raises(ValueError):
        motion_coverage_indices(scores, number)


@pytest.fixture
def video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe unavailable")
    path = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=20:duration=2",
            "-frames:v",
            "40",
            "-c:v",
            "mpeg4",
            "-threads",
            "1",
            str(path),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )
    return path


def test_real_synthetic_video_cpu_manifest_is_deterministic_and_has_no_pixels(video):
    pytest.importorskip("numpy")
    digest = file_hash(video)
    first = build_manifest(video, digest)
    second = build_manifest(video, digest)
    assert first["status"] == "prepared" and first["gpu_used"] is False
    assert first["total_num_frames"] == 40 and first["fps"] == 20
    assert first["duration"] == 2
    assert first["uniform_indices"] == [round(i * 39 / 15) for i in range(16)]
    assert first["adaptive_indices"] == second["adaptive_indices"]
    assert first["per_frame_motion"] == second["per_frame_motion"]
    assert len(first["per_frame_motion"]) == 40 and first["per_frame_motion"][0] == 0
    assert all(score >= 0 for score in first["per_frame_motion"])
    assert any(score > 0 for score in first["per_frame_motion"])
    assert first["video_sha256"] == digest == file_hash(video)
    assert len(first["adaptive_indices"]) == len(set(first["adaptive_indices"])) == 16
    assert "pixel_values" not in first and "frames" not in first
    assert first["decode_seconds"] >= 0 and first["duration_seconds"] >= first["decode_seconds"]


def test_sha_failure_receipt_and_exclusive_output_do_not_modify_video(video, tmp_path):
    original = file_hash(video)
    output = tmp_path / "failure.json"
    args = ["--video", str(video), "--expected-sha256", "0" * 64, "--output", str(output)]
    assert main(args) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert "SHA256 mismatch" in report["error"] and file_hash(video) == original
    prior = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert output.read_bytes() == prior


def test_playlists_rejected_by_shared_container_policy(tmp_path):
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe unavailable")
    playlist = tmp_path / "unsafe.m3u8"
    playlist.write_text("#EXTM3U\n#EXTINF:1,\nhttps://example.invalid/x.ts\n")
    with pytest.raises(subprocess.CalledProcessError):
        build_manifest(playlist, file_hash(playlist))


@pytest.mark.parametrize(
    "update",
    [
        {"duration": "61"},
        {"nb_frames": "15"},
        {"nb_frames": "7201"},
        {"avg_frame_rate": "0/1"},
        {"width": 40000001},
    ],
)
def test_metadata_limits_stop_before_decode(tmp_path, monkeypatch, update):
    path = tmp_path / "fixture.mp4"
    path.write_bytes(b"metadata contract fixture")
    fields = {
        "codec_type": "video",
        "avg_frame_rate": "16/1",
        "nb_frames": "16",
        "width": 64,
        "height": 64,
    }
    fields.update({key: value for key, value in update.items() if key != "duration"})

    def probe(command, **kwargs):
        assert command[0] == "ffprobe"
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert "file,pipe" in command and "-format_whitelist" in command
        kwargs["stdout"].write(
            json.dumps(
                {"format": {"duration": update.get("duration", "1")}, "streams": [fields]}
            ).encode()
        )

    monkeypatch.setattr(subprocess, "run", probe)
    with pytest.raises(ValueError):
        build_manifest(path, file_hash(path))


def test_incorrect_decoded_frame_count_rejected_and_output_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "fixture.mp4"
    path.write_bytes(b"decode contract fixture")

    def decode(command, **kwargs):
        assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["stderr"] == subprocess.DEVNULL
        assert "capture_output" not in kwargs
        if command[0] == "ffprobe":
            kwargs["stdout"].write(
                json.dumps(
                    {
                        "format": {"duration": "1"},
                        "streams": [
                            {
                                "codec_type": "video",
                                "avg_frame_rate": "16/1",
                                "nb_frames": "16",
                                "width": 64,
                                "height": 64,
                            }
                        ],
                    }
                ).encode()
            )
        else:
            assert command[command.index("-frames:v") + 1] == "7201"
            assert command[command.index("-threads") + 1] == "1"
            assert command[command.index("-vf") + 1] == "scale=64:64"
            kwargs["stdout"].write(b"\0" * (15 * 64 * 64))

    monkeypatch.setattr(subprocess, "run", decode)
    with pytest.raises(ValueError, match="decoded frame bytes"):
        build_manifest(path, file_hash(path))
