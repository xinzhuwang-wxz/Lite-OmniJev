"""Explicit loader parity and experimental temporal sampling CPU tests."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from data.schema import Sample
from train.multimodal import file_hash, load_media


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


def test_default_and_explicit_uniform_pixels_and_metadata_are_identical(video):
    pytest.importorskip("PIL")
    implicit = load_media(video, "video", num_frames=16, max_seconds=60)
    explicit = load_media(
        video,
        "video",
        num_frames=16,
        max_seconds=60,
        frame_indices=implicit["metadata"]["frames_indices"],
    )
    assert implicit["metadata"] == explicit["metadata"]
    assert [frame.tobytes() for frame in implicit["frames"]] == [
        frame.tobytes() for frame in explicit["frames"]
    ]
    chosen = list(range(16))
    alternate = load_media(video, "video", num_frames=16, max_seconds=60, frame_indices=chosen)
    assert alternate["metadata"]["frames_indices"] == chosen
    for key in ("fps", "total_num_frames", "duration"):
        assert alternate["metadata"][key] == implicit["metadata"][key]


@pytest.mark.parametrize(
    "indices",
    [
        list(range(15)),
        [0] * 16,
        list(reversed(range(16))),
        list(range(15)) + [40],
        [-1] + list(range(1, 16)),
        [False] + list(range(1, 16)),
        [0.0] + list(range(1, 16)),
    ],
)
def test_explicit_invalid_selection_rejected(video, indices):
    with pytest.raises(ValueError, match="explicit video indices"):
        load_media(video, "video", num_frames=16, max_seconds=60, frame_indices=indices)


def test_explicit_selection_only_video_and_fixed_sixteen(video):
    with pytest.raises(ValueError, match="reserved for video"):
        load_media(video, "image", num_frames=16, max_seconds=60, frame_indices=list(range(16)))
    with pytest.raises(ValueError, match="explicit video indices"):
        load_media(video, "video", num_frames=8, max_seconds=60, frame_indices=list(range(16)))


@pytest.fixture
def experiment():
    path = Path(".scratch/s4/evaluate-video-motion-coverage.py")
    spec = importlib.util.spec_from_file_location("motion_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def samples(video):
    return [
        Sample.model_validate(
            {
                "id": f"video-{i}",
                "prefix": {
                    "state": "state",
                    "media": [{"type": "video", "uri": video.name, "sha256": file_hash(video)}],
                },
                "questions": [
                    {
                        "qid": "q",
                        "type": "noul",
                        "instruction": "Is it moving before the event?"
                        if i < 100
                        else "Is object present?",
                        "target": {"kind": "hard", "value": "yes", "source": "fixture_gold"},
                    }
                ],
                "meta": {
                    "modality": ["video"],
                    "source_dataset": "fixture",
                    "source_revision": "r1",
                    "source_sha256": "a" * 64,
                    "license": "MIT",
                    "usage": "research",
                    "pool": "dev",
                    "family_id": str(i),
                    "data_version": "test",
                },
            }
        )
        for i in range(500)
    ]


def test_temporal_selection_is_stable_and_private_pools_rejected(video, experiment):
    rows = samples(video)
    selected = experiment.select_samples(rows)
    assert len(selected) == 50
    assert {row.id for row in selected} <= {f"video-{i}" for i in range(100)}
    assert [row.id for row in selected] == [
        row.id for row in experiment.select_samples(list(reversed(rows)))
    ]
    rows[0] = rows[0].model_copy(
        update={"meta": rows[0].meta.model_copy(update={"pool": "holdout"})}
    )
    with pytest.raises(Exception, match="pool=dev"):
        experiment.select_samples(rows)


def test_real_cpu_prepare_deduplicates_video_and_hook_binds_indices(
    video, tmp_path, monkeypatch, experiment
):
    pytest.importorskip("numpy")
    selected = experiment.select_samples(samples(video))
    data = tmp_path / "dev.jsonl"
    data.write_text("CPU source binding fixture")
    source = tmp_path / "source.json"
    source.write_text("{}")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"cpu identity fixture")
    (adapter / "adapter_config.json").write_text("{}")
    monkeypatch.setattr(experiment, "SOURCE", source)
    monkeypatch.setattr(experiment, "DATA_SHA", file_hash(data))
    monkeypatch.setattr(experiment, "ADAPTER", adapter)
    monkeypatch.setattr(experiment, "ADAPTER_SHA", file_hash(adapter / "adapter_model.safetensors"))
    monkeypatch.setattr(
        experiment, "source_inputs", lambda: (data, video.parent, Path("model-revision"), selected)
    )
    output = tmp_path / "pilot"
    monkeypatch.setattr(sys, "argv", ["experiment", "--output", str(output)])
    assert experiment.main() == 0
    report = json.loads((output / "preparation.json").read_text())
    assert report["status"] == "prepared" and report["gpu_requested"] is False
    assert len(report["selected"]) == 50 and len(report["manifests"]) == 1
    bound = next(iter(report["manifests"].values()))
    manifest = json.loads(Path(bound["path"]).read_text())

    # Real explicit RGB decode parity is tested separately; this CPU integration
    # checks the experimental routing contract without requiring local Pillow.
    def selected_loader(path, kind, **kwargs):
        return {
            "metadata": {key: manifest[key] for key in ("fps", "total_num_frames", "duration")}
            | {"frames_indices": kwargs["frame_indices"]}
        }

    monkeypatch.setattr(experiment, "load_media", selected_loader)
    for policy, key in [("uniform", "uniform_indices"), ("motion", "adaptive_indices")]:
        decoded = experiment.hook_loader({file_hash(video): manifest}, policy)(
            video, "video", num_frames=16, max_seconds=60
        )
        assert decoded["metadata"]["frames_indices"] == manifest[key]
    assert not (output / "uniform").exists() and not (output / "motion").exists()
