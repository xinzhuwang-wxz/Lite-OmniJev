"""S3 audio/video plumbing: Noul acceptance, media dispatch, and s3 cohort pools.

CPU-only tests over public interfaces. Rendering through the official Gemma
processor and the real audio/video forward pass need the pinned model and GPU;
those paths are exercised by the GPU runbooks, not here.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from data.schema import Sample
from scripts.s1_dev_eval import (
    counterfactual_match_rate,
    counterfactual_variant,
    load_samples,
    metrics_from_records,
    parse_args,
)
from scripts.s2_media_pools import main as pools_main
from scripts.s2_media_pools import sha256_bytes
from train.trainer import (
    MEDIA_FORWARD_KEYS,
    MEDIA_REQUIRED_KEYS,
    LoopConfig,
    cohort_media_type,
    media_processor_kwargs,
    prepare_media_example,
    smoke_samples,
)

AUDIO_POOLS_VERSION = "clotho-aqa-s3-pools-v1"
VIDEO_POOLS_VERSION = "clevrer-train-video-noul-s3-pools-v1"
S3_SOURCE_DATA_VERSION = "s3-source-candidates-v1"
MIXED_S3_VERSIONS = ["s3-mixed-text-v1", AUDIO_POOLS_VERSION, VIDEO_POOLS_VERSION]


def noul_row(
    index: int,
    *,
    media_type: str | None = "audio",
    modality: list[str] | None = None,
    media_count: int = 1,
    data_version: str = AUDIO_POOLS_VERSION,
    source_dataset: str = "Clotho-AQA",
    target: str = "yes",
) -> dict:
    media = (
        [
            {"type": media_type, "uri": f"clip-{index:05d}.bin", "sha256": "a" * 64}
            for _ in range(media_count)
        ]
        if media_type
        else []
    )
    return {
        "id": f"s3-{media_type or 'text'}-noul-{index:05d}",
        "prefix": {"state": f"evidence {index}", "media": media},
        "questions": [
            {
                "qid": "noul",
                "type": "noul",
                "instruction": f"Is the claim supported by the evidence ({index})?",
                "target": {"kind": "hard", "value": target, "source": "human_label"},
            }
        ],
        "meta": {
            "modality": modality or ([media_type] if media_type else ["text"]),
            "source_dataset": source_dataset,
            "source_revision": "zenodo-6473207",
            "source_sha256": "b" * 64,
            "license": "QA:MIT;media:CC0-1.0",
            "usage": "research",
            "pool": "train",
            "family_id": f"s3:{index}",
            "data_version": data_version,
        },
    }


def text_choice_row(index: int) -> dict:
    return {
        "id": f"text-choice-{index:05d}",
        "prefix": {"state": f"utterance {index}", "media": []},
        "questions": [
            {
                "qid": "intent",
                "type": "choice",
                "instruction": "Pick the matching intent.",
                "options": ["alpha", "beta", "gamma"],
                "target": {"kind": "hard", "value": "B", "source": "human"},
            }
        ],
        "meta": {
            "modality": ["text"],
            "source_dataset": "clinc/clinc_oos",
            "source_revision": "155b9c710419136e17307b80d0a13e68cd46b4ec",
            "source_sha256": "a" * 64,
            "license": "CC-BY-3.0",
            "usage": "commercial",
            "pool": "train",
            "family_id": json.dumps(index),
            "data_version": "clinc-oos-plus-train-v1-s1-pools-v1",
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def media_loop_config(modality: str, media_root: Path, **overrides: object) -> LoopConfig:
    values: dict[str, object] = {
        "sample_count": 2,
        "steps": 1,
        "accumulation": 1,
        "modality": modality,
        "media_root": str(media_root),
    }
    values.update(overrides)
    return LoopConfig.model_validate(values)


def test_loop_config_accepts_audio_and_video_and_maps_media_types(tmp_path: Path) -> None:
    assert media_loop_config("audio", tmp_path).modality == "audio"
    assert media_loop_config("video", tmp_path).modality == "video"
    with pytest.raises(ValidationError):
        media_loop_config("vl", tmp_path)
    assert cohort_media_type("image") == "image"
    assert cohort_media_type("screenshot") == "image"
    assert cohort_media_type("audio") == "audio"
    assert cohort_media_type("video") == "video"


def test_audio_and_video_cohorts_require_matching_media(tmp_path: Path) -> None:
    audio_config = media_loop_config("audio", tmp_path)
    ok = write_jsonl(tmp_path / "audio.jsonl", [noul_row(0), noul_row(1, target="no")])
    samples = smoke_samples(audio_config, ok, expected_data_version=AUDIO_POOLS_VERSION)
    assert [m.type for s in samples for m in s.prefix.media] == ["audio", "audio"]
    video_in_audio = write_jsonl(
        tmp_path / "cross.jsonl", [noul_row(0), noul_row(1, media_type="video", modality=["video"])]
    )
    with pytest.raises(ValueError, match="exactly one audio"):
        smoke_samples(audio_config, video_in_audio, expected_data_version=AUDIO_POOLS_VERSION)
    video_config = media_loop_config("video", tmp_path)
    audio_in_video = write_jsonl(
        tmp_path / "cross2.jsonl",
        [noul_row(1, media_type="video", modality=["video"]), noul_row(0)],
    )
    with pytest.raises(ValueError, match="exactly one video"):
        smoke_samples(video_config, audio_in_video, expected_data_version=AUDIO_POOLS_VERSION)


def test_mixed_smoke_samples_accept_text_audio_and_video_rows(tmp_path: Path) -> None:
    config = LoopConfig.model_validate(
        {
            "sample_count": 3,
            "steps": 1,
            "accumulation": 1,
            "modality": "mixed",
            "media_root": str(tmp_path),
            "expected_data_versions": MIXED_S3_VERSIONS,
        }
    )
    rows = [
        noul_row(0, media_type=None, data_version="s3-mixed-text-v1"),
        noul_row(1, media_type="audio", data_version=AUDIO_POOLS_VERSION),
        noul_row(2, media_type="video", modality=["video"], data_version=VIDEO_POOLS_VERSION),
    ]
    data = write_jsonl(tmp_path / "mixed.jsonl", rows)
    samples = smoke_samples(config, data, expected_data_versions=MIXED_S3_VERSIONS)
    assert sorted(m.type for s in samples for m in s.prefix.media) == ["audio", "video"]
    undeclared = noul_row(3, media_type="audio", modality=["video"])
    with pytest.raises(ValueError, match="declared in metadata"):
        smoke_samples(
            config,
            write_jsonl(tmp_path / "bad.jsonl", [rows[0], undeclared]),
            expected_data_versions=MIXED_S3_VERSIONS,
        )


def test_media_processor_kwargs_match_the_official_protocol() -> None:
    image = object()
    assert media_processor_kwargs("image", image) == {"images": [[image]]}
    audio = [0.0, 1.0]
    assert media_processor_kwargs("audio", audio) == {"audio": [audio], "sampling_rate": 16000}
    loaded_video = {"frames": [1, 2], "metadata": {"fps": 5.0}}
    assert media_processor_kwargs("video", loaded_video) == {
        "videos": [[1, 2]],
        "video_metadata": [{"fps": 5.0}],
        "do_sample_frames": False,
    }
    with pytest.raises(ValueError, match="unsupported media type"):
        media_processor_kwargs("text", object())
    # Every key prepare demands from the processor must be forwardable.
    for keys in MEDIA_REQUIRED_KEYS.values():
        assert set(keys) <= set(MEDIA_FORWARD_KEYS)


def test_loop_config_video_num_frames_must_be_8_or_16(tmp_path: Path) -> None:
    assert media_loop_config("video", tmp_path).video_num_frames == 16
    assert media_loop_config("video", tmp_path, video_num_frames=8).video_num_frames == 8
    assert media_loop_config("video", tmp_path, video_num_frames=16).video_num_frames == 16
    for bad in (7, 9):
        with pytest.raises(ValidationError):
            media_loop_config("video", tmp_path, video_num_frames=bad)


def test_video_frames_cli_choices_and_default() -> None:
    base = ["--model-dir", "m", "--data", "d", "--output-dir", "o", "--report", "r"]
    assert parse_args(base).video_frames == 16
    assert parse_args([*base, "--video-frames", "8"]).video_frames == 8
    assert parse_args([*base, "--video-frames", "16"]).video_frames == 16
    with pytest.raises(SystemExit):
        parse_args([*base, "--video-frames", "7"])
    with pytest.raises(SystemExit):
        parse_args([*base, "--video-frames", "9"])


class NoulTokenizer:
    """Single-token yes/no/U encoding with a stable prompt boundary."""

    TOKENS = {"yes": 2000, "no": 2001, "U": 2002}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        for word, token in self.TOKENS.items():
            if text == word:
                return [token]
        for word, token in self.TOKENS.items():
            if text.startswith("official:") and text.endswith(word):
                return [ord(char) for char in text[: -len(word)]] + [token]
        return [ord(char) for char in text]


class FakeVideoProcessor:
    """Processor stand-in exposing the official video kwarg contract."""

    def __init__(self) -> None:
        self.tokenizer = NoulTokenizer()
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        content = messages[0]["content"]
        assert content[0] == {"type": "video"}
        assert kwargs["enable_thinking"] is False
        return "official:" + content[1]["text"]

    def __call__(self, **kwargs) -> dict:
        assert kwargs["text"] and kwargs["return_tensors"] == "pt"
        assert kwargs["truncation"] is False
        self.calls.append(kwargs)
        return {
            "input_ids": [SimpleNamespace(tolist=lambda: [1, 2, 3])],
            "pixel_values_videos": "pixel-values-videos",
            "video_position_ids": "video-position-ids",
        }


def test_prepare_media_example_threads_video_num_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "media"
    root.mkdir()
    payload = b"video-bytes"
    (root / "clip-00000.bin").write_bytes(payload)
    row = noul_row(0, media_type="video")
    row["prefix"]["media"][0]["uri"] = "clip-00000.bin"
    row["prefix"]["media"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    sample = Sample.model_validate(row)
    decoded: dict[str, object] = {}

    def fake_load_media(path, kind, *, num_frames, max_seconds):
        decoded.update(kind=kind, num_frames=num_frames, max_seconds=max_seconds)
        return {"frames": [f"frame-{i}" for i in range(num_frames)], "metadata": {"fps": 5.0}}

    monkeypatch.setattr("train.trainer.load_media", fake_load_media)
    processor = FakeVideoProcessor()
    example = prepare_media_example(sample, processor, root, max_length=100, video_num_frames=8)
    assert decoded == {"kind": "video", "num_frames": 8, "max_seconds": 60}
    assert example["video_num_frames"] == 8
    assert example["pixel_values_videos"] == "pixel-values-videos"
    assert example["video_position_ids"] == "video-position-ids"
    assert processor.calls[0]["videos"] == [[f"frame-{i}" for i in range(8)]]
    assert processor.calls[0]["video_metadata"] == [{"fps": 5.0}]
    assert processor.calls[0]["do_sample_frames"] is False
    with pytest.raises(ValueError, match="video_num_frames"):
        prepare_media_example(sample, processor, root, max_length=100, video_num_frames=7)


def test_prepare_media_example_dispatch_rejects_bad_media_before_rendering(tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    payload = b"audio-bytes"
    (root / "clip-00000.bin").write_bytes(payload)
    sample = Sample.model_validate(noul_row(0))
    sample.prefix.media[0].uri = "clip-00000.bin"
    sample.prefix.media[0].sha256 = "f" * 64
    with pytest.raises(ValueError) as excinfo:
        prepare_media_example(sample, None, root, max_length=100)
    message = str(excinfo.value)
    assert sample.id in message
    assert str(root / "clip-00000.bin") in message
    doubled = noul_row(1, media_count=2)
    with pytest.raises(ValueError, match="exactly one image, audio or video"):
        prepare_media_example(Sample.model_validate(doubled), None, root, max_length=100)


def test_dev_eval_load_samples_accepts_noul_text_and_media_rows(tmp_path: Path) -> None:
    text = write_jsonl(tmp_path / "text.jsonl", [noul_row(0, media_type=None)])
    assert len(load_samples(text)) == 1
    audio = write_jsonl(tmp_path / "audio.jsonl", [noul_row(1)])
    assert len(load_samples(audio, media_root=tmp_path)) == 1
    with pytest.raises(ValueError, match="media-root"):
        load_samples(audio)
    doubled = write_jsonl(tmp_path / "two.jsonl", [noul_row(2, media_count=2)])
    with pytest.raises(ValueError, match="declared modality"):
        load_samples(doubled, media_root=tmp_path)
    score = noul_row(3, media_type=None)
    score["questions"][0] = {
        "qid": "score",
        "type": "score",
        "instruction": "Rate it.",
        "score_levels": 5,
        "target": {"kind": "hard", "value": 2, "source": "human"},
    }
    with pytest.raises(ValueError, match="Choice"):
        load_samples(write_jsonl(tmp_path / "score.jsonl", [score]))


def test_counterfactual_variant_permutes_choice_and_skips_noul() -> None:
    choice = counterfactual_variant(Sample.model_validate(text_choice_row(0)))
    assert choice is not None
    assert choice.id == text_choice_row(0)["id"]
    assert sorted(choice.questions[0].options or []) == ["alpha", "beta", "gamma"]
    assert counterfactual_variant(Sample.model_validate(noul_row(0))) is None


def test_counterfactual_match_rate_ignores_not_applicable_rows() -> None:
    records = [
        {"variant": "original", "content_match": True},
        {"variant": "not_applicable"},
        {"variant": "counterfactual", "content_match": True},
        {"variant": "counterfactual", "content_match": False},
    ]
    assert counterfactual_match_rate(records) == 0.5
    assert counterfactual_match_rate([{"variant": "not_applicable"}]) is None
    assert counterfactual_match_rate([]) is None


def test_metrics_from_records_indexes_noul_targets_in_row_order() -> None:
    def noul_record(p_yes: float, target: str, prediction: str) -> dict:
        return {
            "probabilities": {"yes": p_yes, "no": 1.0 - p_yes},
            "prediction": prediction,
            "target": target,
            "p_insufficient": 0.01,
        }

    records = [
        noul_record(0.8, "yes", "yes"),
        noul_record(0.3, "yes", "yes"),
    ]
    metrics = metrics_from_records(records)
    # Row order is sorted tokens ("no", "yes"); the second row answers "no".
    assert metrics["accuracy"] == 0.5
    assert metrics["product_accuracy"] == 1.0
    assert metrics["count"] == 2
    choice = {
        "probabilities": {"A": 0.1, "B": 0.9},
        "prediction": "B",
        "target": "B",
        "p_insufficient": 0.0,
    }
    assert metrics_from_records([choice])["accuracy"] == 1.0


POOL_SPECS: dict[str, dict[str, str]] = {
    "audio": {
        "id_prefix": "clotho-aqa-train",
        "family_prefix": "clotho",
        "source_dataset": "Clotho-AQA",
        "source_revision": "zenodo-6473207",
        "license": "QA:MIT;media:CC0-1.0",
        "media_type": "audio",
        "media_dir": "audio",
        "media_suffix": ".wav",
        "state": "Answer the yes/no question using the attached audio.",
        "target_source": "clotho_original_unanimous_three_answers",
        "pools_data_version": AUDIO_POOLS_VERSION,
    },
    "video": {
        "id_prefix": "clevrer-train",
        "family_prefix": "clevrer",
        "source_dataset": "CLEVRER",
        "source_revision": (
            "official-train-json-sha11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
        ),
        "license": "CC0-1.0",
        "media_type": "video",
        "media_dir": "video",
        "media_suffix": ".mp4",
        "state": "Answer the yes/no question using the attached video.",
        "target_source": "clevrer_human_annotation",
        "pools_data_version": VIDEO_POOLS_VERSION,
    },
}


def pool_row(
    cohort: str,
    index: int,
    *,
    media_uri: str,
    media_sha256: str,
    family_id: str | None = None,
    target: str = "yes",
    instruction: str | None = None,
) -> dict:
    spec = POOL_SPECS[cohort]
    return {
        "id": f"{spec['id_prefix']}-{index:05d}",
        "prefix": {
            "state": spec["state"],
            "media": [{"type": spec["media_type"], "uri": media_uri, "sha256": media_sha256}],
        },
        "questions": [
            {
                "qid": "noul",
                "type": "noul",
                "instruction": instruction or f"claim number {index}",
                "target": {"kind": "hard", "value": target, "source": spec["target_source"]},
            }
        ],
        "meta": {
            "modality": [spec["media_type"]],
            "source_dataset": spec["source_dataset"],
            "source_revision": spec["source_revision"],
            "source_sha256": "a" * 64,
            "license": spec["license"],
            "usage": "commercial",
            "pool": "train",
            "family_id": family_id or f"{spec['family_prefix']}:{index}",
            "data_version": S3_SOURCE_DATA_VERSION,
        },
    }


def make_media(media_root: Path, cohort: str, index: int) -> tuple[str, str]:
    spec = POOL_SPECS[cohort]
    name = f"{spec['media_dir']}/{index:05d}{spec['media_suffix']}"
    path = media_root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{cohort}-media-{index}".encode()
    path.write_bytes(payload)
    return name, sha256_bytes(payload)


def cohort_rows(media_root: Path, cohort: str, count: int, *, start: int = 0) -> list[dict]:
    rows: list[dict] = []
    for index in range(start, start + count):
        uri, digest = make_media(media_root, cohort, index)
        rows.append(pool_row(cohort, index, media_uri=uri, media_sha256=digest))
    return rows


def s3_pool_argv(
    tmp_path: Path,
    *,
    cohort: str,
    media_root: Path,
    source_rows: list[dict],
    tag: str,
    dev: Path,
    dev_digest: str,
    holdout_count: int | None,
    calibration_count: int | None,
    license_arg: str | None = None,
) -> list[str]:
    source = tmp_path / f"source-{tag}.jsonl"
    write_jsonl(source, source_rows)
    out = tmp_path / f"out-{tag}"
    argv = [
        "--cohort",
        cohort,
        "--source",
        str(source),
        "--media-root",
        str(media_root),
        "--dev",
        str(dev),
        "--train-out",
        str(out / "train.jsonl"),
        "--calibration-out",
        str(out / "calibration.jsonl"),
        "--holdout-out",
        str(out / "holdout.jsonl"),
        "--allocation-ledger-out",
        str(out / "ledger.jsonl"),
        "--report",
        str(out / "report.json"),
    ]
    if holdout_count is not None:
        argv += ["--holdout-count", str(holdout_count)]
    if calibration_count is not None:
        argv += ["--calibration-count", str(calibration_count)]
    argv += [
        "--expected-dev-sha256",
        dev_digest,
        "--expected-source-data-version",
        S3_SOURCE_DATA_VERSION,
    ]
    if license_arg is not None:
        argv += ["--license", license_arg]
    return argv


def prepare_dev(tmp_path: Path, cohort: str, tag: str) -> Path:
    spec = POOL_SPECS[cohort]
    dev = tmp_path / f"dev-{tag}.jsonl"
    write_jsonl(
        dev,
        [
            pool_row(
                cohort,
                90000,
                family_id=f"{spec['family_prefix']}-dev-0",
                media_uri="dev/unused.bin",
                media_sha256="0" * 64,
            )
        ],
    )
    return dev


def test_audio_cohort_split_uses_s3_keys_and_noul_questions(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "audio", 10)
    dev = prepare_dev(tmp_path, "audio", "audio")
    argv = s3_pool_argv(
        tmp_path,
        cohort="audio",
        media_root=media_root,
        source_rows=rows,
        tag="audio",
        dev=dev,
        dev_digest=sha256_bytes(dev.read_bytes()),
        holdout_count=2,
        calibration_count=3,
        license_arg=POOL_SPECS["audio"]["license"],
    )
    assert pools_main(argv) == 0
    out = tmp_path / "out-audio"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["protocol"] == "s3-audio-pools-v1"
    assert report["config"]["license"] == POOL_SPECS["audio"]["license"]
    ledger = [
        json.loads(line) for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(ledger) == 10
    for row in ledger:
        assert row["allocation_key_sha256"] == sha256_bytes(
            f"s3-audio-pools-v1:42:{row['family_id']}".encode()
        )
    train = [
        Sample.model_validate_json(line)
        for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert train
    for sample in train:
        question = sample.questions[0]
        assert question.type == "noul" and question.options is None
        assert str(question.target.value) in {"yes", "no"}
        assert sample.prefix.media[0].type == "audio"
        assert sample.meta.data_version == AUDIO_POOLS_VERSION


def test_video_cohort_split_with_frozen_license_and_s3_keys(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "video", 6)
    dev = prepare_dev(tmp_path, "video", "video")
    argv = s3_pool_argv(
        tmp_path,
        cohort="video",
        media_root=media_root,
        source_rows=rows,
        tag="video",
        dev=dev,
        dev_digest=sha256_bytes(dev.read_bytes()),
        holdout_count=1,
        calibration_count=2,
    )
    assert pools_main(argv) == 0
    out = tmp_path / "out-video"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["protocol"] == "s3-video-pools-v1"
    assert report["config"]["license"] == "CC0-1.0"
    ledger = [
        json.loads(line) for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in ledger:
        assert row["allocation_key_sha256"] == sha256_bytes(
            f"s3-video-pools-v1:42:{row['family_id']}".encode()
        )
    train = [
        Sample.model_validate_json(line)
        for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert train
    assert all(s.prefix.media[0].type == "video" for s in train)
    assert all(s.meta.data_version == VIDEO_POOLS_VERSION for s in train)


def test_audio_cohort_requires_license_and_reserve_counts(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "audio", 6)
    base = {
        "cohort": "audio",
        "media_root": media_root,
        "source_rows": rows,
        "dev": prepare_dev(tmp_path, "audio", "required"),
        "dev_digest": "0" * 64,
    }
    missing_license = s3_pool_argv(
        tmp_path,
        **base,
        tag="no-license",
        holdout_count=2,
        calibration_count=3,
    )
    with pytest.raises(SystemExit):
        pools_main(missing_license)
    missing_counts = s3_pool_argv(
        tmp_path,
        **base,
        tag="no-counts",
        license_arg=POOL_SPECS["audio"]["license"],
        holdout_count=None,
        calibration_count=None,
    )
    with pytest.raises(SystemExit):
        pools_main(missing_counts)


def test_license_cannot_override_frozen_video_license(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "video", 4)
    argv = s3_pool_argv(
        tmp_path,
        cohort="video",
        media_root=media_root,
        source_rows=rows,
        tag="override",
        dev=prepare_dev(tmp_path, "video", "override"),
        dev_digest="0" * 64,
        holdout_count=1,
        calibration_count=1,
        license_arg="MIT",
    )
    with pytest.raises(SystemExit):
        pools_main(argv)


def test_audio_cohort_rejects_choice_questions(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "audio", 4)
    stray = rows[2]
    stray["questions"][0] = {
        "qid": "choice",
        "type": "choice",
        "instruction": "Pick one.",
        "options": ["alpha", "beta"],
        "target": {"kind": "hard", "value": "B", "source": "human"},
    }
    dev = prepare_dev(tmp_path, "audio", "choice")
    argv = s3_pool_argv(
        tmp_path,
        cohort="audio",
        media_root=media_root,
        source_rows=rows,
        tag="choice",
        dev=dev,
        dev_digest=sha256_bytes(dev.read_bytes()),
        holdout_count=1,
        calibration_count=1,
        license_arg=POOL_SPECS["audio"]["license"],
    )
    assert pools_main(argv) == 1
    report = json.loads((tmp_path / "out-choice" / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "Noul" in str(report["blocker"])
    assert not (tmp_path / "out-choice" / "train.jsonl").exists()


def test_noul_family_target_conflict_is_rejected(tmp_path: Path) -> None:
    media_root = tmp_path / "media"
    rows = cohort_rows(media_root, "audio", 4)
    uri, digest = make_media(media_root, "audio", 4)
    # 同 family 同题面、不同 target —— 真冲突。
    conflicting = pool_row(
        "audio",
        4,
        family_id=rows[0]["meta"]["family_id"],
        media_uri=uri,
        media_sha256=digest,
        target="no",
        instruction=rows[0]["questions"][0]["instruction"],
    )
    rows.append(conflicting)
    dev = prepare_dev(tmp_path, "audio", "conflict")
    argv = s3_pool_argv(
        tmp_path,
        cohort="audio",
        media_root=media_root,
        source_rows=rows,
        tag="conflict",
        dev=dev,
        dev_digest=sha256_bytes(dev.read_bytes()),
        holdout_count=1,
        calibration_count=1,
        license_arg=POOL_SPECS["audio"]["license"],
    )
    assert pools_main(argv) == 1
    report = json.loads((tmp_path / "out-conflict" / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert rows[0]["meta"]["family_id"] in str(report["blocker"])
    assert not (tmp_path / "out-conflict" / "train.jsonl").exists()
