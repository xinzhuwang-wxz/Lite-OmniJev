"""S2 multimodal plumbing: config, media verification, counterfactual permutation.

CPU-only tests over public interfaces. Paths that need torch or real model
weights (processor rendering, pixel_values forward, GPU determinism) are not
unit-tested here; they are exercised by the real GPU runbooks.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from data.schema import HardTarget, Sample
from scripts.s1_dev_eval import (
    build_counterfactual,
    counterfactual_seed,
    load_samples,
    prediction_content,
    record_for,
)
from train.trainer import LoopConfig, smoke_samples, verified_media_file


def image_row(index: int, *, media_count: int = 1, modality: list[str] | None = None) -> dict:
    media = [
        {"type": "image", "uri": f"img-{index:02d}.png", "sha256": "b" * 64}
        for _ in range(media_count)
    ]
    return {
        "id": f"aokvqa-train-{index:05d}",
        "prefix": {"state": f"scene {index}", "media": media},
        "questions": [
            {
                "qid": "answer",
                "type": "choice",
                "instruction": "Which answer is best supported by the image?",
                "options": ["alpha", "beta", "gamma"],
                "target": {"kind": "hard", "value": "B", "source": "human"},
            }
        ],
        "meta": {
            "modality": modality or ["image"],
            "source_dataset": "aokvqa",
            "source_revision": "42e8c44f",
            "source_sha256": "a" * 64,
            "license": "Apache-2.0",
            "usage": "research",
            "pool": "train",
            "family_id": json.dumps(index),
            "data_version": "s2-image-train-v1-pools-v1",
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def image_loop_config(media_root: str | None) -> LoopConfig:
    return LoopConfig(
        sample_count=2,
        steps=1,
        accumulation=1,
        modality="image",
        media_root=media_root,
    )


def test_loop_config_requires_media_root_for_media_cohorts(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="media_root"):
        image_loop_config(None)
    with pytest.raises(ValidationError, match="media_root"):
        LoopConfig(sample_count=2, steps=1, accumulation=1, modality="screenshot", media_root="  ")
    assert image_loop_config(str(tmp_path)).media_root == str(tmp_path)
    assert LoopConfig(sample_count=2, steps=1, accumulation=1).media_root is None


def test_image_cohorts_require_a_frozen_data_file() -> None:
    with pytest.raises(ValueError, match="frozen data file"):
        smoke_samples(image_loop_config("/tmp/nowhere"))


def test_text_training_rejects_media_samples(tmp_path: Path) -> None:
    config = LoopConfig(sample_count=2, steps=1, accumulation=1)
    data = write_jsonl(tmp_path / "media.jsonl", [image_row(0), image_row(1)])
    with pytest.raises(ValueError, match="text-only"):
        smoke_samples(config, data)


def test_image_training_requires_exactly_one_declared_image(tmp_path: Path) -> None:
    config = image_loop_config(str(tmp_path))
    data = write_jsonl(tmp_path / "ok.jsonl", [image_row(0), image_row(1)])
    assert all(len(s.prefix.media) == 1 for s in smoke_samples(config, data))
    for name, rows in (
        ("none", [image_row(0, media_count=0), image_row(1, media_count=0)]),
        ("two", [image_row(0, media_count=2), image_row(1)]),
    ):
        with pytest.raises(ValueError, match="exactly one image"):
            smoke_samples(config, write_jsonl(tmp_path / f"{name}.jsonl", rows))
    audio = image_row(2)
    audio["prefix"]["media"] = [{"type": "audio", "uri": "a.wav", "sha256": "c" * 64}]
    audio["meta"]["modality"] = ["audio"]
    with pytest.raises(ValueError, match="exactly one image"):
        smoke_samples(config, write_jsonl(tmp_path / "audio.jsonl", [image_row(0), audio]))


def test_verified_media_file_checks_streaming_sha(tmp_path: Path) -> None:
    root = tmp_path / "media"
    root.mkdir()
    payload = b"image-bytes"
    (root / "a.png").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert verified_media_file(root, "a.png", digest) == root / "a.png"
    (root / "a.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verified_media_file(root, "a.png", digest)


def test_counterfactual_seed_is_derived_from_id_sha256() -> None:
    expected = int.from_bytes(hashlib.sha256(b"cf-1").digest()[:8], "big")
    assert counterfactual_seed("cf-1") == expected == counterfactual_seed("cf-1")
    assert counterfactual_seed("cf-2") != expected


def test_build_counterfactual_is_deterministic_and_remaps_target() -> None:
    sample = Sample.model_validate(image_row(0))
    seed = counterfactual_seed(sample.id)
    first = build_counterfactual(sample, seed)
    assert first.model_dump_json() == build_counterfactual(sample, seed).model_dump_json()
    assert first.id == sample.id
    question = first.questions[0]
    assert question.type == "choice"
    assert sorted(question.options or []) == ["alpha", "beta", "gamma"]
    assert len(question.options or []) == len(sample.questions[0].options or [])
    target = question.target
    assert isinstance(target, HardTarget)
    # The target letter always names the same content after permutation.
    assert str(target.value) == chr(ord("A") + (question.options or []).index("beta"))


def test_build_counterfactual_requires_choice_with_letter_target() -> None:
    row = image_row(0)
    row["questions"][0] = {
        "qid": "score",
        "type": "score",
        "instruction": "Rate the scene.",
        "score_levels": 5,
        "target": {"kind": "hard", "value": 2, "source": "human"},
    }
    with pytest.raises(ValueError, match="Choice"):
        build_counterfactual(Sample.model_validate(row), 7)


def test_dev_eval_load_samples_media_rules(tmp_path: Path) -> None:
    data = write_jsonl(tmp_path / "dev.jsonl", [image_row(0)])
    with pytest.raises(ValueError, match="media-root"):
        load_samples(data)
    assert len(load_samples(data, media_root=tmp_path)) == 1
    two = write_jsonl(tmp_path / "two.jsonl", [image_row(1, media_count=2)])
    with pytest.raises(ValueError, match="declared modality"):
        load_samples(two, media_root=tmp_path)


def test_prediction_content_maps_letters_and_keeps_u() -> None:
    options = ["alpha", "beta", "gamma"]
    assert prediction_content("B", options) == "beta"
    assert prediction_content("U", options) == "U"


def test_record_assembly_needs_no_torch() -> None:
    sample = Sample.model_validate(image_row(0))
    readout = SimpleNamespace(
        probabilities={"A": 0.6, "B": 0.3, "C": 0.1},
        joint_probabilities={"A": 0.5, "B": 0.25, "C": 0.15, "U": 0.1},
        p_insufficient=0.1,
        confidence=0.4,
        prediction="B",
    )
    record = record_for(sample, readout, {"A": 1.0, "B": 2.0, "C": 0.5, "U": -1.0})
    assert record["target"] == "B"
    assert record["prediction"] == "B"
    assert record["n_options"] == 3
    assert "torch" not in record
