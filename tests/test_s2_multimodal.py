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
from train.trainer import (
    LoopConfig,
    prepare_media_example,
    prepare_mixed_example,
    smoke_samples,
    verified_media_file,
)

MIXED_VERSIONS = ["s2-image-train-v1-pools-v1", "s2-mixed-text-v1"]


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


def text_row(index: int) -> dict:
    row = image_row(index, media_count=0, modality=["text"])
    row["id"] = f"clinc-mixed-{index:05d}"
    row["meta"]["source_dataset"] = "clinc/clinc_oos"
    row["meta"]["data_version"] = "s2-mixed-text-v1"
    return row


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


def mixed_loop_config(tmp_path: Path, **overrides: object) -> LoopConfig:
    values: dict[str, object] = {
        "sample_count": 2,
        "steps": 1,
        "accumulation": 1,
        "modality": "mixed",
        "media_root": str(tmp_path),
        "expected_data_versions": MIXED_VERSIONS,
    }
    values.update(overrides)
    return LoopConfig.model_validate(values)


def test_loop_config_mixed_requires_media_root_and_versions(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="media_root"):
        mixed_loop_config(tmp_path, media_root=None)
    with pytest.raises(ValidationError, match="expected_data_versions"):
        mixed_loop_config(tmp_path, expected_data_versions=None)
    with pytest.raises(ValidationError, match="List should have at least 1"):
        mixed_loop_config(tmp_path, expected_data_versions=[])
    scalar = mixed_loop_config(tmp_path, expected_data_versions=None, expected_data_version="v1")
    assert scalar.expected_data_version == "v1"
    with pytest.raises(ValidationError, match="mixed"):
        LoopConfig(sample_count=2, steps=1, accumulation=1, expected_data_versions=["v1"])


def test_mixed_smoke_samples_accept_text_and_image_rows(tmp_path: Path) -> None:
    config = mixed_loop_config(tmp_path)
    data = write_jsonl(tmp_path / "mixed.jsonl", [text_row(0), image_row(1)])
    samples = smoke_samples(config, data, expected_data_versions=MIXED_VERSIONS)
    assert sorted(len(s.prefix.media) for s in samples) == [0, 1]


def test_mixed_smoke_samples_rejects_inconsistent_rows(tmp_path: Path) -> None:
    config = mixed_loop_config(tmp_path)
    # A text-declared row carrying media is refused by the schema itself.
    bad_text = image_row(0)
    bad_text["meta"]["modality"] = ["text"]
    with pytest.raises(ValueError, match="declared in metadata"):
        smoke_samples(
            config,
            write_jsonl(tmp_path / "text-media.jsonl", [bad_text, image_row(1)]),
            expected_data_versions=MIXED_VERSIONS,
        )
    with pytest.raises(ValueError, match="exactly one media"):
        smoke_samples(
            config,
            write_jsonl(tmp_path / "two.jsonl", [image_row(0, media_count=2), text_row(1)]),
            expected_data_versions=MIXED_VERSIONS,
        )
    # An image-declared row without media is not a legal text row either.
    bare = image_row(2, media_count=0)
    with pytest.raises(ValueError, match="text-only"):
        smoke_samples(
            config,
            write_jsonl(tmp_path / "bare.jsonl", [bare, text_row(1)]),
            expected_data_versions=MIXED_VERSIONS,
        )


def test_mixed_smoke_samples_check_frozen_versions(tmp_path: Path) -> None:
    config = mixed_loop_config(
        tmp_path, expected_data_versions=None, expected_data_version="s2-mixed-text-v1"
    )
    text = write_jsonl(tmp_path / "text.jsonl", [text_row(0), text_row(1)])
    assert len(smoke_samples(config, text, expected_data_version="s2-mixed-text-v1")) == 2
    mixed = write_jsonl(tmp_path / "mixed.jsonl", [text_row(0), image_row(1)])
    with pytest.raises(ValueError, match="frozen allocation"):
        smoke_samples(config, mixed, expected_data_version="s2-mixed-text-v1")
    with pytest.raises(ValueError, match="expected_data_versions"):
        smoke_samples(config, text)


def test_mixed_prepare_routes_each_sample_by_its_own_media(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    routed: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "train.trainer.prepare_examples",
        lambda samples, tokenizer, max_length: (
            routed.append(("text", samples[0].id))
            or [{"id": samples[0].id, "input_ids": [1], "token_ids": [2], "target": [1.0]}]
        ),
    )
    monkeypatch.setattr(
        "train.trainer.prepare_media_example",
        lambda sample, processor, media_root, *, max_length: (
            routed.append(("media", sample.id))
            or {
                "id": sample.id,
                "input_ids": [2],
                "pixel_values": object(),
                "token_ids": [3],
                "answer_token_ids": {},
                "target": [1.0],
            }
        ),
    )
    text = Sample.model_validate(text_row(0))
    image = Sample.model_validate(image_row(1))
    examples = [
        prepare_mixed_example(
            sample, tokenizer=None, processor=None, media_root=tmp_path, max_length=100
        )
        for sample in (text, image)
    ]
    assert routed == [("text", text.id), ("media", image.id)]
    assert "pixel_values" not in examples[0]
    assert "pixel_values" in examples[1]


def test_verified_media_file_multi_root_selects_by_sha(tmp_path: Path) -> None:
    import hashlib

    from train.trainer import verified_media_file

    tree_a = tmp_path / "a"
    tree_b = tmp_path / "b"
    tree_a.mkdir()
    tree_b.mkdir()
    payload_a = b"image-bytes-a"
    payload_b = b"image-bytes-b"
    (tree_a / "train").mkdir()
    (tree_b / "train").mkdir()
    (tree_a / "train" / "x.jpg").write_bytes(payload_a)
    (tree_b / "train" / "x.jpg").write_bytes(payload_b)
    sha_a = hashlib.sha256(payload_a).hexdigest()
    sha_b = hashlib.sha256(payload_b).hexdigest()
    hit = verified_media_file(tree_a, "train/x.jpg", sha_b, extra_roots=[tree_b])
    assert hit == tree_b / "train" / "x.jpg"
    hit_a = verified_media_file(tree_a, "train/x.jpg", sha_a)
    assert hit_a == tree_a / "train" / "x.jpg"
    try:
        verified_media_file(tree_a, "train/x.jpg", "f" * 64, extra_roots=[tree_b])
    except ValueError as exc:
        assert "media resolution failed" in str(exc)
        assert "train/x.jpg" in str(exc)
        assert str(tree_a / "train" / "x.jpg") in str(exc)
        assert str(tree_b / "train" / "x.jpg") in str(exc)
    else:
        raise AssertionError("expected multi-root failure")


def test_loop_config_media_roots_rules(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        LoopConfig(
            sample_count=2,
            steps=1,
            accumulation=1,
            modality="mixed",
            media_root=str(tmp_path),
            media_roots=[str(tmp_path / "a"), str(tmp_path / "b")],
            expected_data_versions=MIXED_VERSIONS,
        )
    listed = LoopConfig(
        sample_count=2,
        steps=1,
        accumulation=1,
        modality="image",
        media_roots=[str(tmp_path / "a"), str(tmp_path / "b")],
    )
    assert listed.media_roots == [str(tmp_path / "a"), str(tmp_path / "b")]
    with pytest.raises(ValidationError, match="nonempty strings"):
        LoopConfig(sample_count=2, steps=1, accumulation=1, modality="image", media_roots=["  "])
    with pytest.raises(ValidationError, match="reserved for media"):
        LoopConfig(sample_count=2, steps=1, accumulation=1, media_roots=[str(tmp_path)])


def test_prepare_media_failure_names_sample_and_every_candidate(tmp_path: Path) -> None:
    tree_a = tmp_path / "coco"
    tree_b = tmp_path / "gui"
    tree_a.mkdir()
    tree_b.mkdir()
    (tree_a / "shot.png").write_bytes(b"coconut-bytes")
    (tree_b / "shot.png").write_bytes(b"gui-bytes")
    sample = Sample.model_validate(image_row(0))
    sample.prefix.media[0].uri = "shot.png"
    sample.prefix.media[0].sha256 = "f" * 64
    with pytest.raises(ValueError) as excinfo:
        prepare_media_example(sample, None, tree_a, tree_b, max_length=100)
    message = str(excinfo.value)
    assert sample.id in message
    assert str(tree_a / "shot.png") in message
    assert str(tree_b / "shot.png") in message
