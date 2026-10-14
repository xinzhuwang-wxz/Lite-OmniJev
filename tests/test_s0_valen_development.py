from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import pytest

from data.schema import HardTarget, Media, Metadata, Prefix, Question, Sample
from scripts import s0_valen_development as dev


def _sample(*, media: list[Media] | None = None, question: Question | None = None) -> Sample:
    media = media or []
    modalities = cast(
        list[Literal["text", "image", "audio", "video"]], ["text", *[item.type for item in media]]
    )
    return Sample(
        id="sample-1",
        prefix=Prefix(state="visible state", media=media),
        questions=[
            question
            or Question(
                qid="q",
                type="choice",
                instruction="Choose",
                options=["Alpha", "Beta"],
                target=HardTarget(kind="hard", value="A", source="fixture"),
            )
        ],
        meta=Metadata(
            modality=modalities,
            source_dataset="fixture",
            source_revision="r1",
            source_sha256="0" * 64,
            license="MIT",
            usage="research",
            pool="dev",
            family_id="family-1",
            data_version="v1",
        ),
    )


def test_audio_and_crossmodal_are_explicitly_unsupported() -> None:
    with pytest.raises(dev.ValenDevelopmentError, match="audio"):
        dev.ensure_supported_cohort("audio")
    with pytest.raises(dev.ValenDevelopmentError, match="crossmodal"):
        dev.ensure_supported_cohort("crossmodal")


def test_build_state_messages_embeds_loaded_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image_obj = object()
    media = Media(type="image", uri="x.png", sha256="1" * 64)
    sample = _sample(media=[media])

    monkeypatch.setattr(dev, "resolve_media", lambda root, uri, sha256, max_bytes: tmp_path / uri)
    monkeypatch.setattr(
        dev,
        "load_media",
        lambda path, kind, num_frames, max_seconds: image_obj,
    )

    messages, identity = dev.build_state_messages(sample, cohort="image", media_root=tmp_path)

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": "visible state"},
            ],
        }
    ]
    assert identity["media"] == [{"type": "image", "uri": "x.png", "sha256": "1" * 64}]
    assert identity["media_kwargs"] == dev.VALEN_MEDIA_KWARGS


def test_build_state_messages_embeds_video_frames_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    frames = [object(), object()]
    metadata = {"frames_indices": [0, 1], "fps": 30.0}
    media = Media(type="video", uri="x.mp4", sha256="2" * 64)
    sample = _sample(media=[media])

    monkeypatch.setattr(dev, "resolve_media", lambda root, uri, sha256, max_bytes: tmp_path / uri)
    monkeypatch.setattr(
        dev,
        "load_media",
        lambda path, kind, num_frames, max_seconds: {"frames": frames, "metadata": metadata},
    )

    messages, identity = dev.build_state_messages(sample, cohort="video", media_root=tmp_path)

    assert messages[0]["content"][0] == {"type": "video", "video": frames}
    assert identity["video_metadata"] == [metadata]
    assert identity["media_kwargs"]["video_metadata"] == [metadata]
    assert identity["media_kwargs"]["do_sample_frames"] is False


def test_valen_probabilities_noul_has_null_pu_and_yes_no_order() -> None:
    question = Question(qid="n", type="noul", instruction="Is true?")

    result = dev.valen_probabilities(question, [3.0, 1.0])

    assert result["p_U"] is None
    assert result["u_supported"] is False
    assert result["prediction"] == "yes"
    assert result["probabilities_ordered"] == [
        result["conditional_without_u"]["yes"],
        result["conditional_without_u"]["no"],
    ]


def test_compare_exact_checks_logits_probabilities_and_input_hash() -> None:
    left: list[dict[str, Any]] = [
        {
            "id": "a",
            "raw_logits": [1.0, 2.0],
            "conditional_without_u": {"A": 0.2, "B": 0.8},
            "input_summary_sha256": "x",
            "position_summary_sha256": "p",
            "prediction": "B",
        }
    ]
    right = [dict(left[0], raw_logits=[1.0, 2.1])]

    report = dev.compare_exact(left, right)

    assert report["passed"] is False
    assert report["first_mismatches"][0]["kind"] == "raw_logits"


def test_compare_exact_checks_position_hash() -> None:
    left: list[dict[str, Any]] = [
        {
            "id": "a",
            "raw_logits": [1.0, 2.0],
            "conditional_without_u": {"A": 0.2, "B": 0.8},
            "input_summary_sha256": "x",
            "position_summary_sha256": "p1",
            "prediction": "B",
        }
    ]
    right = [dict(left[0], position_summary_sha256="p2")]

    report = dev.compare_exact(left, right)

    assert report["passed"] is False
    assert report["first_mismatches"][0]["kind"] == "position_summary_sha256"
