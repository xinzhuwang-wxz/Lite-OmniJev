from typing import Any

import pytest

from data.schema import Sample
from scripts.s0_development_bundle import exclusions


def sample(name: str, media: list[tuple[str, str]], pool: str = "dev") -> Sample:
    raw: dict[str, Any] = {
        "id": name,
        "prefix": {
            "state": "state",
            "media": [
                {"type": kind, "uri": f"{kind}/{digest}", "sha256": digest}
                for kind, digest in media
            ],
        },
        "questions": [
            {
                "qid": "q",
                "type": "noul",
                "instruction": "Is it true?",
                "target": {"kind": "hard", "value": "yes", "source": "original"},
            }
        ],
        "meta": {
            "modality": sorted({kind for kind, _ in media}) or ["text"],
            "source_dataset": name,
            "source_revision": "r1",
            "source_sha256": "0" * 64,
            "license": "MIT",
            "usage": "commercial",
            "pool": pool,
            "family_id": "same-id-in-distinct-datasets",
            "data_version": "v1",
        },
    }
    return Sample.model_validate(raw)


def test_shared_media_aliases_join_original_and_derived_families() -> None:
    image = sample("original-image", [("image", "a" * 64)])
    audio = sample("original-audio", [("audio", "b" * 64)])
    combined = sample("derived", [("image", "a" * 64), ("audio", "b" * 64)])
    unrelated = sample("original-text", [])
    rows = exclusions(
        {"image": [image], "audio": [audio], "crossmodal": [combined], "text": [unrelated]}
    )
    components = {row["source_dataset"]: row["component_id"] for row in rows}
    assert components["original-image"] == components["original-audio"] == components["derived"]
    assert components["original-text"] != components["derived"]
    assert len(rows) == 4
    assert rows == exclusions(
        {"text": [unrelated], "crossmodal": [combined], "audio": [audio], "image": [image]}
    )


def test_unbound_crossmodal_components_cannot_pass() -> None:
    with pytest.raises(ValueError, match="both original"):
        exclusions({"crossmodal": [sample("derived", [("image", "a" * 64), ("audio", "b" * 64)])]})


def test_training_sample_cannot_enter_dev_exclusion_bundle() -> None:
    with pytest.raises(ValueError, match="belong to dev"):
        exclusions({"text": [sample("train", [], pool="train")]})
