"""Synthetic public-interface tests; no sealed data is opened."""

import hashlib
import json
from itertools import combinations
from typing import Any

import pytest
from pydantic import ValidationError

from eval.gold_gate import IsolationSummary, audit_pool_isolation


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def row(name: str, pool: str = "dev", **changes: Any) -> dict[str, Any]:
    return {
        "item_sha256": digest(name),
        "cohort": "text",
        "pool": pool,
        "exposure": "current",
        "original_family_sha256": [digest("family-" + name)],
        "family_alias_sha256": [],
        "media_sha256": [],
        "component_item_sha256": [],
        **changes,
    }


def audit(*rows: dict[str, Any]) -> IsolationSummary:
    return audit_pool_isolation(
        {"schema_version": "gold-isolation-metadata-v1", "rows": list(rows)}
    )


@pytest.mark.parametrize(
    "left,right", list(combinations(("train", "dev", "calibration", "holdout"), 2))
)
@pytest.mark.parametrize("link", ["family", "media", "transitive"])
def test_every_pool_pair_across_cohorts(left: str, right: str, link: str) -> None:
    first = row("a", left, cohort="image", media_sha256=[digest("image")])
    second = row("b", right, cohort="video", media_sha256=[digest("video")])
    rows = [first, second]
    if link == "family":
        second["original_family_sha256"] = first["original_family_sha256"]
    elif link == "media":
        second["media_sha256"] = first["media_sha256"]
    else:
        rows.append(
            row(
                "bridge",
                left,
                family_alias_sha256=first["original_family_sha256"],
                media_sha256=second["media_sha256"],
                exposure="historical",
            )
        )
    result = audit(*rows)
    assert result.outcome == "rejected"
    assert result.categories == ("pool_overlap",)
    assert result.overlapping_components == 1
    assert (
        next(
            pair.components
            for pair in result.pool_pairs
            if pair.left == left and pair.right == right
        )
        == 1
    )


def test_global_component_reports_all_six_pairs_and_order_invariance() -> None:
    rows = [
        row(str(i), pool, family_alias_sha256=[digest("common")])
        for i, pool in enumerate(("train", "dev", "calibration", "holdout"))
    ]
    result = audit(*rows)
    assert result == audit(*reversed(rows))
    assert result.connected_components == 1
    assert [pair.components for pair in result.pool_pairs] == [1] * 6


def test_isolated_metadata_does_not_claim_stage_or_inventory_acceptance() -> None:
    result = audit(*(row(pool, pool) for pool in ("train", "dev", "calibration", "holdout")))
    assert result.outcome == "isolated"
    assert result.scope == "supplied_metadata_isolation_only"
    assert result.connected_components == 4
    assert result.unique_items == 4
    assert all(pair.components == 0 for pair in result.pool_pairs)


@pytest.mark.parametrize("old_pool", ["train", "dev"])
def test_past_exposure_of_same_item_cannot_be_erased(old_pool: str) -> None:
    result = audit(row("a", "holdout"), row("a", old_pool, exposure="historical"))
    assert result.categories == ("pool_overlap",)
    assert result.historical_rows == 1
    assert result.unique_items == 1


def test_removed_historical_items_still_join_current_aliases() -> None:
    result = audit(
        row("retired", "train", exposure="historical", media_sha256=[digest("shared")]),
        row("new", "holdout", media_sha256=[digest("shared")]),
    )
    assert result.categories == ("pool_overlap",)
    assert result.unique_items == 2


def test_identical_historical_exposures_do_not_inflate_unique_item_count() -> None:
    historical = row("a", exposure="historical")
    result = audit(row("a"), historical, historical)
    assert result.outcome == "isolated"
    assert result.unique_items == 1
    assert result.rows == 3
    assert result.historical_rows == 2


def components() -> list[dict[str, Any]]:
    image = row("image", cohort="image", media_sha256=[digest("image-media")])
    audio = row("audio", cohort="audio", media_sha256=[digest("audio-media")])
    composite = row(
        "composite",
        cohort="crossmodal",
        media_sha256=image["media_sha256"] + audio["media_sha256"],
        component_item_sha256=[image["item_sha256"], audio["item_sha256"]],
    )
    return [image, audio, composite]


def test_crossmodal_links_originals_and_their_history() -> None:
    rows = components()
    assert audit(*rows).connected_components == 1
    assert audit(*rows).outcome == "isolated"
    historical_image = {**rows[0], "pool": "train", "exposure": "historical"}
    assert audit(*rows, historical_image).categories == ("pool_overlap",)


@pytest.mark.parametrize("mutation", ["absent", "wrong_cohort", "missing_media", "self"])
def test_crossmodal_requires_resolvable_original_components(mutation: str) -> None:
    rows = components()
    if mutation == "absent":
        rows.pop(0)
    elif mutation == "wrong_cohort":
        rows[0]["cohort"] = "video"
    elif mutation == "missing_media":
        rows[2]["media_sha256"] = [digest("unrelated")]
    else:
        rows[2]["component_item_sha256"][0] = rows[2]["item_sha256"]
    assert audit(*rows).categories == ("missing_component",)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"cohort": "audio", "media_sha256": [digest("m")]},
        {"original_family_sha256": [digest("other")]},
        {"family_alias_sha256": [digest("other")]},
    ],
)
def test_duplicate_or_conflicting_identity_rejected(change: dict[str, Any]) -> None:
    assert audit(row("a"), row("a", **change)).categories == ("conflicting_identity",)


@pytest.mark.parametrize("field", ["answer", "payload", "source_revision", "family_id", "isolated"])
def test_unrecognized_fields_cannot_bypass_or_escape(field: str, capsys: Any) -> None:
    secret = "SECRET-DO-NOT-EXPORT"
    result = audit(row("a", **{field: secret}))
    assert result.categories == ("invalid_metadata",)
    assert secret not in result.model_dump_json()
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("field", list(row("a")))
def test_missing_required_fields_fail_closed(field: str) -> None:
    metadata = row("a")
    del metadata[field]
    assert audit(metadata).categories == ("invalid_metadata",)


@pytest.mark.parametrize(
    "change",
    [
        {"original_family_sha256": []},
        {"item_sha256": "SECRET-BAD-HASH"},
        {"media_sha256": [1]},
        {"cohort": "image"},
        {"cohort": "crossmodal"},
        {"component_item_sha256": [digest("absent")]},
        {"pool": "SECRET-BAD-POOL"},
    ],
)
def test_invalid_values_fail_closed(change: dict[str, Any]) -> None:
    assert audit(row("a", **change)).categories == ("invalid_metadata",)


@pytest.mark.parametrize("metadata", [None, [], {}, {"schema_version": "wrong", "rows": []}])
def test_invalid_envelope(metadata: object) -> None:
    assert audit_pool_isolation(metadata).categories == ("invalid_metadata",)


def test_summary_contains_no_identifiers_and_is_strict_and_immutable() -> None:
    rows = components()
    result = audit(*rows)
    encoded = result.model_dump_json()
    for item in rows:
        assert item["item_sha256"] not in encoded
        assert all(value not in encoded for value in item["media_sha256"])
        assert all(value not in encoded for value in item["original_family_sha256"])
    with pytest.raises(ValidationError):
        IsolationSummary.model_validate_json(
            json.dumps({**json.loads(encoded), "payload": "SECRET"})
        )
    assert IsolationSummary.model_validate_json(encoded) == result
    with pytest.raises(ValidationError):
        result.outcome = "rejected"
    with pytest.raises(ValidationError):
        result.pool_pairs[0].components = 999
