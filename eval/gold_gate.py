"""Metadata-only isolation audit, not an existing-gold or stage acceptance gate.

Only the supplied current and historical rows are audited. This cannot attest
gold labels, inventory/lineage completeness, source authenticity, quotas, private
job permissions, or stage quality. Extraction and verification belong to a future
private job; never pass sample payloads or answers to this API.

Family hashes are global canonical ORIGINAL identities, independent of source
revision. Explicit aliases must include parent families for derived media; exact
media hashes alone cannot detect crops/re-encoding. Raw family names are not
accepted, so raw/hash reconciliation must happen before this boundary. Historical
rows preserve past pool membership even when an item is no longer selected.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal, Self

from pydantic import ConfigDict, Field, ValidationError, model_validator

from data.schema import Contract, Pool, Sha256

Cohort = Literal["text", "image", "screenshot", "audio", "video", "crossmodal"]
Category = Literal["invalid_metadata", "conflicting_identity", "missing_component", "pool_overlap"]


class IsolationRow(Contract):
    """A frozen question identity; component references resolve to supplied rows.

    All hash lists are required, including explicitly empty lists. Crossmodal
    rows must reference original image and audio items, not other crossmodal
    items. Each component's media must be included in the composite media list.
    Repeated historical exposure is allowed; one current row per identity is
    allowed. Pool and exposure may differ across otherwise identical identities.
    """

    model_config = ConfigDict(revalidate_instances="always")

    item_sha256: Sha256
    cohort: Cohort
    pool: Pool
    exposure: Literal["current", "historical"]
    original_family_sha256: list[Sha256] = Field(min_length=1)
    family_alias_sha256: list[Sha256]
    media_sha256: list[Sha256]
    component_item_sha256: list[Sha256]

    @model_validator(mode="after")
    def required_metadata(self) -> Self:
        if self.cohort != "text" and not self.media_sha256:
            raise ValueError("media metadata required")
        if self.cohort == "crossmodal" and len(set(self.component_item_sha256)) < 2:
            raise ValueError("component metadata required")
        if self.cohort != "crossmodal" and self.component_item_sha256:
            raise ValueError("only crossmodal rows have component references")
        return self


class IsolationInventory(Contract):
    model_config = ConfigDict(revalidate_instances="always")

    schema_version: Literal["gold-isolation-metadata-v1"]
    rows: list[IsolationRow] = Field(min_length=1)


class PoolPairCount(Contract):
    model_config = ConfigDict(frozen=True)
    left: Pool
    right: Pool
    components: int = Field(ge=0)


class IsolationSummary(Contract):
    """Allowlisted aggregate result; never a whole-stage acceptance receipt."""

    model_config = ConfigDict(frozen=True)
    schema_version: Literal["gold-isolation-summary-v1"] = "gold-isolation-summary-v1"
    scope: Literal["supplied_metadata_isolation_only"] = "supplied_metadata_isolation_only"
    outcome: Literal["isolated", "rejected"]
    categories: tuple[Category, ...] = ()
    rows: int = Field(default=0, ge=0)
    historical_rows: int = Field(default=0, ge=0)
    unique_items: int = Field(default=0, ge=0)
    connected_components: int = Field(default=0, ge=0)
    overlapping_components: int = Field(default=0, ge=0)
    pool_pairs: tuple[PoolPairCount, ...] = ()


def audit_pool_isolation(metadata: object) -> IsolationSummary:
    """Validate a raw metadata object and return counts/fixed categories only.

    Supply a JSON-decoded dictionary. Validation exceptions (which can include
    secret inputs) never escape this boundary or get copied into the result.
    This function performs no I/O, logs nothing, and accepts no model callbacks.
    """
    try:
        inventory = IsolationInventory.model_validate(metadata)
    except (ValidationError, ValueError, TypeError):
        return IsolationSummary(outcome="rejected", categories=("invalid_metadata",))

    rows = inventory.rows
    identities: dict[str, IsolationRow] = {}
    current: set[str] = set()
    for row in rows:
        prior = identities.get(row.item_sha256)
        if prior is not None and (
            prior.cohort != row.cohort
            or set(prior.original_family_sha256) != set(row.original_family_sha256)
            or set(prior.family_alias_sha256) != set(row.family_alias_sha256)
            or set(prior.media_sha256) != set(row.media_sha256)
            or set(prior.component_item_sha256) != set(row.component_item_sha256)
        ):
            return IsolationSummary(outcome="rejected", categories=("conflicting_identity",))
        if row.exposure == "current":
            if row.item_sha256 in current:
                return IsolationSummary(outcome="rejected", categories=("conflicting_identity",))
            current.add(row.item_sha256)
        identities[row.item_sha256] = row

    # Same path-compressed union-find as development exclusions, now global
    # across all cohorts and pools; no dataset/revision namespace can split it.
    parents: dict[str, str] = {}

    def find(key: str) -> str:
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(left: str, right: str) -> None:
        a, b = sorted((find(left), find(right)))
        parents[b] = a

    for row in rows:
        item = "item:" + row.item_sha256
        find(item)
        for family in row.original_family_sha256 + row.family_alias_sha256:
            join(item, "family:" + family)
        for media in row.media_sha256:
            join(item, "media:" + media)
        components = [identities.get(key) for key in row.component_item_sha256]
        if row.cohort == "crossmodal":
            if (
                any(component is None for component in components)
                or {component.cohort for component in components if component is not None}
                != {"image", "audio"}
                or any(
                    not set(component.media_sha256) <= set(row.media_sha256)
                    for component in components
                    if component is not None
                )
            ):
                return IsolationSummary(outcome="rejected", categories=("missing_component",))
            for key in row.component_item_sha256:
                join(item, "item:" + key)

    pools: dict[str, set[Pool]] = {}
    for row in rows:
        pools.setdefault(find("item:" + row.item_sha256), set()).add(row.pool)
    overlap = sum(len(values) > 1 for values in pools.values())
    pool_order: tuple[Pool, ...] = ("train", "dev", "calibration", "holdout")
    return IsolationSummary(
        outcome="rejected" if overlap else "isolated",
        categories=("pool_overlap",) if overlap else (),
        rows=len(rows),
        historical_rows=sum(row.exposure == "historical" for row in rows),
        unique_items=len(identities),
        connected_components=len(pools),
        overlapping_components=overlap,
        pool_pairs=tuple(
            PoolPairCount(
                left=left,
                right=right,
                components=sum({left, right} <= values for values in pools.values()),
            )
            for left, right in combinations(pool_order, 2)
        ),
    )
