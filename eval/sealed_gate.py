"""Legacy v2 double-annotation canary (superseded for first release by ADR0012).

The isolated gate reads raw double annotations for holdout cohorts and a
redacted development ledger for dev cohorts. Development code can exercise this
with canaries, but real holdout acceptance still requires an externally
controlled identity or host because this workspace has root access to the GPU
instance.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Cohort = Literal["text", "image", "screenshot", "audio", "video", "crossmodal"]
Pool = Literal["dev", "holdout"]
LabelOrigin = Literal["hard_gold", "program_truth", "human_verified_gold", "public_benchmark_gold"]

CONTRACT_VERSION: Literal["sealed-cohort-gate-v2"] = "sealed-cohort-gate-v2"
REAL_COHORTS: tuple[Cohort, ...] = (
    "text",
    "image",
    "screenshot",
    "audio",
    "video",
    "crossmodal",
)
PAYLOAD_KEYS = frozenset(
    {
        "answer",
        "answers",
        "content",
        "instruction",
        "label",
        "labels",
        "media",
        "media_uri",
        "messages",
        "options",
        "prompt",
        "question",
        "response",
        "state",
        "target",
        "text",
        "uri",
    }
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AnnotationRecord(StrictModel):
    item_id: Nonempty
    modality: Cohort
    pool: Literal["holdout", "dev"]
    annotator_id: Nonempty
    label: Nonempty
    family_hash: Sha256
    source_hash: Sha256


class DevelopmentRecord(StrictModel):
    item_id: Nonempty
    modality: Cohort
    family_hash: Sha256
    source_hash: Sha256
    source_dataset: Nonempty
    source_revision: Nonempty
    license: Nonempty
    label_origin: LabelOrigin

    @model_validator(mode="after")
    def validate_development_contract(self) -> DevelopmentRecord:
        if self.license.casefold() in {"unknown", "unknown-license", "none", "n/a"}:
            raise ValueError("development license must be known")
        return self


class SealedManifest(StrictModel):
    audit_id: Nonempty
    dataset_version: Nonempty
    source_digest: Sha256
    canary: bool
    annotator_ids: tuple[Nonempty, Nonempty]
    min_kappa: float = Field(default=0.7, ge=-1.0, le=1.0)
    expected_counts: dict[Cohort, dict[Pool, int]]

    @model_validator(mode="after")
    def validate_gate_contract(self) -> SealedManifest:
        if self.annotator_ids[0] == self.annotator_ids[1]:
            raise ValueError("annotator identities must be distinct")
        if not self.expected_counts:
            raise ValueError("expected counts must not be empty")
        for pool_counts in self.expected_counts.values():
            if not pool_counts:
                raise ValueError("each cohort must declare at least one pool")
            for expected_items in pool_counts.values():
                if expected_items <= 0:
                    raise ValueError("expected counts must be positive")
        if not self.canary:
            if self.min_kappa < 0.7:
                raise ValueError("real sealed gate min_kappa must be at least 0.7")
            if set(self.expected_counts) != set(REAL_COHORTS):
                raise ValueError("real sealed gate requires all six evaluation cohorts")
            for cohort in REAL_COHORTS:
                if self.expected_counts[cohort] != {"dev": 500, "holdout": 1000}:
                    raise ValueError("real sealed gate requires dev=500 and holdout=1000")
        return self


class CohortSummary(StrictModel):
    items: int
    annotations: int
    annotators: int
    kappa: float | None
    kappa_degenerate: bool
    missing_items: int
    invalid_items: int
    failure_counts: dict[str, int]
    source_origin_counts: dict[str, int]


class SealedSummary(StrictModel):
    contract_version: Literal["sealed-cohort-gate-v2"]
    audit_id: str
    dataset_version: str
    source_digest: str
    canary: bool
    passed: bool
    cohorts: dict[str, CohortSummary]
    total_items: int
    total_annotations: int
    redaction: Literal["aggregate_counts_no_payload_labels_or_item_ids"]
    failures: dict[str, int]
    blocked_reason: str | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: record must be an object")
            rows.append(value)
    return rows


def validate_redacted(raw: dict[str, Any], *, allow_label: bool) -> None:
    forbidden = PAYLOAD_KEYS if not allow_label else PAYLOAD_KEYS - {"label"}
    present = forbidden.intersection(raw)
    if present:
        raise ValueError(f"sealed record includes payload fields: {sorted(present)}")


def load_annotations(path: Path) -> list[AnnotationRecord]:
    records: list[AnnotationRecord] = []
    for raw in read_jsonl(path):
        validate_redacted(raw, allow_label=True)
        records.append(AnnotationRecord.model_validate(raw))
    return records


def load_development(path: Path) -> list[DevelopmentRecord]:
    records: list[DevelopmentRecord] = []
    for raw in read_jsonl(path):
        validate_redacted(raw, allow_label=False)
        records.append(DevelopmentRecord.model_validate(raw))
    return records


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> tuple[float, bool]:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("kappa requires paired non-empty labels")
    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    labels = set(counts_a) | set(counts_b)
    expected = math.fsum((counts_a[label] / total) * (counts_b[label] / total) for label in labels)
    degenerate = len(labels) < 2 or math.isclose(expected, 1.0)
    if math.isclose(expected, 1.0):
        return (1.0 if math.isclose(observed, 1.0) else 0.0), True
    return (observed - expected) / (1.0 - expected), degenerate


def _failure_totals(cohorts: dict[str, CohortSummary]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for summary in cohorts.values():
        totals.update(summary.failure_counts)
    return dict(sorted(totals.items()))


def _empty_summary(expected_items: int, failures: Counter[str]) -> CohortSummary:
    missing = max(expected_items, 0)
    if missing:
        failures["missing_items"] += missing
    return CohortSummary(
        items=0,
        annotations=0,
        annotators=0,
        kappa=None,
        kappa_degenerate=True,
        missing_items=missing,
        invalid_items=0,
        failure_counts=dict(sorted(failures.items())),
        source_origin_counts={},
    )


def _summarize_development(
    manifest: SealedManifest,
    development: list[DevelopmentRecord],
    failure_totals: Counter[str],
    family_pools: dict[str, set[Pool]],
    item_cohorts: dict[str, set[Cohort]],
) -> dict[str, CohortSummary]:
    grouped: dict[Cohort, dict[str, DevelopmentRecord]] = defaultdict(dict)
    duplicates: Counter[Cohort] = Counter()
    for record in development:
        if record.modality not in manifest.expected_counts:
            failure_totals["unexpected_modality"] += 1
            continue
        if "dev" not in manifest.expected_counts[record.modality]:
            failure_totals["unexpected_pool"] += 1
            continue
        if record.item_id in grouped[record.modality]:
            duplicates[record.modality] += 1
            continue
        grouped[record.modality][record.item_id] = record
        family_pools[record.family_hash].add("dev")
        item_cohorts[record.item_id].add(record.modality)

    summaries: dict[str, CohortSummary] = {}
    for cohort, pool_counts in manifest.expected_counts.items():
        if "dev" not in pool_counts:
            continue
        expected_items = pool_counts["dev"]
        item_groups = grouped[cohort]
        failures: Counter[str] = Counter()
        if duplicates[cohort]:
            failures["duplicate_development_item"] += duplicates[cohort]
        observed_items = len(item_groups)
        missing = max(expected_items - observed_items, 0)
        extra = max(observed_items - expected_items, 0)
        if missing:
            failures["missing_items"] += missing
        if extra:
            failures["extra_items"] += extra
        source_origin_counts = Counter(record.label_origin for record in item_groups.values())
        summaries[f"{cohort}:dev"] = CohortSummary(
            items=observed_items,
            annotations=0,
            annotators=0,
            kappa=None,
            kappa_degenerate=False,
            missing_items=missing,
            invalid_items=duplicates[cohort],
            failure_counts=dict(sorted(failures.items())),
            source_origin_counts=dict(sorted(source_origin_counts.items())),
        )
    return summaries


def _summarize_annotation_pool(
    manifest: SealedManifest,
    annotations: list[AnnotationRecord],
    pools: set[Pool],
    failure_totals: Counter[str],
    family_pools: dict[str, set[Pool]],
    item_cohorts: dict[str, set[Cohort]],
) -> dict[str, CohortSummary]:
    grouped: dict[tuple[Cohort, Pool], dict[str, list[AnnotationRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    item_sources: dict[str, set[str]] = defaultdict(set)
    for record in annotations:
        if record.modality not in manifest.expected_counts:
            failure_totals["unexpected_modality"] += 1
            continue
        if record.pool not in manifest.expected_counts[record.modality] or record.pool not in pools:
            failure_totals["unexpected_pool"] += 1
            continue
        grouped[(record.modality, record.pool)][record.item_id].append(record)
        family_pools[record.family_hash].add(record.pool)
        item_cohorts[record.item_id].add(record.modality)
        item_sources[record.item_id].add(record.source_hash)

    cross_item_source_mismatch = sum(1 for sources in item_sources.values() if len(sources) > 1)
    if cross_item_source_mismatch:
        failure_totals["source_hash_mismatch_across_annotations"] += cross_item_source_mismatch

    cohorts: dict[str, CohortSummary] = {}
    for cohort, pool_counts in manifest.expected_counts.items():
        for pool, expected_items in pool_counts.items():
            if pool not in pools:
                continue
            item_groups = grouped[(cohort, pool)]
            labels_a: list[str] = []
            labels_b: list[str] = []
            annotators: set[str] = set()
            failures: Counter[str] = Counter()

            for item_records in item_groups.values():
                annotators.update(record.annotator_id for record in item_records)
                if len(item_records) != 2:
                    failures["wrong_annotation_count"] += 1
                    continue
                by_annotator = {record.annotator_id: record for record in item_records}
                if len(by_annotator) != 2:
                    failures["duplicate_annotator"] += 1
                    continue
                if set(by_annotator) != set(manifest.annotator_ids):
                    failures["unexpected_annotator"] += 1
                    continue
                first = by_annotator[manifest.annotator_ids[0]]
                second = by_annotator[manifest.annotator_ids[1]]
                if (
                    first.family_hash != second.family_hash
                    or first.source_hash != second.source_hash
                ):
                    failures["annotation_hash_mismatch"] += 1
                    continue
                labels_a.append(first.label)
                labels_b.append(second.label)

            observed_items = len(item_groups)
            missing = max(expected_items - observed_items, 0)
            extra = max(observed_items - expected_items, 0)
            if missing:
                failures["missing_items"] += missing
            if extra:
                failures["extra_items"] += extra

            if labels_a:
                kappa, degenerate = cohen_kappa(labels_a, labels_b)
            else:
                kappa, degenerate = 0.0, True
                failures["no_valid_pairs"] += 1
            if not manifest.canary and kappa < manifest.min_kappa:
                failures["kappa_below_threshold"] += 1
            if not manifest.canary and degenerate:
                failures["kappa_degenerate"] += 1

            cohort_key = f"{cohort}:{pool}"
            cohorts[cohort_key] = CohortSummary(
                items=observed_items,
                annotations=sum(len(records) for records in item_groups.values()),
                annotators=len(annotators),
                kappa=kappa,
                kappa_degenerate=degenerate,
                missing_items=missing,
                invalid_items=sum(
                    count
                    for key, count in failures.items()
                    if key
                    in {
                        "wrong_annotation_count",
                        "duplicate_annotator",
                        "annotation_hash_mismatch",
                    }
                ),
                failure_counts=dict(sorted(failures.items())),
                source_origin_counts={},
            )
    return cohorts


def summarize(
    manifest: SealedManifest,
    annotations: list[AnnotationRecord],
    development: list[DevelopmentRecord] | None = None,
) -> SealedSummary:
    failure_totals: Counter[str] = Counter()
    family_pools: dict[str, set[Pool]] = defaultdict(set)
    item_cohorts: dict[str, set[Cohort]] = defaultdict(set)
    cohorts: dict[str, CohortSummary] = {}
    item_pools: dict[str, set[Pool]] = defaultdict(set)
    item_identities: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for annotation in annotations:
        item_pools[annotation.item_id].add(annotation.pool)
        item_identities[annotation.item_id].add((annotation.family_hash, annotation.source_hash))
    for entry in development or []:
        item_pools[entry.item_id].add("dev")
        item_identities[entry.item_id].add((entry.family_hash, entry.source_hash))
    cross_pool_items = sum(len(pools) > 1 for pools in item_pools.values())
    identity_mismatches = sum(len(identities) > 1 for identities in item_identities.values())
    if cross_pool_items:
        failure_totals["duplicate_item_across_pools"] = cross_pool_items
    if identity_mismatches:
        failure_totals["item_identity_mismatch"] = identity_mismatches

    if development is None:
        if not manifest.canary:
            failure_totals["missing_development_ledger"] += 1
        cohorts.update(
            _summarize_annotation_pool(
                manifest,
                annotations,
                {"dev", "holdout"},
                failure_totals,
                family_pools,
                item_cohorts,
            )
        )
    else:
        cohorts.update(
            _summarize_development(
                manifest, development, failure_totals, family_pools, item_cohorts
            )
        )
        cohorts.update(
            _summarize_annotation_pool(
                manifest,
                annotations,
                {"holdout"},
                failure_totals,
                family_pools,
                item_cohorts,
            )
        )

    cross_pool_family_count = sum(1 for pools in family_pools.values() if len(pools) > 1)
    duplicate_item_across_cohorts = sum(
        1 for cohort_set in item_cohorts.values() if len(cohort_set) > 1
    )
    if cross_pool_family_count:
        failure_totals["cross_pool_family_overlap"] = cross_pool_family_count
    if duplicate_item_across_cohorts:
        failure_totals["duplicate_item_across_cohorts"] = duplicate_item_across_cohorts
    failure_totals.update(_failure_totals(cohorts))

    blocked_reason = None
    if not manifest.canary:
        blocked_reason = (
            "real acceptance requires an externally controlled gate identity/host; "
            "root-access development workspaces cannot prove sealed-content isolation"
        )

    return SealedSummary(
        contract_version=CONTRACT_VERSION,
        audit_id=manifest.audit_id,
        dataset_version=manifest.dataset_version,
        source_digest=manifest.source_digest,
        canary=manifest.canary,
        passed=not failure_totals and manifest.canary,
        cohorts=cohorts,
        total_items=sum(summary.items for summary in cohorts.values()),
        total_annotations=sum(summary.annotations for summary in cohorts.values()),
        redaction="aggregate_counts_no_payload_labels_or_item_ids",
        failures=dict(sorted(failure_totals.items())),
        blocked_reason=blocked_reason,
    )
