"""Redacted sealed-set annotation summaries.

The isolated gate reads raw double annotations and emits only aggregate counts.
Development code can exercise this with canaries, but real holdout acceptance
still requires an externally controlled identity or host because this workspace
has root access to the GPU instance.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Modality = Literal["text", "image", "audio", "video"]
Pool = Literal["dev", "holdout"]

PAYLOAD_KEYS = frozenset(
    {
        "answer",
        "answers",
        "content",
        "instruction",
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
    modality: Modality
    pool: Pool
    annotator_id: Nonempty
    label: Nonempty
    family_hash: Sha256
    source_hash: Sha256


class SealedManifest(StrictModel):
    audit_id: Nonempty
    dataset_version: Nonempty
    source_digest: Sha256
    canary: bool
    min_kappa: float = Field(default=0.7, ge=-1.0, le=1.0)
    expected_counts: dict[Modality, dict[Pool, int]]


class CohortSummary(StrictModel):
    items: int
    annotations: int
    annotators: int
    kappa: float
    kappa_degenerate: bool
    missing_items: int
    invalid_items: int
    failure_counts: dict[str, int]


class SealedSummary(StrictModel):
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


def validate_redacted(raw: dict[str, Any]) -> None:
    present = PAYLOAD_KEYS.intersection(raw)
    if present:
        raise ValueError(f"annotation record includes sealed payload fields: {sorted(present)}")


def load_annotations(path: Path) -> list[AnnotationRecord]:
    records: list[AnnotationRecord] = []
    for raw in read_jsonl(path):
        validate_redacted(raw)
        records.append(AnnotationRecord.model_validate(raw))
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


def summarize(manifest: SealedManifest, annotations: list[AnnotationRecord]) -> SealedSummary:
    grouped: dict[tuple[Modality, Pool], dict[str, list[AnnotationRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    family_pools: dict[str, set[Pool]] = defaultdict(set)
    for record in annotations:
        grouped[(record.modality, record.pool)][record.item_id].append(record)
        family_pools[record.family_hash].add(record.pool)

    cross_pool_family_count = sum(1 for pools in family_pools.values() if len(pools) > 1)
    cohorts: dict[str, CohortSummary] = {}
    for modality, pool_counts in manifest.expected_counts.items():
        for pool, expected_items in pool_counts.items():
            item_groups = grouped[(modality, pool)]
            labels_a: list[str] = []
            labels_b: list[str] = []
            annotators: set[str] = set()
            failures: Counter[str] = Counter()

            for item_records in item_groups.values():
                annotators.update(record.annotator_id for record in item_records)
                if len(item_records) != 2:
                    failures["wrong_annotation_count"] += 1
                    continue
                first, second = item_records
                if first.annotator_id == second.annotator_id:
                    failures["duplicate_annotator"] += 1
                    continue
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
            if missing:
                failures["missing_items"] += missing
            extra = max(observed_items - expected_items, 0)
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

            cohort_key = f"{modality}:{pool}"
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
            )

    failure_totals = _failure_totals(cohorts)
    if cross_pool_family_count:
        failure_totals["cross_pool_family_overlap"] = cross_pool_family_count
    blocked_reason = None
    if not manifest.canary:
        blocked_reason = (
            "real acceptance requires an externally controlled gate identity/host; "
            "root-access development workspaces cannot prove sealed-content isolation"
        )

    return SealedSummary(
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
