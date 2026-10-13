"""Redacted sealed-set annotation summaries.

The gate receives annotation metadata and labels inside an isolated boundary, then
emits only aggregate evidence. It rejects records that carry sample payload fields
so the development lane can test the contract with canaries without seeing
sealed content.
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
    annotator_id: Nonempty
    label: Nonempty
    family_hash: Sha256
    source_hash: Sha256


class SealedManifest(StrictModel):
    audit_id: Nonempty
    dataset_version: Nonempty
    source_digest: Sha256
    canary: bool
    expected_modalities: list[Modality] = Field(min_length=1)
    expected_items_per_modality: dict[Modality, int]


class ModalitySummary(StrictModel):
    items: int
    annotations: int
    annotators: int
    kappa: float
    missing_items: int
    invalid_items: int


class SealedSummary(StrictModel):
    audit_id: str
    dataset_version: str
    source_digest: str
    canary: bool
    passed: bool
    modalities: dict[Modality, ModalitySummary]
    total_items: int
    total_annotations: int
    redaction: Literal["metadata_only_no_payload_or_labels"]
    failures: list[str]


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


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("kappa requires paired non-empty labels")
    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / total
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    labels = set(counts_a) | set(counts_b)
    expected = math.fsum(
        (counts_a[label] / total) * (counts_b[label] / total) for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def summarize(manifest: SealedManifest, annotations: list[AnnotationRecord]) -> SealedSummary:
    failures: list[str] = []
    grouped: dict[Modality, dict[str, list[AnnotationRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in annotations:
        if record.modality not in manifest.expected_modalities:
            failures.append(f"unexpected modality: {record.modality}")
            continue
        grouped[record.modality][record.item_id].append(record)

    summaries: dict[Modality, ModalitySummary] = {}
    for modality in manifest.expected_modalities:
        expected_items = manifest.expected_items_per_modality.get(modality)
        if expected_items is None:
            failures.append(f"missing expected count for modality: {modality}")
            expected_items = 0
        labels_a: list[str] = []
        labels_b: list[str] = []
        invalid = 0
        annotators: set[str] = set()
        for item_id, item_records in sorted(grouped[modality].items()):
            annotators.update(record.annotator_id for record in item_records)
            if len(item_records) != 2:
                invalid += 1
                failures.append(f"{modality}:{item_id}: expected exactly two raw annotations")
                continue
            first, second = item_records
            if first.annotator_id == second.annotator_id:
                invalid += 1
                failures.append(f"{modality}:{item_id}: duplicate annotator")
                continue
            if first.family_hash != second.family_hash or first.source_hash != second.source_hash:
                invalid += 1
                failures.append(f"{modality}:{item_id}: hash mismatch between annotators")
                continue
            labels_a.append(first.label)
            labels_b.append(second.label)

        observed_items = len(grouped[modality])
        missing = max(expected_items - observed_items, 0)
        if missing:
            failures.append(f"{modality}: missing {missing} expected items")
        kappa = cohen_kappa(labels_a, labels_b) if labels_a else 0.0
        if not manifest.canary and kappa < 0.7:
            failures.append(f"{modality}: kappa below 0.7")
        summaries[modality] = ModalitySummary(
            items=observed_items,
            annotations=sum(len(records) for records in grouped[modality].values()),
            annotators=len(annotators),
            kappa=kappa,
            missing_items=missing,
            invalid_items=invalid,
        )

    return SealedSummary(
        audit_id=manifest.audit_id,
        dataset_version=manifest.dataset_version,
        source_digest=manifest.source_digest,
        canary=manifest.canary,
        passed=not failures,
        modalities=summaries,
        total_items=sum(summary.items for summary in summaries.values()),
        total_annotations=sum(summary.annotations for summary in summaries.values()),
        redaction="metadata_only_no_payload_or_labels",
        failures=failures,
    )
