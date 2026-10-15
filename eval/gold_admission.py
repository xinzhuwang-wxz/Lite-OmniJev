"""Assemble metadata bindings only; never authenticate evidence or pass a gold gate.

Callers must independently hash-check every referenced artifact and verify source
adapter execution. This pure API has no filesystem, trust store or private data.
Even a structurally passed receipt leaves authenticity and runtime verification
pending. Counts are declared audit outputs, not an IID/effective-N assertion.
"""

from typing import Literal

from pydantic import Field, ValidationError

from data.schema import Contract, Sha256
from eval.gold_gate import Cohort
from scripts.private_gold_job import PublicReceipt

COHORTS = {"text", "image", "screenshot", "audio", "video", "crossmodal"}


class PoolCounts(Contract):
    questions: int = Field(ge=0)
    unique_questions: int = Field(ge=0)
    unique_original_questions: int = Field(ge=0)
    original_families: int = Field(ge=0)
    media: int = Field(ge=0)
    dependency_components: int = Field(ge=0)


class Selection(Contract):
    cohort: Cohort
    pool: Literal["dev", "calibration", "holdout"]
    artifact_sha256: Sha256
    source_evidence_sha256: Sha256
    counts: PoolCounts
    frozen_original_dev_questions: int = Field(ge=0)
    invalid_labels: int = Field(ge=0)
    missing_labels: int = Field(ge=0)
    teacher_labels: int = Field(ge=0)
    soft_labels: int = Field(ge=0)
    canary_questions: int = Field(ge=0)
    nonreal_questions: int = Field(ge=0)


class SourceEvidence(Contract):
    """A claimed adapter output binding, requiring external execution verification."""

    adapter: Literal[
        "text-source-replay-v1",
        "gui-source-replay-v1",
        "vqa-source-replay-v1",
        "speech-source-replay-v1",
        "clotho-source-replay-v1",
        "clevrer-source-replay-v1",
        "crossmodal-rule-replay-v1",
    ]
    report_sha256: Sha256
    code_sha256: Sha256
    source_manifest_sha256: Sha256
    source_artifact_sha256: list[Sha256] = Field(min_length=1)
    output_artifact_sha256: list[Sha256] = Field(min_length=1)
    status: Literal["replayed", "rejected"]


class RuntimeEvidence(Contract):
    report_sha256: Sha256
    launcher_code_sha256: Sha256
    status: Literal["passed", "rejected"]
    failure_injection: Literal["passed", "rejected"]
    log_redaction: Literal["passed", "rejected"]
    scope: Literal["ordinary_identity_root_visible"]


class AdmissionInput(Contract):
    schema_version: Literal["gold-admission-input-v1"]
    inventory_sha256: Sha256
    inventory_artifact_sha256: list[Sha256] = Field(min_length=1)
    selection_audit_sha256: Sha256
    selection_audit_code_sha256: Sha256
    selection_audit_inventory_sha256: Sha256
    isolation_report_sha256: Sha256
    isolation: PublicReceipt
    runtime: RuntimeEvidence
    sources: list[SourceEvidence] = Field(min_length=1)
    selected: list[Selection] = Field(min_length=1)


Category = Literal[
    "invalid_metadata",
    "inventory_binding",
    "isolation",
    "runtime_binding",
    "source_binding",
    "selection",
    "quota",
    "labels",
]


class AdmissionSummary(Contract):
    schema_version: Literal["gold-structural-admission-v1"] = "gold-structural-admission-v1"
    structural_admission: Literal["passed", "rejected"]
    categories: list[
        Literal[
            "invalid_metadata",
            "inventory_binding",
            "isolation",
            "runtime_binding",
            "source_binding",
            "selection",
            "quota",
            "labels",
        ]
    ]
    source_authenticity: Literal["requires_artifact_verification"] = (
        "requires_artifact_verification"
    )
    runtime_permissions: Literal["requires_artifact_verification"] = (
        "requires_artifact_verification"
    )
    final_gold_gate: Literal["not_evaluated"] = "not_evaluated"
    iid_claim: Literal[False] = False


def assemble_admission(metadata: object) -> AdmissionSummary:
    """Return fixed aggregate categories, never validation inputs or private IDs."""
    try:
        data = AdmissionInput.model_validate(metadata)
    except (ValidationError, ValueError, TypeError):
        return AdmissionSummary(structural_admission="rejected", categories=["invalid_metadata"])
    errors: list[Category] = []
    inventory = data.inventory_artifact_sha256
    if (
        len(set(inventory)) != len(inventory)
        or data.isolation.source_manifest_sha256 != data.inventory_sha256
        or data.selection_audit_inventory_sha256 != data.inventory_sha256
    ):
        errors.append("inventory_binding")
    isolation = data.isolation
    if (
        isolation.outcome != "isolated"
        or isolation.categories
        or isolation.exit_code != 0
        or not isolation.ordinary_identity_boundary
        or isolation.isolation is None
        or isolation.isolation.outcome != "isolated"
        or isolation.isolation.categories
        or isolation.isolation.overlapping_components
        or any(pair.components for pair in isolation.isolation.pool_pairs)
    ):
        errors.append("isolation")
    runtime = data.runtime
    if (
        runtime.launcher_code_sha256 != isolation.code_sha256
        or runtime.status != "passed"
        or runtime.failure_injection != "passed"
        or runtime.log_redaction != "passed"
    ):
        errors.append("runtime_binding")
    sources = {source.report_sha256: source for source in data.sources}
    if len(sources) != len(data.sources):
        errors.append("source_binding")
    for source in data.sources:
        if (
            source.status != "replayed"
            or source.source_manifest_sha256 != data.inventory_sha256
            or not set(source.source_artifact_sha256) <= set(inventory)
            or len(set(source.output_artifact_sha256)) != len(source.output_artifact_sha256)
        ):
            errors.append("source_binding")
    seen: set[str] = set()
    groups: dict[tuple[str, str], list[Selection]] = {}
    for selected in data.selected:
        selected_source = sources.get(selected.source_evidence_sha256)
        if (
            selected_source is None
            or selected.artifact_sha256 not in selected_source.output_artifact_sha256
            or selected.artifact_sha256 not in inventory
        ):
            errors.append("source_binding")
        if selected.artifact_sha256 in seen:
            errors.append("selection")
        seen.add(selected.artifact_sha256)
        groups.setdefault((selected.cohort, selected.pool), []).append(selected)
        counts = selected.counts
        if (
            counts.questions != counts.unique_questions
            or counts.questions != counts.unique_original_questions
            or not 0 < counts.original_families <= counts.questions
            or not 0 < counts.dependency_components <= counts.questions
            or (selected.cohort != "text" and counts.media == 0)
            or selected.frozen_original_dev_questions > counts.questions
            or (selected.pool != "dev" and selected.frozen_original_dev_questions)
        ):
            errors.append("selection")
        if any(
            (
                selected.invalid_labels,
                selected.missing_labels,
                selected.teacher_labels,
                selected.soft_labels,
                selected.canary_questions,
                selected.nonreal_questions,
            )
        ):
            errors.append("labels")
    for cohort in COHORTS:
        for pool in ("dev", "calibration", "holdout"):
            rows = groups.get((cohort, pool), [])
            count = sum(row.counts.unique_original_questions for row in rows)
            frozen = sum(row.frozen_original_dev_questions for row in rows)
            if (
                (pool == "holdout" and count != 1000)
                or (pool == "calibration" and not 3000 <= count <= 5000)
                or (pool == "dev" and (count < 500 or frozen < 500))
            ):
                errors.append("quota")
    return AdmissionSummary(
        structural_admission="rejected" if errors else "passed",
        categories=list(dict.fromkeys(errors)),
    )
