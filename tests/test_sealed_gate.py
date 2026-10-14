import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from eval.sealed_gate import (
    CONTRACT_VERSION,
    REAL_COHORTS,
    AnnotationRecord,
    DevelopmentRecord,
    SealedManifest,
    cohen_kappa,
    load_annotations,
    load_development,
    summarize,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64


def real_manifest(**updates: object) -> SealedManifest:
    values = {
        "audit_id": "real-audit",
        "dataset_version": "sealed-real-v2",
        "source_digest": HEX_A,
        "canary": False,
        "annotator_ids": ("ann-a", "ann-b"),
        "expected_counts": {cohort: {"dev": 500, "holdout": 1000} for cohort in REAL_COHORTS},
    }
    values.update(updates)
    return SealedManifest.model_validate(values)


def manifest(*, canary: bool = True) -> SealedManifest:
    return SealedManifest(
        audit_id="canary-audit",
        dataset_version="sealed-canary-v1",
        source_digest=HEX_A,
        canary=canary,
        annotator_ids=("ann-a", "ann-b"),
        expected_counts={"text": {"dev": 2, "holdout": 1}},
    )


def record(
    item_id: str,
    annotator_id: str,
    label: str,
    *,
    pool: Literal["dev", "holdout"] = "dev",
    modality: Literal["text", "image", "screenshot", "audio", "video", "crossmodal"] = "text",
    family_hash: str | None = None,
    source_hash: str = HEX_B,
) -> AnnotationRecord:
    return AnnotationRecord(
        item_id=item_id,
        modality=modality,
        pool=pool,
        annotator_id=annotator_id,
        label=label,
        family_hash=family_hash or (HEX_A if pool == "dev" else HEX_C),
        source_hash=source_hash,
    )


def dev_record(
    item_id: str,
    *,
    modality: Literal["text", "image", "screenshot", "audio", "video", "crossmodal"] = "text",
    family_hash: str = HEX_A,
    source_hash: str = HEX_B,
    license: str = "cc-by-4.0",
) -> DevelopmentRecord:
    return DevelopmentRecord(
        item_id=item_id,
        modality=modality,
        family_hash=family_hash,
        source_hash=source_hash,
        source_dataset="public-fixture",
        source_revision="rev-1",
        license=license,
        label_origin="program_truth",
    )


def complete_canary_records() -> list[AnnotationRecord]:
    return [
        record("item-1", "ann-a", "allow"),
        record("item-1", "ann-b", "allow"),
        record("item-2", "ann-a", "deny"),
        record("item-2", "ann-b", "allow"),
        record("item-3", "ann-a", "review", pool="holdout"),
        record("item-3", "ann-b", "review", pool="holdout"),
    ]


def real_development_records() -> list[DevelopmentRecord]:
    rows: list[DevelopmentRecord] = []
    for cohort_index, cohort in enumerate(REAL_COHORTS):
        family_prefix = f"{cohort_index:064x}"[-63:]
        source_prefix = f"{cohort_index + 10:064x}"[-63:]
        for index in range(500):
            rows.append(
                dev_record(
                    f"dev-{cohort}-{index}",
                    modality=cohort,
                    family_hash=f"a{family_prefix[:-3]}{index % 4096:03x}",
                    source_hash=f"b{source_prefix[:-3]}{index % 4096:03x}",
                )
            )
    return rows


def real_holdout_records(*, low_kappa_cohort: str | None = None) -> list[AnnotationRecord]:
    rows: list[AnnotationRecord] = []
    for cohort_index, cohort in enumerate(REAL_COHORTS):
        for index in range(1000):
            item_id = f"holdout-{cohort}-{index}"
            family_hash = f"c{cohort_index:03x}{index:060x}"[-64:]
            source_hash = f"d{cohort_index:03x}{index:060x}"[-64:]
            label_a = "yes" if index % 2 == 0 else "no"
            if cohort == low_kappa_cohort:
                label_b = "no" if label_a == "yes" else "yes"
            elif index % 10 == 0:
                label_b = "no" if label_a == "yes" else "yes"
            else:
                label_b = label_a
            rows.append(
                record(
                    item_id,
                    "ann-a",
                    label_a,
                    pool="holdout",
                    modality=cohort,
                    family_hash=family_hash,
                    source_hash=source_hash,
                )
            )
            rows.append(
                record(
                    item_id,
                    "ann-b",
                    label_b,
                    pool="holdout",
                    modality=cohort,
                    family_hash=family_hash,
                    source_hash=source_hash,
                )
            )
    return rows


def test_redacted_canary_summary_excludes_labels_payload_and_item_ids() -> None:
    summary = summarize(manifest(), complete_canary_records())
    dumped = summary.model_dump(mode="json")
    encoded = json.dumps(dumped)
    assert dumped["contract_version"] == CONTRACT_VERSION
    assert dumped["passed"] is True
    assert dumped["cohorts"]["text:dev"]["items"] == 2
    assert dumped["cohorts"]["text:dev"]["annotations"] == 4
    assert dumped["redaction"] == "aggregate_counts_no_payload_labels_or_item_ids"
    for forbidden in ["allow", "deny", "review", "item-1", "item-2", "ann-a", "ann-b"]:
        assert forbidden not in encoded


def test_loader_rejects_payload_fields(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "item_id": "item-1",
                "modality": "text",
                "pool": "dev",
                "annotator_id": "ann-a",
                "label": "allow",
                "family_hash": HEX_A,
                "source_hash": HEX_B,
                "question": "sealed prompt",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sealed record"):
        load_annotations(annotations)


def test_development_loader_rejects_label_payload_and_unknown_license(tmp_path: Path) -> None:
    development = tmp_path / "development.jsonl"
    development.write_text(
        json.dumps(
            {
                "item_id": "dev-1",
                "modality": "text",
                "family_hash": HEX_A,
                "source_hash": HEX_B,
                "source_dataset": "dataset",
                "source_revision": "rev",
                "license": "cc-by-4.0",
                "label_origin": "program_truth",
                "label": "secret",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sealed record"):
        load_development(development)
    with pytest.raises(ValidationError):
        dev_record("dev-2", license="unknown")


def test_missing_or_duplicate_annotations_fail_closed() -> None:
    summary = summarize(manifest(), [record("item-1", "ann-a", "allow")])
    assert summary.passed is False
    assert summary.failures["wrong_annotation_count"] == 1
    assert summary.failures["missing_items"] == 2
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate(
            {
                "item_id": "item-1",
                "modality": "text",
                "pool": "dev",
                "annotator_id": "ann-a",
                "label": "allow",
                "family_hash": HEX_A,
                "source_hash": HEX_B,
                "extra": "forbidden",
            }
        )


def test_real_rejects_legacy_four_cohort_shape_and_low_min_kappa() -> None:
    with pytest.raises(ValidationError, match="six evaluation cohorts"):
        SealedManifest(
            audit_id="real-audit",
            dataset_version="sealed-real-v1",
            source_digest=HEX_A,
            canary=False,
            annotator_ids=("ann-a", "ann-b"),
            expected_counts={
                "text": {"dev": 500, "holdout": 1000},
                "image": {"dev": 500, "holdout": 1000},
                "audio": {"dev": 500, "holdout": 1000},
                "video": {"dev": 500, "holdout": 1000},
            },
        )
    with pytest.raises(ValidationError, match="min_kappa"):
        real_manifest(min_kappa=0.69)


def test_real_requires_screenshot_and_crossmodal_expected_counts() -> None:
    expected = {cohort: {"dev": 500, "holdout": 1000} for cohort in REAL_COHORTS}
    del expected["screenshot"]
    with pytest.raises(ValidationError, match="six evaluation cohorts"):
        real_manifest(expected_counts=expected)
    expected = {cohort: {"dev": 500, "holdout": 1000} for cohort in REAL_COHORTS}
    expected["crossmodal"] = {"dev": 499, "holdout": 1000}
    with pytest.raises(ValidationError, match="dev=500"):
        real_manifest(expected_counts=expected)


def test_real_missing_development_ledger_fails() -> None:
    summary = summarize(real_manifest(), real_holdout_records())
    assert summary.passed is False
    assert summary.failures["missing_development_ledger"] == 1
    assert summary.blocked_reason is not None


def test_real_complete_dev_ledger_has_no_dev_kappa_but_still_blocked_by_isolation() -> None:
    summary = summarize(real_manifest(), real_holdout_records(), real_development_records())
    assert summary.passed is False
    assert summary.failures == {}
    assert summary.blocked_reason is not None
    for cohort in REAL_COHORTS:
        dev = summary.cohorts[f"{cohort}:dev"]
        assert dev.items == 500
        assert dev.annotations == 0
        assert dev.kappa is None
        assert dev.source_origin_counts == {"program_truth": 500}
        holdout = summary.cohorts[f"{cohort}:holdout"]
        assert holdout.items == 1000
        assert holdout.annotations == 2000
        assert holdout.kappa is not None
        assert holdout.kappa >= 0.7


def test_real_low_holdout_kappa_rejected() -> None:
    summary = summarize(
        real_manifest(),
        real_holdout_records(low_kappa_cohort="text"),
        real_development_records(),
    )
    assert summary.failures["kappa_below_threshold"] >= 1
    assert summary.cohorts["text:holdout"].kappa is not None
    assert summary.cohorts["text:holdout"].kappa < 0.7


def test_cross_pool_family_overlap_fails() -> None:
    rows = complete_canary_records()
    rows[-1] = rows[-1].model_copy(update={"family_hash": HEX_A})
    summary = summarize(manifest(), rows)
    assert summary.passed is False
    assert summary.failures["cross_pool_family_overlap"] == 1


def test_duplicate_item_across_cohorts_fails() -> None:
    fixed_manifest = SealedManifest(
        audit_id="canary-audit",
        dataset_version="sealed-canary-v1",
        source_digest=HEX_A,
        canary=True,
        annotator_ids=("ann-a", "ann-b"),
        expected_counts={"text": {"dev": 1}, "image": {"dev": 1}},
    )
    summary = summarize(
        fixed_manifest,
        [
            record("same-item", "ann-a", "allow"),
            record("same-item", "ann-b", "allow"),
            record("same-item", "ann-a", "allow", modality="image", family_hash=HEX_D),
            record("same-item", "ann-b", "allow", modality="image", family_hash=HEX_D),
        ],
    )
    assert summary.passed is False
    assert summary.failures["duplicate_item_across_cohorts"] == 1


def test_holdout_missing_one_human_fails() -> None:
    fixed_manifest = SealedManifest(
        audit_id="canary-audit",
        dataset_version="sealed-canary-v1",
        source_digest=HEX_A,
        canary=True,
        annotator_ids=("ann-a", "ann-b"),
        expected_counts={"text": {"holdout": 1}},
    )
    summary = summarize(fixed_manifest, [record("item-1", "ann-a", "allow", pool="holdout")])
    assert summary.passed is False
    assert summary.failures["wrong_annotation_count"] == 1


@pytest.mark.parametrize("different_family", [False, True])
def test_same_item_across_dev_and_holdout_rejected_even_with_different_hashes(
    different_family: bool,
) -> None:
    fixed_manifest = SealedManifest(
        audit_id="canary-cross-pool",
        dataset_version="synthetic-v2",
        source_digest=HEX_A,
        canary=True,
        annotator_ids=("ann-a", "ann-b"),
        expected_counts={"text": {"dev": 1, "holdout": 1}},
    )
    family = HEX_D if different_family else HEX_A
    summary = summarize(
        fixed_manifest,
        [
            record("same-item", "ann-a", "allow", pool="holdout", family_hash=family),
            record("same-item", "ann-b", "allow", pool="holdout", family_hash=family),
        ],
        [dev_record("same-item", family_hash=HEX_A)],
    )
    assert summary.passed is False
    assert summary.failures["duplicate_item_across_pools"] == 1
    if different_family:
        assert summary.failures["item_identity_mismatch"] == 1


def test_kappa_reports_perfect_degenerate_agreement() -> None:
    assert cohen_kappa(["same", "same"], ["same", "same"]) == (1.0, True)


def test_kappa_pairs_by_frozen_annotator_identity_not_row_order() -> None:
    fixed_manifest = SealedManifest(
        audit_id="canary-audit",
        dataset_version="sealed-canary-v1",
        source_digest=HEX_A,
        canary=True,
        annotator_ids=("ann-a", "ann-b"),
        expected_counts={"text": {"dev": 3}},
    )
    rows = [
        record("item-1", "ann-a", "allow"),
        record("item-1", "ann-b", "deny"),
        record("item-2", "ann-a", "deny"),
        record("item-2", "ann-b", "deny"),
        record("item-3", "ann-a", "allow"),
        record("item-3", "ann-b", "allow"),
    ]
    permuted = [rows[1], rows[0], rows[3], rows[2], rows[5], rows[4]]
    assert (
        summarize(fixed_manifest, rows).cohorts["text:dev"].kappa
        == summarize(fixed_manifest, permuted).cohorts["text:dev"].kappa
    )


def test_rejects_unexpected_annotator_and_unmanifested_records() -> None:
    text_dev_only = SealedManifest(
        audit_id="canary-audit",
        dataset_version="sealed-canary-v1",
        source_digest=HEX_A,
        canary=True,
        annotator_ids=("ann-a", "ann-b"),
        expected_counts={"text": {"dev": 1}},
    )
    summary = summarize(
        text_dev_only,
        [
            record("item-1", "ann-a", "allow"),
            record("item-1", "ann-c", "allow"),
            record("item-2", "ann-a", "allow", pool="holdout"),
            record("item-3", "ann-a", "allow", modality="image"),
        ],
    )
    assert summary.passed is False
    assert summary.failures["unexpected_annotator"] == 1
    assert summary.failures["unexpected_pool"] == 1
    assert summary.failures["unexpected_modality"] == 1


def test_manifest_fails_closed_for_empty_zero_and_bad_real_shape() -> None:
    with pytest.raises(ValidationError):
        SealedManifest(
            audit_id="bad",
            dataset_version="sealed-canary-v1",
            source_digest=HEX_A,
            canary=True,
            annotator_ids=("ann-a", "ann-b"),
            expected_counts={},
        )
    with pytest.raises(ValidationError):
        SealedManifest(
            audit_id="bad",
            dataset_version="sealed-canary-v1",
            source_digest=HEX_A,
            canary=True,
            annotator_ids=("ann-a", "ann-b"),
            expected_counts={"text": {"dev": 0}},
        )
    with pytest.raises(ValidationError):
        SealedManifest(
            audit_id="bad",
            dataset_version="sealed-real-v1",
            source_digest=HEX_A,
            canary=False,
            annotator_ids=("ann-a", "ann-b"),
            expected_counts={"text": {"dev": 500, "holdout": 1000}},
        )


def test_cli_redacts_validation_errors(tmp_path: Path) -> None:
    sentinel = "SECRET_SENTINEL_ITEM_LABEL_AND_CONTENT"
    manifest_path = tmp_path / "manifest.json"
    annotations_path = tmp_path / "annotations.jsonl"
    out_path = tmp_path / "summary.json"
    manifest_path.write_text(manifest().model_dump_json(), encoding="utf-8")
    annotations_path.write_text(
        json.dumps(
            {
                "item_id": sentinel,
                "modality": "text",
                "pool": "dev",
                "annotator_id": "ann-a",
                "label": sentinel,
                "family_hash": HEX_A,
                "source_hash": HEX_B,
                "question": sentinel,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.sealed_gate",
            "--manifest",
            str(manifest_path),
            "--annotations",
            str(annotations_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    encoded = out_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "invalid_annotations" in encoded
    assert sentinel not in result.stdout
    assert sentinel not in result.stderr
    assert sentinel not in encoded


def test_cli_redacts_development_validation_errors(tmp_path: Path) -> None:
    sentinel = "SECRET_SENTINEL_DEV_LABEL"
    manifest_path = tmp_path / "manifest.json"
    annotations_path = tmp_path / "annotations.jsonl"
    development_path = tmp_path / "development.jsonl"
    out_path = tmp_path / "summary.json"
    manifest_path.write_text(manifest().model_dump_json(), encoding="utf-8")
    annotations_path.write_text("", encoding="utf-8")
    development_path.write_text(
        json.dumps(
            {
                "item_id": sentinel,
                "modality": "text",
                "family_hash": HEX_A,
                "source_hash": HEX_B,
                "source_dataset": "dataset",
                "source_revision": "rev",
                "license": "cc-by-4.0",
                "label_origin": "program_truth",
                "label": sentinel,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.sealed_gate",
            "--manifest",
            str(manifest_path),
            "--annotations",
            str(annotations_path),
            "--development",
            str(development_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    encoded = out_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "invalid_development" in encoded
    assert sentinel not in result.stdout
    assert sentinel not in result.stderr
    assert sentinel not in encoded


def test_cli_writes_redacted_summary_and_returns_fail_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    annotations_path = tmp_path / "annotations.jsonl"
    out_path = tmp_path / "summary.json"
    manifest_path.write_text(manifest().model_dump_json(), encoding="utf-8")
    annotations_path.write_text(
        "\n".join(record.model_dump_json() for record in complete_canary_records()) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.sealed_gate",
            "--manifest",
            str(manifest_path),
            "--annotations",
            str(annotations_path),
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    encoded = out_path.read_text(encoding="utf-8")
    summary = json.loads(encoded)
    assert summary["passed"] is True
    assert summary["contract_version"] == CONTRACT_VERSION
    assert "allow" not in encoded
