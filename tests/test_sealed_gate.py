import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from eval.sealed_gate import (
    AnnotationRecord,
    SealedManifest,
    cohen_kappa,
    load_annotations,
    summarize,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def manifest(*, canary: bool = True) -> SealedManifest:
    return SealedManifest(
        audit_id="canary-audit",
        dataset_version="sealed-canary-v1",
        source_digest=HEX_A,
        canary=canary,
        expected_counts={"text": {"dev": 2, "holdout": 1}},
    )


def record(
    item_id: str,
    annotator_id: str,
    label: str,
    *,
    pool: Literal["dev", "holdout"] = "dev",
) -> AnnotationRecord:
    return AnnotationRecord(
        item_id=item_id,
        modality="text",
        pool=pool,
        annotator_id=annotator_id,
        label=label,
        family_hash=HEX_A if pool == "dev" else HEX_C,
        source_hash=HEX_B,
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


def test_redacted_canary_summary_excludes_labels_payload_and_item_ids() -> None:
    summary = summarize(manifest(), complete_canary_records())
    dumped = summary.model_dump(mode="json")
    encoded = json.dumps(dumped)
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
    with pytest.raises(ValueError, match="sealed payload"):
        load_annotations(annotations)


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


def test_non_canary_requires_kappa_threshold_and_never_passes_dev_summary() -> None:
    summary = summarize(
        manifest(canary=False),
        [
            record("item-1", "ann-a", "allow"),
            record("item-1", "ann-b", "deny"),
            record("item-2", "ann-a", "deny"),
            record("item-2", "ann-b", "allow"),
            record("item-3", "ann-a", "review", pool="holdout"),
            record("item-3", "ann-b", "review", pool="holdout"),
        ],
    )
    assert summary.passed is False
    assert summary.cohorts["text:dev"].kappa < 0.7
    assert summary.failures["kappa_below_threshold"] >= 1
    assert summary.blocked_reason is not None


def test_non_canary_rejects_degenerate_kappa() -> None:
    summary = summarize(manifest(canary=False), complete_canary_records())
    assert summary.passed is False
    assert summary.failures["kappa_degenerate"] >= 1


def test_cross_pool_family_overlap_fails() -> None:
    rows = complete_canary_records()
    rows[-1] = rows[-1].model_copy(update={"family_hash": HEX_A})
    summary = summarize(manifest(), rows)
    assert summary.passed is False
    assert summary.failures["cross_pool_family_overlap"] == 1


def test_kappa_reports_perfect_degenerate_agreement() -> None:
    assert cohen_kappa(["same", "same"], ["same", "same"]) == (1.0, True)


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
    )
    assert result.returncode == 0
    encoded = out_path.read_text(encoding="utf-8")
    summary = json.loads(encoded)
    assert summary["passed"] is True
    assert "allow" not in encoded
