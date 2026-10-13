import json
import subprocess
import sys
from pathlib import Path

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


def manifest() -> SealedManifest:
    return SealedManifest(
        audit_id="canary-audit",
        dataset_version="sealed-canary-v1",
        source_digest=HEX_A,
        canary=True,
        expected_modalities=["text"],
        expected_items_per_modality={"text": 2},
    )


def record(item_id: str, annotator_id: str, label: str) -> AnnotationRecord:
    return AnnotationRecord(
        item_id=item_id,
        modality="text",
        annotator_id=annotator_id,
        label=label,
        family_hash=HEX_A,
        source_hash=HEX_B,
    )


def test_redacted_canary_summary_excludes_labels_and_payload() -> None:
    summary = summarize(
        manifest(),
        [
            record("item-1", "ann-a", "allow"),
            record("item-1", "ann-b", "allow"),
            record("item-2", "ann-a", "deny"),
            record("item-2", "ann-b", "allow"),
        ],
    )
    dumped = summary.model_dump(mode="json")
    encoded = json.dumps(dumped)
    assert dumped["passed"] is True
    assert dumped["modalities"]["text"]["items"] == 2
    assert dumped["modalities"]["text"]["annotations"] == 4
    assert dumped["redaction"] == "metadata_only_no_payload_or_labels"
    assert "allow" not in encoded
    assert "deny" not in encoded


def test_loader_rejects_payload_fields(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "item_id": "item-1",
                "modality": "text",
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
    assert "expected exactly two raw annotations" in "\n".join(summary.failures)
    assert "missing 1 expected items" in "\n".join(summary.failures)
    with pytest.raises(ValidationError):
        AnnotationRecord.model_validate(
            {
                "item_id": "item-1",
                "modality": "text",
                "annotator_id": "ann-a",
                "label": "allow",
                "family_hash": HEX_A,
                "source_hash": HEX_B,
                "extra": "forbidden",
            }
        )


def test_non_canary_requires_kappa_threshold() -> None:
    real_manifest = manifest().model_copy(update={"canary": False})
    summary = summarize(
        real_manifest,
        [
            record("item-1", "ann-a", "allow"),
            record("item-1", "ann-b", "deny"),
            record("item-2", "ann-a", "deny"),
            record("item-2", "ann-b", "allow"),
        ],
    )
    assert summary.passed is False
    assert summary.modalities["text"].kappa < 0.7
    assert "kappa below 0.7" in "\n".join(summary.failures)


def test_kappa_handles_perfect_degenerate_agreement() -> None:
    assert cohen_kappa(["same", "same"], ["same", "same"]) == 1.0


def test_cli_writes_redacted_summary_and_returns_fail_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    annotations_path = tmp_path / "annotations.jsonl"
    out_path = tmp_path / "summary.json"
    manifest_path.write_text(manifest().model_dump_json(), encoding="utf-8")
    annotations_path.write_text(
        "\n".join(
            [
                record("item-1", "ann-a", "allow").model_dump_json(),
                record("item-1", "ann-b", "allow").model_dump_json(),
                record("item-2", "ann-a", "deny").model_dump_json(),
                record("item-2", "ann-b", "allow").model_dump_json(),
            ]
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
    )
    assert result.returncode == 0
    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert "allow" not in out_path.read_text(encoding="utf-8")
