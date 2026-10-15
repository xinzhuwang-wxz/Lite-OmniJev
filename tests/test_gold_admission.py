"""Synthetic metadata contract tests; no real gold or private payloads."""

from copy import deepcopy
from hashlib import sha256

import pytest

from eval.gold_admission import COHORTS, assemble_admission


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def valid() -> dict:
    manifest = digest("manifest")
    code = digest("code")
    selected = []
    for cohort in sorted(COHORTS):
        for pool, count in (("dev", 500), ("calibration", 3000), ("holdout", 1000)):
            selected.append(
                {
                    "cohort": cohort,
                    "pool": pool,
                    "artifact_sha256": digest(cohort + pool),
                    "source_evidence_sha256": digest("source-report"),
                    "counts": {
                        "questions": count,
                        "unique_questions": count,
                        "unique_original_questions": count,
                        "original_families": count,
                        "media": 0 if cohort == "text" else count,
                        "dependency_components": count,
                    },
                    "frozen_original_dev_questions": 500 if pool == "dev" else 0,
                    "invalid_labels": 0,
                    "missing_labels": 0,
                    "teacher_labels": 0,
                    "soft_labels": 0,
                    "canary_questions": 0,
                    "nonreal_questions": 0,
                }
            )
    artifacts = [row["artifact_sha256"] for row in selected]
    return {
        "schema_version": "gold-admission-input-v1",
        "inventory_sha256": manifest,
        "inventory_artifact_sha256": artifacts,
        "bound_source_artifact_sha256": [digest("raw-source")],
        "selection_audit_sha256": digest("selection-report"),
        "selection_audit_code_sha256": digest("selection-code"),
        "selection_audit_inventory_sha256": manifest,
        "isolation_report_sha256": digest("isolation-report"),
        "isolation": {
            "source_manifest_sha256": manifest,
            "translated_manifest_sha256": digest("translated"),
            "code_sha256": code,
            "config_sha256": digest("config"),
            "nonce_sha256": digest("nonce"),
            "duration_seconds": 1.0,
            "exit_code": 0,
            "ordinary_identity_boundary": True,
            "outcome": "isolated",
            "categories": [],
            "isolation": {"outcome": "isolated"},
        },
        "runtime": {
            "report_sha256": digest("runtime"),
            "launcher_code_sha256": code,
            "status": "passed",
            "failure_injection": "passed",
            "log_redaction": "passed",
            "scope": "ordinary_identity_root_visible",
        },
        "sources": [
            {
                "adapter": "text-source-replay-v1",
                "report_sha256": digest("source-report"),
                "code_sha256": code,
                "source_manifest_sha256": manifest,
                "source_artifact_sha256": [digest("raw-source")],
                "output_artifact_sha256": artifacts,
                "status": "replayed",
            }
        ],
        "selected": selected,
    }


def test_structural_success_never_authenticates_or_passes_final_gate() -> None:
    result = assemble_admission(valid())
    assert result.structural_admission == "passed"
    assert result.source_authenticity == "requires_artifact_verification"
    assert result.runtime_permissions == "requires_artifact_verification"
    assert result.final_gold_gate == "not_evaluated"
    assert result.iid_claim is False


@pytest.mark.parametrize(
    "mutation,category",
    [
        ("missing_cohort", "quota"),
        ("inflated_variants", "selection"),
        ("bad_source_binding", "source_binding"),
        ("canary", "labels"),
        ("runtime_code", "runtime_binding"),
        ("stale_inventory", "inventory_binding"),
        ("overlap", "isolation"),
        ("unknown_field", "invalid_metadata"),
        ("duplicate", "selection"),
        ("unknown_cohort", "invalid_metadata"),
        ("missing_frozen_dev", "quota"),
        ("teacher", "labels"),
    ],
)
def test_rejections(mutation: str, category: str) -> None:
    data = deepcopy(valid())
    if mutation == "missing_cohort":
        data["selected"] = [r for r in data["selected"] if r["cohort"] != "audio"]
    elif mutation == "inflated_variants":
        data["selected"][0]["counts"]["unique_original_questions"] -= 1
    elif mutation == "bad_source_binding":
        data["sources"][0]["output_artifact_sha256"] = [digest("different")]
    elif mutation == "canary":
        data["selected"][0]["canary_questions"] = 1
    elif mutation == "runtime_code":
        data["runtime"]["launcher_code_sha256"] = digest("stale")
    elif mutation == "stale_inventory":
        data["isolation"]["source_manifest_sha256"] = digest("stale")
    elif mutation == "overlap":
        data["isolation"]["isolation"]["overlapping_components"] = 1
    elif mutation == "unknown_field":
        data["trusted_success"] = True
    elif mutation == "duplicate":
        data["selected"].append(deepcopy(data["selected"][0]))
    elif mutation == "unknown_cohort":
        data["selected"][0]["cohort"] = "unknown"
    elif mutation == "missing_frozen_dev":
        for row in data["selected"]:
            row["frozen_original_dev_questions"] = 0
    elif mutation == "teacher":
        data["selected"][0]["teacher_labels"] = 1
    result = assemble_admission(data)
    assert result.structural_admission == "rejected"
    assert category in result.categories


def test_raw_source_catalogue_is_distinct_from_sample_inventory() -> None:
    data = valid()
    raw_hash = data["sources"][0]["source_artifact_sha256"][0]
    assert raw_hash not in data["inventory_artifact_sha256"]
    assert assemble_admission(data).structural_admission == "passed"
    data["bound_source_artifact_sha256"] = [digest("tampered-raw-catalogue")]
    assert "source_binding" in assemble_admission(data).categories


def test_catalogues_require_unique_digests_not_historical_observations() -> None:
    data = valid()
    data["inventory_artifact_sha256"].append(data["inventory_artifact_sha256"][0])
    assert "inventory_binding" in assemble_admission(data).categories
    data = valid()
    data["bound_source_artifact_sha256"] *= 2
    assert "source_binding" in assemble_admission(data).categories
