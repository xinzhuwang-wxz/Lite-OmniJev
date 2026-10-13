"""Paired report diagnostics preserve differences without granting acceptance."""

import copy
import json
import math
import sys

import pytest

from scripts import parity_impact
from scripts.p0_parity import row


def reports(left=(-20.0, -22.0, -0.001), right=(-22.0, -20.0, -0.001)):
    def rows(logprobs):
        return [
            row({"id": str(i), "input_ids": [1, i], "token_ids": [10, 11, 12]}, list(logprobs))
            for i in range(100)
        ]

    return (
        {
            "status": "reference-ready",
            "action": "hf-reference",
            "provenance": {"model": "x"},
            "rows": rows(left),
        },
        {
            "status": "failed",
            "action": "compare-vllm",
            "provenance": {"model": "x"},
            "rows": rows(right),
        },
    )


def test_high_u_does_not_hide_conditional_changes_or_create_gate_pass():
    result = parity_impact.analyze(*reports())
    assert result["metrics"]["conditional_max"]["max"] > 0.7
    assert result["metrics"]["joint_TV"]["max"] < 1e-8
    assert result["top_answer_set_changes"] == 100
    assert result["slices"]["high_U"]["count"] == 100
    assert result["status"] == "analysis-only"
    assert result["gate_changed"] is result["quality_verified"] is False


def test_tie_change_is_not_reported_as_stable_answer():
    result = parity_impact.analyze(*reports((-1, -1, -20), (-0.9, -1.1, -20)))
    assert result["top_answer_set_changes"] == 100
    assert result["slices"]["low_U"]["top_answer_set_changes"] == 100
    assert result["slices"]["high_U"]["conditional_max"] is None


@pytest.mark.parametrize(
    "mutation", ["probability", "identity", "provenance", "missing", "duplicate"]
)
def test_malformed_or_unpaired_evidence_is_rejected(mutation):
    reference, candidate = reports()
    if mutation == "probability":
        candidate["rows"][0]["conditional"][0] += 0.01
    elif mutation == "identity":
        candidate["rows"][0]["input_sha256"] = "other"
    elif mutation == "provenance":
        candidate["provenance"]["model"] = "other"
    elif mutation == "missing":
        candidate["rows"].pop()
    else:
        for report in (reference, candidate):
            report["rows"][1] = copy.deepcopy(report["rows"][0])
    with pytest.raises(ValueError):
        parity_impact.analyze(reference, candidate)


def test_cli_preserves_existing_report(tmp_path, monkeypatch):
    reference, candidate = reports()
    paths = [tmp_path / name for name in ("hf.json", "vllm.json", "impact.json")]
    for path, report in zip(paths, (reference, candidate), strict=False):
        path.write_text(json.dumps(report))
    paths[2].write_text("previous evidence")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parity-impact",
            "--reference",
            str(paths[0]),
            "--candidate",
            str(paths[1]),
            "--output",
            str(paths[2]),
        ],
    )
    with pytest.raises(FileExistsError):
        parity_impact.main()
    assert paths[2].read_text() == "previous evidence"


def test_libm_roundoff_is_allowed_but_not_material_tampering():
    reference, candidate = reports()
    original = candidate["rows"][0]["p_U"]
    candidate["rows"][0]["p_U"] = math.nextafter(original, 0.0)
    assert parity_impact.analyze(reference, candidate)["sample_count"] == 100
    candidate["rows"][0]["p_U"] = original - 1e-8
    with pytest.raises(ValueError, match="probabilities"):
        parity_impact.analyze(reference, candidate)
