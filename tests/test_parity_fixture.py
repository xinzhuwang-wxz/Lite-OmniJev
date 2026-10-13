import hashlib
import json
import subprocess
import sys

import pytest

from data.parity_fixture import (
    CALIBRATION_GENERATOR,
    CONFIRMATION_GENERATOR,
    build_fixture,
    generate_calibration_samples,
    generate_confirmation_samples,
    load_calibration,
    quota_counts,
)
from data.schema import (
    HardTarget,
    InsufficientTarget,
    Sample,
    validate_pool_isolation,
)

EXPECTED_QUOTAS = {
    "choice-2": 12,
    "choice-3": 12,
    "choice-5": 12,
    "choice-10": 12,
    "choice-20": 12,
    "noul": 20,
    "score-2": 5,
    "score-3": 5,
    "score-5": 5,
    "score-10": 5,
}


def test_calibration_fixture_is_reproducible_independent_and_quota_balanced():
    first = generate_calibration_samples()
    second = generate_calibration_samples()
    assert [sample.model_dump() for sample in first] == [sample.model_dump() for sample in second]
    assert len(first) == 100
    validate_pool_isolation(first)
    assert {sample.meta.pool for sample in first} == {"calibration"}
    assert {sample.meta.source_dataset for sample in first} == {"lite-omnijev-parity-program-truth"}
    assert len({sample.meta.family_id for sample in first}) == 100
    assert quota_counts(first) == EXPECTED_QUOTAS
    assert all(len(sample.questions) == 1 and sample.meta.modality == ["text"] for sample in first)
    assert all(
        CALIBRATION_GENERATOR in question.target.source
        for sample in first
        for question in sample.questions
        if isinstance(question.target, HardTarget)
    )


def test_confirmation_fixture_uses_distinct_semantic_generator_and_dev_pool():
    calibration = generate_calibration_samples()
    confirmation = generate_confirmation_samples()
    assert {sample.meta.pool for sample in confirmation} == {"dev"}
    assert quota_counts(confirmation) == EXPECTED_QUOTAS
    assert len({sample.meta.family_id for sample in confirmation}) == 100
    assert {sample.meta.family_id for sample in calibration}.isdisjoint(
        {sample.meta.family_id for sample in confirmation}
    )
    assert all(
        CONFIRMATION_GENERATOR in question.target.source
        for sample in confirmation
        for question in sample.questions
        if isinstance(question.target, HardTarget)
    )
    assert all(CALIBRATION_GENERATOR not in sample.prefix.state for sample in confirmation)


def test_fixtures_cover_sufficient_missing_and_conflict_as_explicit_u_cases():
    for samples in (generate_calibration_samples(), generate_confirmation_samples()):
        insufficient = [
            sample
            for sample in samples
            if isinstance(sample.questions[0].target, InsufficientTarget)
        ]
        sufficient = [
            sample for sample in samples if isinstance(sample.questions[0].target, HardTarget)
        ]
        assert len(insufficient) >= 30
        assert len(sufficient) >= 30
        text = "\n".join(sample.prefix.state for sample in insufficient)
        assert "unavailable" in text or "missing" in text
        assert "contradictory" in text
        assert all(
            "answer U" in sample.prefix.state or "require U" in sample.prefix.state
            for sample in insufficient
        )


def test_manifest_records_file_hashes_rows_pool_and_generator_families(tmp_path):
    manifest = build_fixture(tmp_path)
    assert set(manifest["files"]) == {"calibration.jsonl", "confirmation.jsonl"}
    calibration = tmp_path / "calibration.jsonl"
    confirmation = tmp_path / "confirmation.jsonl"
    assert manifest["files"]["calibration.jsonl"] == {
        "sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "rows": 100,
        "pool": "calibration",
        "generator_families": [CALIBRATION_GENERATOR],
        "effective_sample_families": 100,
        "quota_counts": EXPECTED_QUOTAS,
        "case_status_counts": {"insufficient": 65, "sufficient": 35},
    }
    assert (
        manifest["files"]["confirmation.jsonl"]["sha256"]
        == hashlib.sha256(confirmation.read_bytes()).hexdigest()
    )
    assert manifest["files"]["confirmation.jsonl"]["pool"] == "dev"
    assert manifest["files"]["confirmation.jsonl"]["generator_families"] == [CONFIRMATION_GENERATOR]
    assert (tmp_path / "manifest.json").is_file()
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_load_calibration_validates_basename_pool_shape_and_families(tmp_path):
    build_fixture(tmp_path)
    samples = load_calibration(tmp_path / "calibration.jsonl")
    assert len(samples) == 100
    with pytest.raises(ValueError, match="basename"):
        load_calibration(tmp_path / "renamed.jsonl")
    bad = tmp_path / "bad-calibration.jsonl"
    payload = samples[0].model_dump()
    payload["id"] = "bad"
    payload["meta"]["pool"] = "dev"
    bad.write_text(Sample.model_validate(payload).model_dump_json() + "\n")
    with pytest.raises(ValueError, match="basename"):
        load_calibration(bad)
    pool_bad = tmp_path / "calibration.jsonl"
    pool_bad.write_text(Sample.model_validate(payload).model_dump_json() + "\n")
    with pytest.raises(ValueError, match="exactly 100"):
        load_calibration(pool_bad)


def test_cli_writes_requested_directory(tmp_path):
    output = tmp_path / "fixture"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.build_parity_fixture",
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(result.stdout)
    assert manifest["files"]["calibration.jsonl"]["rows"] == 100
    assert (output / "calibration.jsonl").is_file()
    assert (output / "confirmation.jsonl").is_file()
    assert (output / "manifest.json").is_file()
