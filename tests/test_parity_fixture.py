import hashlib
import json
import re
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
    stable_hash,
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


def independent_target(sample):
    state = sample.prefix.state
    question = sample.questions[0]
    facts = state.rsplit(". ", maxsplit=1)[-1]
    if "unavailable" in facts or "missing" in facts or "unknown" in facts:
        return None
    if "second register value" in facts or "second log time" in facts:
        return None
    if "=on and" in facts or "=done and" in facts:
        return None
    if question.qid == "interval-choice":
        low, high = map(int, re.search(r"\[(\d+), (\d+)\]", state).groups())
        values = dict(re.findall(r"(asset-\d+-\d+): value (-?\d+)", state))
        matches = [
            option for option in question.options or [] if low <= int(values[option]) <= high
        ]
        assert len(matches) == 1
        return chr(65 + (question.options or []).index(matches[0]))
    if question.qid == "threshold-noul":
        threshold = int(re.search(r"at least (\d+)", state).group(1))
        item = re.search(r"Does (probe-\d+) pass", question.instruction).group(1)
        value = int(re.search(rf"{item}: value (-?\d+)", state).group(1))
        return "yes" if value >= threshold else "no"
    if question.qid == "signal-score":
        levels = question.score_levels
        assert levels is not None
        facts = re.findall(r"s\d+=(on|off)", state)
        assert len(facts) == levels - 1
        return min(sum(fact == "on" for fact in facts), levels - 1)
    if question.qid == "event-choice":
        anchor = int(re.search(r"Anchor time is (\d+)", state).group(1))
        times = dict(re.findall(r"(action-\d+-\d+): time (\d+)", state))
        after_anchor = [
            (option, int(times[option]))
            for option in question.options or []
            if int(times[option]) > anchor
        ]
        winner = min(after_anchor, key=lambda item: item[1])[0]
        return chr(65 + (question.options or []).index(winner))
    if question.qid == "deadline-noul":
        deadline = int(re.search(r"at or before (\d+)", state).group(1))
        event = re.search(r"Is (event-\d+) on time", question.instruction).group(1)
        value = int(re.search(rf"{event}: time (\d+)", state).group(1))
        return "yes" if value <= deadline else "no"
    if question.qid == "sequence-score":
        levels = question.score_levels
        assert levels is not None
        facts = re.findall(r"stage-\d+=(done|open)", state)
        assert len(facts) == levels - 1
        return min(sum(fact == "done" for fact in facts), levels - 1)
    raise AssertionError(f"unhandled qid {question.qid}")


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


def test_rendered_state_oracle_recomputes_all_targets_independently():
    for sample in [*generate_calibration_samples(), *generate_confirmation_samples()]:
        expected = independent_target(sample)
        target = sample.questions[0].target
        if expected is None:
            assert isinstance(target, InsufficientTarget)
        else:
            assert isinstance(target, HardTarget)
            assert target.value == expected


def test_family_hash_excludes_target_truth():
    sample = generate_calibration_samples()[0]
    question = sample.questions[0]
    payload = sample.model_dump()
    assert isinstance(question.target, HardTarget)
    payload["questions"][0]["target"] = {
        **question.target.model_dump(),
        "value": "B" if question.target.value != "B" else "A",
    }
    relabeled = Sample.model_validate(payload)
    assert relabeled.meta.family_id == sample.meta.family_id
    expected_family = stable_hash(
        {
            "state": sample.prefix.state,
            "question": {
                key: value
                for key, value in sample.questions[0].model_dump(exclude_none=True).items()
                if key != "target"
            },
            "generator_family": CALIBRATION_GENERATOR,
            "case_status": "sufficient",
            "quota_key": "choice-2",
        }
    )
    assert sample.meta.family_id == expected_family


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
        "generator_source_sha256": manifest["files"]["calibration.jsonl"][
            "generator_source_sha256"
        ],
        "generator_rule_count": 3,
        "effective_sample_families": 100,
        "independence_limit": (
            "100 case families are content-isolated rows, not 100 independent algorithms "
            "or statistically independent production samples; each file uses one combined "
            "generator family with three task rules."
        ),
        "quota_counts": EXPECTED_QUOTAS,
        "case_status_counts": {"insufficient": 65, "sufficient": 35},
    }
    assert len(manifest["files"]["calibration.jsonl"]["generator_source_sha256"]) == 64
    assert (
        manifest["files"]["confirmation.jsonl"]["sha256"]
        == hashlib.sha256(confirmation.read_bytes()).hexdigest()
    )
    assert manifest["files"]["confirmation.jsonl"]["pool"] == "dev"
    assert manifest["files"]["confirmation.jsonl"]["generator_families"] == [CONFIRMATION_GENERATOR]
    assert manifest["files"]["confirmation.jsonl"]["generator_rule_count"] == 3
    assert len(manifest["files"]["confirmation.jsonl"]["generator_source_sha256"]) == 64
    assert (tmp_path / "manifest.json").is_file()
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_build_fixture_refuses_to_overwrite_existing_bytes(tmp_path):
    target = tmp_path / "calibration.jsonl"
    target.write_text("keep me\n")
    before = target.read_bytes()
    with pytest.raises(FileExistsError, match="calibration.jsonl"):
        build_fixture(tmp_path)
    assert target.read_bytes() == before
    assert not (tmp_path / "confirmation.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()


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
