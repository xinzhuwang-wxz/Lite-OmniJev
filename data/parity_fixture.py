"""Independent program-truth parity fixtures for low-precision gate calibration."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from data.schema import Pool, Sample, validate_pool_isolation

FixturePool = Literal["calibration", "dev"]
CaseStatus = Literal["sufficient", "missing", "conflict"]
QuestionType = Literal["choice", "noul", "score"]

CALIBRATION_GENERATOR = "calibration-interval-register-v1"
CONFIRMATION_GENERATOR = "confirmation-event-sequence-v1"
DATA_VERSION = "parity-fixture-v1"
SOURCE_REVISION = "lite-omnijev-parity-fixture-v1"
LICENSE = "internal"
USAGE = "commercial"
CHOICE_COUNTS = (2, 3, 5, 10, 20)
SCORE_LEVELS = (2, 3, 5, 10)


def stable_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode()).hexdigest()


def _letter(index: int) -> str:
    return chr(65 + index)


def _family_question(question: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in question.items() if key != "target"}


def _sample(
    *,
    sample_id: str,
    state: str,
    question: dict[str, Any],
    pool: FixturePool,
    generator_family: str,
    case_status: CaseStatus,
    quota_key: str,
) -> Sample:
    family_payload = {
        "state": state,
        "question": _family_question(question),
        "generator_family": generator_family,
        "case_status": case_status,
        "quota_key": quota_key,
    }
    family_id = stable_hash(family_payload)
    return Sample.model_validate(
        {
            "id": sample_id,
            "prefix": {"state": state},
            "questions": [question],
            "meta": {
                "modality": ["text"],
                "source_dataset": "lite-omnijev-parity-program-truth",
                "source_revision": SOURCE_REVISION,
                "source_sha256": family_id,
                "license": LICENSE,
                "usage": USAGE,
                "pool": pool,
                "family_id": family_id,
                "data_version": DATA_VERSION,
            },
        }
    )


def _target(value: str | int | None, source: str) -> dict[str, Any]:
    if value is None:
        return {"kind": "insufficient"}
    return {"kind": "hard", "value": value, "source": source}


def _calibration_choice(index: int, option_count: int, status: CaseStatus) -> Sample:
    correct = (index * 3 + option_count) % option_count
    low, high = 40 + index % 5, 50 + index % 5
    values = [low - 11 - i for i in range(option_count)]
    values[correct] = low + (index % (high - low + 1))
    options = [f"asset-{index:02d}-{i:02d}" for i in range(option_count)]
    if status == "missing":
        values[correct] = -1
        records = [
            f"{name}: value unavailable" if i == correct else f"{name}: value {value}"
            for i, (name, value) in enumerate(zip(options, values, strict=True))
        ]
        value: str | None = None
    elif status == "conflict":
        records = [
            (
                f"{name}: value {low - 7}; second register value {low + 2}"
                if i == correct
                else f"{name}: value {value}"
            )
            for i, (name, value) in enumerate(zip(options, values, strict=True))
        ]
        value = None
    else:
        records = [f"{name}: value {value}" for name, value in zip(options, values, strict=True)]
        value = _letter(correct)
    state = (
        "Calibration interval register. "
        f"Target interval is inclusive [{low}, {high}]. "
        "If any candidate needed for the answer is unavailable or contradictory, answer U. "
        + "; ".join(records)
        + "."
    )
    question = {
        "qid": "interval-choice",
        "type": "choice",
        "instruction": "Which listed asset is inside the target interval?",
        "options": options,
        "target": _target(value, f"program-oracle:{CALIBRATION_GENERATOR}"),
    }
    return _sample(
        sample_id=f"parity-calibration-choice-{option_count}-{index:03d}",
        state=state,
        question=question,
        pool="calibration",
        generator_family=CALIBRATION_GENERATOR,
        case_status=status,
        quota_key=f"choice-{option_count}",
    )


def _calibration_noul(index: int, status: CaseStatus) -> Sample:
    threshold = 30 + index % 9
    value = threshold + 4 if index % 2 == 0 else threshold - 4
    item = f"probe-{index:02d}"
    if status == "missing":
        fact = f"{item}: value unavailable"
        target: str | None = None
    elif status == "conflict":
        fact = f"{item}: value {threshold - 3}; second register value {threshold + 3}"
        target = None
    else:
        fact = f"{item}: value {value}"
        target = "yes" if value >= threshold else "no"
    state = (
        "Calibration threshold register. "
        f"Pass rule: a probe passes when value is at least {threshold}. "
        "Unavailable or contradictory required values require U. "
        f"{fact}."
    )
    question = {
        "qid": "threshold-noul",
        "type": "noul",
        "instruction": f"Does {item} pass the threshold rule?",
        "target": _target(target, f"program-oracle:{CALIBRATION_GENERATOR}"),
    }
    return _sample(
        sample_id=f"parity-calibration-noul-{index:03d}",
        state=state,
        question=question,
        pool="calibration",
        generator_family=CALIBRATION_GENERATOR,
        case_status=status,
        quota_key="noul",
    )


def _calibration_score(index: int, levels: int, status: CaseStatus) -> Sample:
    active = index % levels
    signals = [i < active for i in range(levels - 1)]
    if active == levels - 1:
        signals = [True for _ in signals]
    names = [f"s{i}" for i in range(len(signals))]
    if status == "missing" and names:
        facts = [
            f"{name}=unknown" if i == len(names) - 1 else f"{name}={'on' if flag else 'off'}"
            for i, (name, flag) in enumerate(zip(names, signals, strict=True))
        ]
        target: int | None = None
    elif status == "conflict" and names:
        facts = [
            f"{name}=on and {name}=off" if i == 0 else f"{name}={'on' if flag else 'off'}"
            for i, (name, flag) in enumerate(zip(names, signals, strict=True))
        ]
        target = None
    else:
        facts = [
            f"{name}={'on' if flag else 'off'}" for name, flag in zip(names, signals, strict=True)
        ]
        target = active
    state = (
        f"Calibration score board case {index}. "
        f"Score is the count of signals that are on, capped to levels 0..{levels - 1}. "
        "Unknown or contradictory required signals require U. " + "; ".join(facts) + "."
    )
    question = {
        "qid": "signal-score",
        "type": "score",
        "instruction": "What score does the board define?",
        "score_levels": levels,
        "score_descriptions": [f"{i} active signals" for i in range(levels)],
        "target": _target(target, f"program-oracle:{CALIBRATION_GENERATOR}"),
    }
    return _sample(
        sample_id=f"parity-calibration-score-{levels}-{index:03d}",
        state=state,
        question=question,
        pool="calibration",
        generator_family=CALIBRATION_GENERATOR,
        case_status=status,
        quota_key=f"score-{levels}",
    )


def _confirmation_choice(index: int, option_count: int, status: CaseStatus) -> Sample:
    anchor = 100 + index
    winner = (index * 5 + 1) % option_count
    options = [f"action-{index:02d}-{i:02d}" for i in range(option_count)]
    times = [anchor + 20 + i * 3 for i in range(option_count)]
    times[winner] = anchor + 1 + index % 4
    if status == "missing":
        facts = [
            f"{name}: time missing" if i == winner else f"{name}: time {time}"
            for i, (name, time) in enumerate(zip(options, times, strict=True))
        ]
        value: str | None = None
    elif status == "conflict":
        facts = [
            (
                f"{name}: time {anchor + 1}; second log time {anchor + 40}"
                if i == winner
                else f"{name}: time {time}"
            )
            for i, (name, time) in enumerate(zip(options, times, strict=True))
        ]
        value = None
    else:
        facts = [f"{name}: time {time}" for name, time in zip(options, times, strict=True)]
        value = _letter(winner)
    state = (
        "Confirmation event log. "
        f"Anchor time is {anchor}. "
        "Choose the action with the earliest known time after the anchor. "
        "Missing or contradictory required timestamps require U. " + "; ".join(facts) + "."
    )
    question = {
        "qid": "event-choice",
        "type": "choice",
        "instruction": "Which action is the earliest event after the anchor?",
        "options": options,
        "target": _target(value, f"program-oracle:{CONFIRMATION_GENERATOR}"),
    }
    return _sample(
        sample_id=f"parity-confirmation-choice-{option_count}-{index:03d}",
        state=state,
        question=question,
        pool="dev",
        generator_family=CONFIRMATION_GENERATOR,
        case_status=status,
        quota_key=f"choice-{option_count}",
    )


def _confirmation_noul(index: int, status: CaseStatus) -> Sample:
    deadline = 200 + index
    event_time = deadline - 2 if index % 2 == 0 else deadline + 2
    event = f"event-{index:02d}"
    if status == "missing":
        fact = f"{event}: time missing"
        target: str | None = None
    elif status == "conflict":
        fact = f"{event}: time {deadline - 2}; second log time {deadline + 2}"
        target = None
    else:
        fact = f"{event}: time {event_time}"
        target = "yes" if event_time <= deadline else "no"
    state = (
        "Confirmation deadline log. "
        f"An event is on time when its timestamp is at or before {deadline}. "
        "Missing or contradictory required timestamps require U. "
        f"{fact}."
    )
    question = {
        "qid": "deadline-noul",
        "type": "noul",
        "instruction": f"Is {event} on time?",
        "target": _target(target, f"program-oracle:{CONFIRMATION_GENERATOR}"),
    }
    return _sample(
        sample_id=f"parity-confirmation-noul-{index:03d}",
        state=state,
        question=question,
        pool="dev",
        generator_family=CONFIRMATION_GENERATOR,
        case_status=status,
        quota_key="noul",
    )


def _confirmation_score(index: int, levels: int, status: CaseStatus) -> Sample:
    expected = index % levels
    stages = [i < expected for i in range(levels - 1)]
    names = [f"stage-{i}" for i in range(len(stages))]
    if status == "missing" and names:
        facts = [
            (
                f"{name}=missing"
                if i == expected % len(names)
                else f"{name}={'done' if done else 'open'}"
            )
            for i, (name, done) in enumerate(zip(names, stages, strict=True))
        ]
        target: int | None = None
    elif status == "conflict" and names:
        facts = [
            f"{name}=done and {name}=open" if i == 0 else f"{name}={'done' if done else 'open'}"
            for i, (name, done) in enumerate(zip(names, stages, strict=True))
        ]
        target = None
    else:
        facts = [
            f"{name}={'done' if done else 'open'}" for name, done in zip(names, stages, strict=True)
        ]
        target = expected
    state = (
        f"Confirmation sequence log case {index}. "
        f"Score is the count of completed ordered stages, capped to levels 0..{levels - 1}. "
        "Missing or contradictory required stage states require U. " + "; ".join(facts) + "."
    )
    question = {
        "qid": "sequence-score",
        "type": "score",
        "instruction": "What completion score does the sequence define?",
        "score_levels": levels,
        "score_descriptions": [f"{i} completed stages" for i in range(levels)],
        "target": _target(target, f"program-oracle:{CONFIRMATION_GENERATOR}"),
    }
    return _sample(
        sample_id=f"parity-confirmation-score-{levels}-{index:03d}",
        state=state,
        question=question,
        pool="dev",
        generator_family=CONFIRMATION_GENERATOR,
        case_status=status,
        quota_key=f"score-{levels}",
    )


def _statuses(count: int) -> list[CaseStatus]:
    pattern: tuple[CaseStatus, ...] = ("sufficient", "missing", "conflict")
    return [pattern[i % len(pattern)] for i in range(count)]


def generate_calibration_samples() -> list[Sample]:
    samples: list[Sample] = []
    case = 0
    for option_count in CHOICE_COUNTS:
        for status in _statuses(12):
            samples.append(_calibration_choice(case, option_count, status))
            case += 1
    for status in _statuses(20):
        samples.append(_calibration_noul(case, status))
        case += 1
    for levels in SCORE_LEVELS:
        for status in _statuses(5):
            samples.append(_calibration_score(case, levels, status))
            case += 1
    _validate_fixture(samples, "calibration", CALIBRATION_GENERATOR)
    return samples


def generate_confirmation_samples() -> list[Sample]:
    samples: list[Sample] = []
    case = 0
    for option_count in CHOICE_COUNTS:
        for status in _statuses(12):
            samples.append(_confirmation_choice(case, option_count, status))
            case += 1
    for status in _statuses(20):
        samples.append(_confirmation_noul(case, status))
        case += 1
    for levels in SCORE_LEVELS:
        for status in _statuses(5):
            samples.append(_confirmation_score(case, levels, status))
            case += 1
    _validate_fixture(samples, "dev", CONFIRMATION_GENERATOR)
    return samples


def _quota_key(sample: Sample) -> str:
    question = sample.questions[0]
    if question.type == "choice":
        return f"choice-{len(question.options or [])}"
    if question.type == "score":
        return f"score-{question.score_levels}"
    return "noul"


def quota_counts(samples: list[Sample]) -> dict[str, int]:
    return dict(sorted(Counter(_quota_key(sample) for sample in samples).items()))


def case_status_counts(samples: list[Sample]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sample in samples:
        target = sample.questions[0].target
        key = "sufficient" if target is not None and target.kind == "hard" else "insufficient"
        counts[key] += 1
    return dict(sorted(counts.items()))


def _validate_fixture(samples: list[Sample], pool: FixturePool, generator_family: str) -> None:
    if len(samples) != 100:
        raise ValueError("parity fixture requires exactly 100 samples")
    validate_pool_isolation(samples)
    if len({sample.meta.family_id for sample in samples}) != 100:
        raise ValueError("parity fixture requires 100 effective sample families")
    if any(sample.meta.pool != pool for sample in samples):
        raise ValueError(f"all samples must use pool {pool}")
    if any(sample.meta.modality != ["text"] or sample.prefix.media for sample in samples):
        raise ValueError("parity fixture is text-only")
    if any(len(sample.questions) != 1 for sample in samples):
        raise ValueError("parity fixture requires one question per sample")
    for sample in samples:
        target = sample.questions[0].target
        if target is not None and target.kind == "hard" and generator_family not in target.source:
            raise ValueError("hard targets must name the generator oracle")
    expected = {
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
    if quota_counts(samples) != expected:
        raise ValueError("parity fixture quota mismatch")


def write_jsonl(path: Path, samples: list[Sample]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(sample.model_dump_json() for sample in samples) + "\n"
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_manifest(
    path: Path, samples: list[Sample], pool: FixturePool, generator_family: str
) -> dict[str, Any]:
    generators = {
        CALIBRATION_GENERATOR: (
            _calibration_choice,
            _calibration_noul,
            _calibration_score,
            generate_calibration_samples,
        ),
        CONFIRMATION_GENERATOR: (
            _confirmation_choice,
            _confirmation_noul,
            _confirmation_score,
            generate_confirmation_samples,
        ),
    }
    source = "\n".join(inspect.getsource(function) for function in generators[generator_family])
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(samples),
        "pool": pool,
        "generator_families": [generator_family],
        "generator_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "generator_rule_count": 3,
        "effective_sample_families": len({sample.meta.family_id for sample in samples}),
        "independence_limit": (
            "100 case families are content-isolated rows, not 100 independent algorithms "
            "or statistically independent production samples; each file uses one combined "
            "generator family with three task rules."
        ),
        "quota_counts": quota_counts(samples),
        "case_status_counts": case_status_counts(samples),
    }


def build_fixture(output_dir: Path) -> dict[str, Any]:
    calibration = generate_calibration_samples()
    confirmation = generate_confirmation_samples()
    calibration_path = output_dir / "calibration.jsonl"
    confirmation_path = output_dir / "confirmation.jsonl"
    manifest_path = output_dir / "manifest.json"
    existing = [
        path for path in (calibration_path, confirmation_path, manifest_path) if path.exists()
    ]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite existing parity fixture files: {names}")
    write_jsonl(calibration_path, calibration)
    write_jsonl(confirmation_path, confirmation)
    manifest = {
        "schema_version": 1,
        "data_version": DATA_VERSION,
        "source_revision": SOURCE_REVISION,
        "license": LICENSE,
        "usage": USAGE,
        "scope": (
            "program-truth calibration/dev fixtures for numerical parity policy; "
            "not a production-quality or holdout benchmark"
        ),
        "files": {
            "calibration.jsonl": file_manifest(
                calibration_path, calibration, "calibration", CALIBRATION_GENERATOR
            ),
            "confirmation.jsonl": file_manifest(
                confirmation_path, confirmation, "dev", CONFIRMATION_GENERATOR
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_samples(path: Path, *, expected_pool: Pool) -> list[Sample]:
    samples = [
        Sample.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]
    validate_pool_isolation(samples)
    if len(samples) != 100:
        raise ValueError("expected exactly 100 samples")
    if any(sample.meta.pool != expected_pool for sample in samples):
        raise ValueError(f"expected all samples in pool {expected_pool}")
    if any(sample.meta.modality != ["text"] or sample.prefix.media for sample in samples):
        raise ValueError("expected text-only fixture samples")
    if any(len(sample.questions) != 1 for sample in samples):
        raise ValueError("expected one question per sample")
    return samples


def load_calibration(path: Path) -> list[Sample]:
    if path.name != "calibration.jsonl":
        raise ValueError("calibration fixture basename must be calibration.jsonl")
    return load_samples(path, expected_pool="calibration")


def load_confirmation(path: Path) -> list[Sample]:
    if path.name != "confirmation.jsonl":
        raise ValueError("confirmation fixture basename must be confirmation.jsonl")
    return load_samples(path, expected_pool="dev")
