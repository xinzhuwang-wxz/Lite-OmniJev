"""CPU tests for immutable course windows and nonautomatic continuation."""

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".scratch/s4"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decision(module, start=0):
    return {
        "decision": "approve_next10",
        "start_step": start,
        "end_step": start + 10,
        "max_total_steps": 40,
        "invocation_steps": 10,
        "max_seconds": 240,
        "prepared_report_sha256": "a" * 64,
        "initializer_sha256": module.INIT_SHA,
        "strategy_sha256": module.STRATEGY_SHA,
        "window_start_utc": "2026-10-16T10:00:00Z",
        "window_deadline_utc": "2026-10-16T10:20:00Z",
        "learning_observed": True,
        "previous_launch_sha256": "b" * 64,
        "screening_report_sha256": "c" * 64,
    }


def timestamp(minute):
    return datetime(2026, 10, 16, 10, minute, tzinfo=UTC).timestamp()


def test_resume_cannot_slide_the_twenty_minute_window():
    launcher = load("launch-program-stratified-pool.py")
    frozen = launcher.validate_decision(decision(launcher), "a" * 64, timestamp(0))
    continuation = decision(launcher, 10)
    launcher.validate_decision(continuation, "a" * 64, timestamp(7), frozen)
    changed = continuation | {
        "window_start_utc": "2026-10-16T10:05:00Z",
        "window_deadline_utc": "2026-10-16T10:25:00Z",
    }
    with pytest.raises(ValueError, match="extend original window"):
        launcher.validate_decision(changed, "a" * 64, timestamp(7), frozen)
    with pytest.raises(ValueError, match="insufficient"):
        launcher.validate_decision(continuation, "a" * 64, timestamp(12), frozen)


@pytest.mark.parametrize(
    "change",
    [
        {"max_seconds": 241},
        {"max_total_steps": 50},
        {"end_step": 30},
        {"learning_observed": False},
        {"prepared_report_sha256": "bad"},
        {"previous_launch_sha256": None},
    ],
)
def test_continuation_requires_fresh_bound_review(change):
    launcher = load("launch-program-stratified-pool.py")
    with pytest.raises(ValueError):
        launcher.validate_decision(decision(launcher, 10) | change, "a" * 64, timestamp(5))


def test_after20_reserves_time_for_old_modalities_as_well_as_video():
    launcher = load("launch-program-stratified-pool.py")
    later = decision(launcher, 20)
    with pytest.raises(ValueError, match="old-retention"):
        launcher.validate_decision(later, "a" * 64, timestamp(5))
    later["old_retention_reserve_seconds"] = 120
    launcher.validate_decision(later, "a" * 64, timestamp(5))
    with pytest.raises(ValueError, match="insufficient"):
        launcher.validate_decision(later, "a" * 64, timestamp(10))


def test_all_step_snapshots_are_distinct_and_correspond_to_matching_invocations(tmp_path):
    launcher = load("launch-program-stratified-pool.py")
    first, second = launcher.paths(tmp_path, 10), launcher.paths(tmp_path, 20)
    assert set(first.values()).isdisjoint(second.values())
    assert first["training"].name == "s4-program-stratified-training-step0010.json"
    assert second["runtime"].name == "s4-program-stratified-runtime-step0020.json"


@pytest.mark.parametrize(
    "correct,brier,expected",
    [
        (30, 0.4, "stop"),
        (31, 0.4, "gray_root_review_no_automatic_continuation"),
        (32, 0.46, "root_learning_review_before_step20"),
        (35, 0.48, "stop"),
    ],
)
def test_step10_disposition_never_automatically_promotes(correct, brier, expected):
    evaluator = load("evaluate-program-stratified-dev50.py")
    assert (
        evaluator.screening_disposition({"count": 50, "accuracy": correct / 50, "brier": brier}, 10)
        == expected
    )


def test_resume_gate_rejects_gray31_and_requires_wide_review_after20():
    launcher = load("launch-program-stratified-pool.py")
    screen = {
        "status": "completed",
        "exit_code": 0,
        "step": 10,
        "paired_input_hashes_verified": 50,
        "candidate": {"count": 50, "accuracy": 0.64, "brier": 0.46},
    }
    launcher.continuation_screen(screen, 10, {})
    screen["candidate"]["accuracy"] = 0.62
    with pytest.raises(ValueError):
        launcher.continuation_screen(screen, 10, {})
    screen["candidate"]["accuracy"] = 0.66
    screen["step"] = 20
    with pytest.raises(ValueError, match="wide-retention"):
        launcher.continuation_screen(screen, 20, {})
    launcher.continuation_screen(
        screen, 20, {"wide_retention_passed": True, "video150_report_sha256": "d" * 64}
    )
