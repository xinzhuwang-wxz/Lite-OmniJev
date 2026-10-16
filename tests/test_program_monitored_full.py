"""CPU-only full-course resume, window, and process-group supervision checks."""

import importlib.util
import math
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest


def controller():
    path = (
        Path(__file__).resolve().parents[1] / ".scratch/s4/run-program-stratified-monitored-full.py"
    )
    spec = importlib.util.spec_from_file_location("monitored_full", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_full_course_supersession_and_90minute_limit():
    module = controller()
    policy = {
        "policy_family": "monitored_full_course",
        "decision": "run_monitored_full_course",
        "user_authorized_supersession": True,
        "prepared_report_sha256": module.PREPARED_SHA,
        "config_sha256": module.CONFIG_SHA,
        "initializer_sha256": module.h.INIT_SHA,
        "target_step": 100,
        "window_start_utc": "2026-10-16T10:00:00Z",
        "window_deadline_utc": "2026-10-16T11:30:00Z",
        "adr_sha256": "a" * 64,
    }
    now = datetime(2026, 10, 16, 10, tzinfo=UTC).timestamp()
    assert module.validate_policy(policy, now) == now + 5400
    for change in [
        {"user_authorized_supersession": False},
        {"policy_family": "old_window"},
        {"target_step": 200},
        {"window_deadline_utc": "2026-10-16T12:00:00Z"},
    ]:
        with pytest.raises(ValueError):
            module.validate_policy(policy | change, now)


def test_segments_resume_without_replaying_prior_updates():
    module = controller()
    assert module.segment_ends(10) == [20, 40, 60, 80, 100]
    assert module.segment_ends(10, True) == [20]
    assert module.segment_ends(20) == [40, 60, 80, 100]
    assert module.segment_ends(100) == []
    with pytest.raises(ValueError):
        module.segment_ends(35)


def test_resume_checks_actual_saved_optimizer_rng_files_and_exact_trajectory(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from train.checkpoint import load_checkpoint_metadata, save_checkpoint

    module = controller()
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(2)).sum().backward()
    optimizer.step()
    order = list(range(1200))
    random.Random(42).shuffle(order)
    losses = [1.0] * 10
    config = {"steps": 100, "rank": 16, "alpha": 32}
    path = save_checkpoint(
        tmp_path,
        step=10,
        config=config,
        model_revision="cpu",
        data_sha256=module.DATA_SHA,
        asset_hashes={},
        order=order,
        losses=losses,
        model=model,
        optimizer=optimizer,
    )
    metadata = load_checkpoint_metadata(path)
    raw = {
        "status": "paused",
        "exit_code": 0,
        "steps_completed": 10,
        "data_sha256": module.DATA_SHA,
        "init_adapter": {"weights_sha256": module.h.INIT_SHA},
        "config": config,
        "losses": losses,
    }
    module.validate_resume(raw, metadata, 10)
    for change in [
        {"config": config | {"rank": 64}},
        {"order": list(reversed(order))},
        {"step": 20},
        {"losses": [2.0] * 10},
    ]:
        with pytest.raises(ValueError):
            module.validate_resume(raw, metadata | change, 10)
    (path / "optimizer.safetensors").write_bytes(b"changed optimizer")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint_metadata(path)


@pytest.mark.parametrize(
    "correct,brier,expected",
    [(31, 0.47, False), (30, 0.50, False), (29, 0.47, True), (29, 0.43, False)],
)
def test_noisy_small_dev_changes_do_not_stop_full_course(correct, brier, expected):
    module = controller()
    best = {"candidate": {"count": 50, "accuracy": 34 / 50, "brier": 0.44}}
    current = {"candidate": {"count": 50, "accuracy": correct / 50, "brier": brier}}
    assert module.catastrophic(current, best) is expected
    current["candidate"]["brier"] = math.nan
    with pytest.raises(ValueError):
        module.catastrophic(current, best)


def test_cpu_supervisor_terminates_timed_out_process_group(monkeypatch, tmp_path):
    module = controller()
    monkeypatch.setattr(module, "SOURCE", tmp_path)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        module.monitor_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            tmp_path / "timeout.log",
            dict(os.environ),
            0.1,
        )
    assert time.monotonic() - started < 3
