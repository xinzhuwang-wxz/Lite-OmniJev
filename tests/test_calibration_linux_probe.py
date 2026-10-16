"""CPU safety checks for an inert-by-default root execution probe."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".scratch/s5/probe-calibration-linux-boundary.py"


def module() -> Any:
    spec = importlib.util.spec_from_file_location("cal_linux_probe", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_default_is_inert() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    assert report["status"] == "prepared_not_executed"
    assert report["hardcap_seconds"] == 180
    assert report["cases"] == [
        "normal",
        "timeout",
        "term",
        "kill",
        "competition",
        "stale_recovery",
        "private_snapshot",
    ]


def test_polling_has_a_deadline(monkeypatch: Any) -> None:
    probe = module()
    values = iter([0, 0, 2])
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(values))
    monkeypatch.setattr(probe.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="bounded_wait_expired"):
        probe.wait_until(lambda: False, 1)


def test_lingering_identity_fails_evidence(monkeypatch: Any) -> None:
    probe = module()

    def occupied(ids: set[int]) -> None:
        assert ids == {64001}
        raise probe.broker.JobRejected("identity_unavailable")

    monkeypatch.setattr(probe.broker, "check_process_ids", occupied)
    assert probe.no_processes(64001) is False


def test_worker_payload_is_valid_python_and_uses_shared_verifier() -> None:
    probe = module()
    compile(probe.WORKER, "<synthetic worker>", "exec")
    assert "verify_child_identity()" in probe.WORKER
    assert "synthetic public model" in probe.WORKER
