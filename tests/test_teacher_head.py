"""Teacher head conversion tests use tiny tensor fixtures, never upstream weights."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import train.teacher_head as teacher_head
from scripts.teacher_head import main
from train.checkpoint import file_sha256
from train.teacher_head import (
    PINNED_SOURCE_REVISION,
    IsolationError,
    _copy_regular_nofollow,
    convert_head,
    load_expected_spec,
)


def _write_head(path: Path, tensors: dict[str, object]) -> str:
    torch = pytest.importorskip("torch")

    torch.save(tensors, path)
    return file_sha256(path)


def _spec(path: Path, tensors: dict[str, object]) -> Path:
    spec = {"schema_version": 1, "tensors": {}}
    for key, tensor in tensors.items():
        spec["tensors"][key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    path.write_text(json.dumps(spec))
    return path


def test_convert_head_writes_audited_safetensors_and_round_trips(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    tensors = {
        "decision_head.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "decision_head.bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
    }
    source = tmp_path / "head.pt"
    expected_sha = _write_head(source, tensors)
    expected = load_expected_spec(_spec(tmp_path / "spec.json", tensors))
    output = tmp_path / "head.safetensors"

    report = convert_head(
        source,
        output,
        expected_sha256=expected_sha,
        expected_size=source.stat().st_size,
        expected_spec=expected,
    )

    assert report["status"] == "converted"
    assert report["source_revision"] == PINNED_SOURCE_REVISION
    assert report["source_sha256"] == expected_sha
    assert report["output_sha256"] == file_sha256(output)
    assert set(report["tensors"]) == set(tensors)
    loaded = safetensors.load_file(str(output), device="cpu")
    assert all(torch.equal(loaded[key], tensors[key]) for key in tensors)


def test_convert_head_rejects_wrong_hash_before_loading(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    source = tmp_path / "head.pt"
    _write_head(source, {"x": torch.ones(1)})
    called = False

    def fail_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load should not run")

    monkeypatch.setattr(torch, "load", fail_load)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        convert_head(source, tmp_path / "head.safetensors", expected_sha256="0" * 64)
    assert called is False


def test_convert_head_uses_weights_only_and_rejects_unsafe_payload(tmp_path):
    torch = pytest.importorskip("torch")

    source = tmp_path / "head.pt"
    torch.save(torch.nn.Linear(1, 1), source)
    expected_sha = file_sha256(source)
    with pytest.raises(ValueError, match="weights_only|tensor mapping"):
        convert_head(source, tmp_path / "head.safetensors", expected_sha256=expected_sha)


def test_convert_head_rejects_missing_expected_key(tmp_path):
    torch = pytest.importorskip("torch")
    tensors = {"decision_head.weight": torch.ones(2, 2)}
    source = tmp_path / "head.pt"
    expected_sha = _write_head(source, tensors)
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tensors": {
                    "decision_head.weight": {"shape": [2, 2], "dtype": "torch.float32"},
                    "decision_head.bias": {"shape": [2], "dtype": "torch.float32"},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="keys mismatch"):
        convert_head(
            source,
            tmp_path / "head.safetensors",
            expected_sha256=expected_sha,
            expected_spec=load_expected_spec(spec),
        )


def test_convert_head_rejects_nonfinite_float_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "head.pt"
    expected_sha = _write_head(source, {"x": torch.tensor([float("nan")])})
    with pytest.raises(ValueError, match="nonfinite"):
        convert_head(source, tmp_path / "head.safetensors", expected_sha256=expected_sha)


def test_convert_head_rejects_forbidden_source(tmp_path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "head.pt"
    expected_sha = _write_head(source, {"x": torch.ones(1)})
    with pytest.raises(ValueError, match="pinned Jev-Omni"):
        convert_head(
            source,
            tmp_path / "head.safetensors",
            expected_sha256=expected_sha,
            source_repo="OmniJev/JEV-27B",
        )


def test_cli_records_failure_without_leaking_payload(tmp_path):
    source = tmp_path / "head.pt"
    source.write_bytes(b"not a torch file")
    report = tmp_path / "report.json"
    code = main(
        [
            "convert",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "head.safetensors"),
            "--expected-sha256",
            hashlib.sha256(source.read_bytes()).hexdigest(),
            "--report",
            str(report),
        ]
    )
    assert code == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "blocked"
    assert payload["blocker"] == "conversion_failed"
    assert "not a torch file" not in json.dumps(payload)


def test_cli_isolate_requires_preapproved_spec(tmp_path):
    source = tmp_path / "head.pt"
    source.write_bytes(b"not loaded before spec check")
    report = tmp_path / "report.json"
    output = tmp_path / "head.safetensors"
    code = main(
        [
            "convert",
            "--isolate",
            "--source",
            str(source),
            "--output",
            str(output),
            "--expected-sha256",
            hashlib.sha256(source.read_bytes()).hexdigest(),
            "--report",
            str(report),
        ]
    )
    payload = json.loads(report.read_text())
    assert code == 1
    assert payload["status"] == "blocked"
    assert payload["blocker"] == "isolation_unavailable"
    assert not output.exists()


def test_parent_copy_refuses_child_symlink_to_host_secret(tmp_path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlink unavailable")
    host_secret = tmp_path / "host-secret"
    host_secret.write_text("private-key-material")
    child_output = tmp_path / "jail-output.safetensors"
    child_output.symlink_to(host_secret)
    destination = tmp_path / "admitted.safetensors"
    destination.write_text("previous-good-output")

    with pytest.raises(IsolationError, match="regular file"):
        _copy_regular_nofollow(child_output, destination, max_bytes=1024)

    assert destination.read_text() == "previous-good-output"


def test_isolated_cli_refuses_child_fifo_without_overwriting_output(tmp_path, monkeypatch):
    if not hasattr(os, "mkfifo"):
        pytest.skip("fifo unavailable")
    source = tmp_path / "head.pt"
    source.write_bytes(b"fixture")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"schema_version": 1, "tensors": {"x": {"shape": [1], "dtype": "torch.uint8"}}})
    )
    destination = tmp_path / "head.safetensors"
    destination.write_text("previous-good-output")
    report = tmp_path / "report.json"

    monkeypatch.setattr(teacher_head.platform, "system", lambda: "Linux")
    monkeypatch.setattr(teacher_head.os, "geteuid", lambda: 0)

    def fake_run(command, **_kwargs):
        jail = Path(command[command.index("--jail-root") + 1])
        os.mkfifo(jail / "output" / "head.safetensors")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "schema_version": 1,
                    "status": "converted",
                    "output_sha256": "0" * 64,
                    "exit_code": 0,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(teacher_head.subprocess, "run", fake_run)
    code = main(
        [
            "convert",
            "--isolate",
            "--source",
            str(source),
            "--output",
            str(destination),
            "--expected-sha256",
            hashlib.sha256(source.read_bytes()).hexdigest(),
            "--expected-spec",
            str(spec),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(report.read_text())
    assert code == 1
    assert payload["status"] == "blocked"
    assert payload["blocker"] == "isolation_unavailable"
    assert destination.read_text() == "previous-good-output"
    assert "previous-good-output" not in json.dumps(payload)


def test_isolated_cli_timeout_is_redacted_and_preserves_output(tmp_path, monkeypatch):
    source = tmp_path / "head.pt"
    source.write_bytes(b"fixture")
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps({"schema_version": 1, "tensors": {"x": {"shape": [1], "dtype": "torch.uint8"}}})
    )
    destination = tmp_path / "head.safetensors"
    destination.write_text("previous-good-output")
    report = tmp_path / "report.json"

    monkeypatch.setattr(teacher_head.platform, "system", lambda: "Linux")
    monkeypatch.setattr(teacher_head.os, "geteuid", lambda: 0)

    def fake_run(command, **_kwargs):
        raise subprocess.TimeoutExpired(command, teacher_head.ISOLATED_CONVERSION_TIMEOUT_SECONDS)

    monkeypatch.setattr(teacher_head.subprocess, "run", fake_run)
    code = main(
        [
            "convert",
            "--isolate",
            "--source",
            str(source),
            "--output",
            str(destination),
            "--expected-sha256",
            hashlib.sha256(source.read_bytes()).hexdigest(),
            "--expected-spec",
            str(spec),
            "--report",
            str(report),
        ]
    )

    payload = json.loads(report.read_text())
    assert code == 1
    assert payload["status"] == "blocked"
    assert payload["blocker"] == "isolation_unavailable"
    assert destination.read_text() == "previous-good-output"
    assert "TimeoutExpired" not in json.dumps(payload)


def test_cli_direct_conversion_report_marks_nonisolated_mode(tmp_path):
    torch = pytest.importorskip("torch")
    source = tmp_path / "head.pt"
    expected_sha = _write_head(source, {"x": torch.ones(1)})
    report = tmp_path / "report.json"
    code = main(
        [
            "convert",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "head.safetensors"),
            "--expected-sha256",
            expected_sha,
            "--report",
            str(report),
        ]
    )
    assert code == 0
    payload = json.loads(report.read_text())
    assert payload["status"] == "converted"
    assert payload["isolation"]["mode"] == "direct"
    assert payload["isolation"]["network_denied"] is False


def test_seccomp_boundary_denies_socket_or_reports_unavailable():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.teacher_head",
            "probe-isolation",
            "--report",
            os.devnull,
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode in {0, 1}
    payload = json.loads(completed.stdout)
    assert payload["status"] in {"ready", "blocked"}
    if payload["status"] == "ready":
        assert payload["network_denied"] is True
