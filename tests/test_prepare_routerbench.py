import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prepare_routerbench as prep


def _child_report(output_path: Path, mode: str = "0shot") -> dict:
    raw = output_path.read_bytes()
    return {
        "mode": mode,
        "output_name": f"routerbench_{mode}.jsonl",
        "output_bytes": len(raw),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "roundtrip": {
            "row_count": 1,
            "schema_sha256": "a" * 64,
            "row_jsonl_sha256": hashlib.sha256(raw).hexdigest(),
            "source_jsonsafe_sha256": hashlib.sha256(raw).hexdigest(),
        },
        "versions": {"pandas": "3.0.6", "numpy": "2.4.6", "pyarrow": "25.0.1"},
        "isolation": {
            "mode": "chroot+dropuid+seccomp",
            "filesystem_boundary": "chroot",
            "uid": 65534,
            "gid": 65534,
            "network_denied": True,
            "host_canary_denied": True,
            "no_new_privs": True,
            "stdin": "DEVNULL",
            "close_fds": True,
            "rlimits": {
                "fsize_bytes": prep.OUTPUT_LIMIT_BYTES,
                "nofile": 64,
                "cpu_seconds": 900,
                "nproc": 32,
                "address_space_bytes": prep.DEFAULT_CHILD_MEMORY_BYTES,
            },
        },
    }


def test_summarize_jsonl_redacts_values_and_hashes_schema(tmp_path):
    data = tmp_path / "routerbench_0shot.jsonl"
    data.write_text(
        '{"question":"secret prompt","answer":"secret answer","score":1}\n'
        '{"question":"another","answer":null,"score":2}\n',
        encoding="utf-8",
    )

    summary = prep.summarize_jsonl(data)
    encoded = json.dumps(summary, ensure_ascii=False)

    assert summary["row_count"] == 2
    assert summary["schema"]["column_count"] == 3
    assert summary["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert "secret prompt" not in encoded
    assert "secret answer" not in encoded
    assert "question" not in encoded
    assert summary["content_redaction"].startswith("report stores")


def test_summarize_jsonl_rejects_corrupt_output(tmp_path):
    data = tmp_path / "bad.jsonl"
    data.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")

    with pytest.raises(prep.IsolationError, match="valid JSONL"):
        prep.summarize_jsonl(data)


def test_validate_source_pins_size_and_hash(tmp_path, monkeypatch):
    source = tmp_path / "routerbench_0shot.pkl"
    source.write_bytes(b"fixture")
    monkeypatch.setitem(
        prep.EXPECTED_PICKLES,
        "0shot",
        {
            "filename": "routerbench_0shot.pkl",
            "bytes": len(b"fixture"),
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
        },
    )

    assert prep.validate_source(source, "0shot")["bytes"] == len(b"fixture")
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="size_mismatch"):
        prep.validate_source(source, "0shot")


def test_parent_never_unpickles_and_refuses_overwrite(tmp_path, monkeypatch):
    raw = b"not a safe pickle but pinned"
    source_root = tmp_path / "input"
    output = tmp_path / "output"
    source_root.mkdir()
    output.mkdir()
    source = source_root / "routerbench_0shot.pkl"
    source.write_bytes(raw)
    (output / "routerbench_0shot.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(
        prep.EXPECTED_PICKLES,
        "0shot",
        {
            "filename": "routerbench_0shot.pkl",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    called = False

    def fake_run_child_process(**kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(prep, "run_child_process", fake_run_child_process)

    with pytest.raises(FileExistsError, match="refuses_overwrite"):
        prep.isolated_convert_one(
            source=source,
            output_dir=output,
            mode="0shot",
            reader_python=Path("/usr/bin/python3"),
        )
    assert called is False


def test_parent_copies_only_regular_bounded_child_output(tmp_path, monkeypatch):
    raw = b"pickle fixture"
    source = tmp_path / "routerbench_0shot.pkl"
    source.write_bytes(raw)
    output = tmp_path / "converted"
    monkeypatch.setitem(
        prep.EXPECTED_PICKLES,
        "0shot",
        {
            "filename": "routerbench_0shot.pkl",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    monkeypatch.setattr(
        prep.tempfile, "TemporaryDirectory", lambda prefix: _TempDir(tmp_path / "jail")
    )

    def fake_child(**kwargs):
        child_output = tmp_path / "jail" / "output" / "routerbench_0shot.jsonl"
        child_output.write_text('{"x": 1}\n', encoding="utf-8")
        report = _child_report(child_output)
        (tmp_path / "jail" / "output" / "child-report.json").write_text(
            json.dumps(report) + "\n", encoding="utf-8"
        )
        return {}

    monkeypatch.setattr(prep, "run_child_process", fake_child)

    result = prep.isolated_convert_one(
        source=source,
        output_dir=output,
        mode="0shot",
        reader_python=Path(sys.executable),
    )

    assert result["plain_output"]["row_count"] == 1
    assert (output / "routerbench_0shot.jsonl").read_text(encoding="utf-8") == '{"x": 1}\n'


def test_parent_rejects_child_symlink_escape(tmp_path, monkeypatch):
    raw = b"pickle fixture"
    source = tmp_path / "routerbench_0shot.pkl"
    source.write_bytes(raw)
    output = tmp_path / "converted"
    secret = tmp_path / "host-secret"
    secret.write_text('{"x": 1}\n', encoding="utf-8")
    monkeypatch.setitem(
        prep.EXPECTED_PICKLES,
        "0shot",
        {
            "filename": "routerbench_0shot.pkl",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    monkeypatch.setattr(
        prep.tempfile, "TemporaryDirectory", lambda prefix: _TempDir(tmp_path / "jail")
    )

    def fake_child(**kwargs):
        child_output = tmp_path / "jail" / "output" / "routerbench_0shot.jsonl"
        child_output.symlink_to(secret)
        report = _child_report(secret)
        (tmp_path / "jail" / "output" / "child-report.json").write_text(
            json.dumps(report) + "\n", encoding="utf-8"
        )
        return {}

    monkeypatch.setattr(prep, "run_child_process", fake_child)

    with pytest.raises(prep.IsolationError, match="regular file"):
        prep.isolated_convert_one(
            source=source,
            output_dir=output,
            mode="0shot",
            reader_python=Path(sys.executable),
        )
    assert not (output / "routerbench_0shot.jsonl").exists()


def test_child_command_uses_scrubbed_env_devnull_close_fds(monkeypatch, tmp_path):
    seen = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            seen["command"] = command
            seen.update(kwargs)
            self.pid = 123456
            self.returncode = 0

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(prep.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(prep, "read_memory_probe", lambda: {"memory_current_bytes": 1})
    monkeypatch.setattr(prep, "_read_process_memory", lambda pid: {"vmrss_bytes": 2})

    result = prep.run_child_process(
        reader_python=Path("/reader/python"),
        jail=tmp_path / "jail",
        host_canary=tmp_path / "canary",
        mode="0shot",
        max_output_bytes=123,
        cpu_seconds=5,
        memory_bytes=prep.DEFAULT_CHILD_MEMORY_BYTES,
    )

    assert result == {
        "resource_probe": {
            "before": {"memory_current_bytes": 1},
            "after": {"memory_current_bytes": 1},
            "child_peak": {"vmrss_bytes": 2},
        }
    }
    assert seen["stdin"] is subprocess.DEVNULL
    assert seen["stdout"] is subprocess.DEVNULL
    assert seen["stderr"] is subprocess.DEVNULL
    assert seen["close_fds"] is True
    assert "SECRET" not in seen["env"]
    assert "PYTHONHOME" not in seen["env"]
    assert "LD_LIBRARY_PATH" not in seen["env"]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert seen["env"]["OMP_NUM_THREADS"] == "1"
    assert seen["command"][:3] == ["/reader/python", "-m", "scripts.prepare_routerbench"]
    assert "--child-report" in seen["command"]
    assert str(prep.DEFAULT_CHILD_MEMORY_BYTES) in seen["command"]


class _TempDir:
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        if self.path.exists():
            for child in sorted(self.path.rglob("*"), reverse=True):
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            self.path.rmdir()
        self.path.mkdir()
        return str(self.path)

    def __exit__(self, exc_type, exc, tb):
        return False


def test_child_report_allowlist_rejects_extra_keys(tmp_path):
    output = tmp_path / "routerbench_0shot.jsonl"
    output.write_text('{"x": 1}\n', encoding="utf-8")
    report = _child_report(output)
    report["leak"] = "must not pass"

    with pytest.raises(prep.IsolationError, match="unexpected keys"):
        prep.validate_child_report(report, mode="0shot")


def test_child_failure_hides_untrusted_stderr(monkeypatch, tmp_path):
    seen = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            seen.update(kwargs)
            self.pid = 123456
            self.returncode = 1
            assert kwargs["stderr"] is subprocess.DEVNULL

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(prep.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(prep, "read_memory_probe", lambda: {"memory_current_bytes": 1})
    monkeypatch.setattr(prep, "_read_process_memory", lambda pid: {})

    with pytest.raises(prep.IsolationError) as raised:
        prep.run_child_process(
            reader_python=Path("/reader/python"),
            jail=tmp_path / "jail",
            host_canary=tmp_path / "canary",
            mode="0shot",
            max_output_bytes=123,
            cpu_seconds=5,
            memory_bytes=prep.DEFAULT_CHILD_MEMORY_BYTES,
        )

    message = str(raised.value)
    assert "category=child_nonzero" in message
    assert "rc=1" in message
    assert "SECRET_SENTINEL" not in message


class _MaliciousDuringUnpickle:
    def __init__(self, host_canary: str):
        self.host_canary = host_canary

    def __reduce__(self):
        payload = f"""
import json
import pathlib
import socket
import sys
proof = {{"host_read": "not-run", "socket": "not-run"}}
try:
    pathlib.Path({self.host_canary!r}).read_text()
    proof["host_read"] = "readable"
except Exception as exc:
    proof["host_read"] = type(exc).__name__
try:
    socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    proof["socket"] = "created"
except Exception as exc:
    proof["socket"] = type(exc).__name__
print("SECRET_SENTINEL_FROM_MALICIOUS_PICKLE", file=sys.stderr)
pathlib.Path("/output/malicious-proof.json").write_text(json.dumps(proof, sort_keys=True))
result = [{{"safe": 1}}]
"""
        expression = "(lambda ns: (exec(" + repr(payload) + ", ns), ns['result'])[1])({})"
        return eval, (expression,)


@pytest.mark.skipif(
    platform.system() != "Linux" or os.geteuid() != 0,
    reason="requires Linux root chroot/seccomp integration",
)
def test_malicious_pickle_payload_denied_during_read_pickle(tmp_path):
    reader = os.environ.get("ROUTERBENCH_READER_PYTHON")
    if not reader:
        pytest.skip("ROUTERBENCH_READER_PYTHON required for real reader integration")
    jail = tmp_path / "jail"
    input_dir = jail / "input"
    output_dir = jail / "output"
    input_dir.mkdir(parents=True)
    output_dir.mkdir()
    jail.chmod(0o755)
    input_dir.chmod(0o755)
    output_dir.chmod(0o777)
    host_canary = tmp_path / "host-canary"
    host_canary.write_text("must-not-read", encoding="utf-8")
    with (input_dir / "input.pkl").open("wb") as stream:
        pickle.dump(_MaliciousDuringUnpickle(str(host_canary)), stream)

    completed = subprocess.run(
        [
            reader,
            "-m",
            "scripts.prepare_routerbench",
            "_child-convert",
            "--jail-root",
            str(jail),
            "--host-canary",
            str(host_canary),
            "--mode",
            "0shot",
            "--max-output-bytes",
            "1048576",
            "--cpu-seconds",
            "60",
            "--child-report",
            "/output/child-report.json",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={
            "PATH": f"{Path(reader).parent}:/usr/bin:/bin",
            "PYTHONPATH": str(Path.cwd()),
            "PYTHONNOUSERSITE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0
    assert "SECRET_SENTINEL_FROM_MALICIOUS_PICKLE" in completed.stderr
    proof = json.loads((output_dir / "malicious-proof.json").read_text(encoding="utf-8"))
    assert proof["host_read"] in {"FileNotFoundError", "PermissionError", "NotADirectoryError"}
    assert proof["socket"] == "PermissionError"
    report = json.loads((output_dir / "child-report.json").read_text(encoding="utf-8"))
    assert report["isolation"]["host_canary_denied"] is True
    assert report["isolation"]["network_denied"] is True
