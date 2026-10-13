"""Safely convert pinned RouterBench pickle artifacts to plain JSONL.

RouterBench sources are public evaluation data. The pickle inputs are never loaded in
normal project processes: conversion runs in a short-lived Linux child that preloads
pandas/numpy/pyarrow, enters a chroot containing only copied input and output
directories, enables no-new-privileges, drops to nobody, and denies socket syscalls
with the existing teacher-head seccomp helper. The parent process validates fixed
hashes and only reads bounded JSONL output plus redacted schema/count summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import stat
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from scripts.assets import write_json
from train.teacher_head import (
    ISOLATED_CONVERSION_TIMEOUT_SECONDS,
    IsolationError,
    _assert_secret_denied,
    _assert_socket_denied,
    _copy_regular_nofollow,
    _drop_to_nobody,
    _set_no_new_privs,
    install_network_deny_seccomp,
)

SOURCE_REPO = "withmartian/routerbench"
SOURCE_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
OUTPUT_LIMIT_BYTES = 1024**3
DEFAULT_CHILD_MEMORY_BYTES = 8 * 1024**3
DEFAULT_CPU_SECONDS = 900
CHILD_REPORT_LIMIT_BYTES = 64 * 1024

EXPECTED_PICKLES: dict[str, dict[str, Any]] = {
    "0shot": {
        "filename": "routerbench_0shot.pkl",
        "bytes": 99_567_659,
        "sha256": "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d",
    },
    "5shot": {
        "filename": "routerbench_5shot.pkl",
        "bytes": 171_091_324,
        "sha256": "fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_source(path: Path, mode: str) -> dict[str, Any]:
    spec = EXPECTED_PICKLES[mode]
    if not path.is_file() or path.is_symlink():
        raise ValueError("routerbench_source_missing_or_symlink")
    size = path.stat().st_size
    if size != spec["bytes"]:
        raise ValueError("routerbench_source_size_mismatch")
    digest = sha256_file(path)
    if digest != spec["sha256"]:
        raise ValueError("routerbench_source_sha256_mismatch")
    return {"path": str(path), "bytes": size, "sha256": digest}


def _json_kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def summarize_jsonl(path: Path, *, max_bytes: int = OUTPUT_LIMIT_BYTES) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise IsolationError("plain output is not a regular file")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise IsolationError("plain output size is outside expected bounds")
    row_hash = hashlib.sha256()
    column_counts: dict[str, Counter[str]] = defaultdict(Counter)
    row_count = 0
    with path.open("rb") as binary:
        for raw_line in binary:
            row_hash.update(raw_line)
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise IsolationError("plain output is not valid JSONL") from exc
            if not isinstance(row, dict):
                raise IsolationError("plain output row is not a JSON object")
            row_count += 1
            for key, value in row.items():
                if not isinstance(key, str):
                    raise IsolationError("plain output column name is not a string")
                column_counts[key][_json_kind(value)] += 1
    if row_count <= 0:
        raise IsolationError("plain output has no rows")
    columns = [
        {
            "name_sha256": sha256_text(name),
            "observed_json_types": dict(sorted(counter.items())),
            "null_count": counter.get("null", 0),
        }
        for name, counter in sorted(column_counts.items())
    ]
    schema = {
        "format": "jsonl_records",
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
    }
    return {
        "bytes": size,
        "sha256": sha256_file(path),
        "row_count": row_count,
        "schema": schema,
        "schema_sha256": hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "row_jsonl_sha256": row_hash.hexdigest(),
        "content_redaction": "report stores schema/count/hash only; original rows are not embedded",
    }


def _set_child_limits(max_output_bytes: int, cpu_seconds: int, memory_bytes: int) -> dict[str, Any]:
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_output_bytes, max_output_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 30))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except (OSError, ValueError):
        pass
    return {
        "fsize_bytes": max_output_bytes,
        "nofile": 64,
        "cpu_seconds": cpu_seconds,
        "nproc": 32,
        "address_space_bytes": None,
    }


def _set_address_space_limit(memory_bytes: int) -> int | None:
    if memory_bytes > 0:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        return memory_bytes
    return None


def _preload_reader_modules() -> dict[str, str]:
    import importlib.metadata

    for module in (
        "pandas",
        "pandas.io.pickle",
        "numpy",
        "numpy.core",
        "numpy.core.multiarray",
        "numpy.core.numeric",
        "numpy.core.fromnumeric",
        "numpy.core._multiarray_umath",
        "numpy._core",
        "numpy._core.multiarray",
        "pyarrow",
    ):
        try:
            __import__(module)
        except ModuleNotFoundError:
            if module.startswith("numpy._core"):
                continue
            raise
    return {module: importlib.metadata.version(module) for module in ("pandas", "numpy", "pyarrow")}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def copy_source_to_jail(source: Path, destination: Path, mode: str) -> dict[str, Any]:
    spec = EXPECTED_PICKLES[mode]
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise ValueError("routerbench_source_missing_or_symlink") from exc
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("routerbench_source_not_regular")
        if status.st_size != spec["bytes"]:
            raise ValueError("routerbench_source_size_mismatch")
        digest = hashlib.sha256()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(fd, "rb", closefd=False) as src, destination.open("xb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                digest.update(chunk)
                dst.write(chunk)
        actual = digest.hexdigest()
        if actual != spec["sha256"]:
            destination.unlink(missing_ok=True)
            raise ValueError("routerbench_source_sha256_mismatch")
        if sha256_file(destination) != actual:
            destination.unlink(missing_ok=True)
            raise ValueError("routerbench_jail_source_sha256_mismatch")
        return {"path": str(source), "bytes": status.st_size, "sha256": actual}
    finally:
        os.close(fd)


def validate_child_report(report: Any, *, mode: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise IsolationError("routerbench child report is invalid")
    allowed = {
        "mode",
        "output_name",
        "output_bytes",
        "output_sha256",
        "roundtrip",
        "versions",
        "isolation",
    }
    if set(report) != allowed:
        raise IsolationError("routerbench child report has unexpected keys")
    if report["mode"] != mode or report["output_name"] != f"routerbench_{mode}.jsonl":
        raise IsolationError("routerbench child report identity mismatch")
    if not isinstance(report["output_bytes"], int) or report["output_bytes"] <= 0:
        raise IsolationError("routerbench child report size invalid")
    if not _is_sha256(report["output_sha256"]):
        raise IsolationError("routerbench child report hash invalid")
    roundtrip = report["roundtrip"]
    if not isinstance(roundtrip, dict) or set(roundtrip) != {
        "row_count",
        "schema_sha256",
        "row_jsonl_sha256",
        "source_jsonsafe_sha256",
    }:
        raise IsolationError("routerbench child report roundtrip invalid")
    if not isinstance(roundtrip["row_count"], int) or roundtrip["row_count"] <= 0:
        raise IsolationError("routerbench child report row count invalid")
    if (
        not _is_sha256(roundtrip["schema_sha256"])
        or not _is_sha256(roundtrip["row_jsonl_sha256"])
        or not _is_sha256(roundtrip["source_jsonsafe_sha256"])
    ):
        raise IsolationError("routerbench child report roundtrip hash invalid")
    if roundtrip["source_jsonsafe_sha256"] != roundtrip["row_jsonl_sha256"]:
        raise IsolationError("routerbench child report roundtrip digest mismatch")
    versions = report["versions"]
    if not isinstance(versions, dict) or set(versions) != {"pandas", "numpy", "pyarrow"}:
        raise IsolationError("routerbench child report versions invalid")
    clean_versions = {}
    for key, value in versions.items():
        if not isinstance(value, str) or not value or len(value) > 64:
            raise IsolationError("routerbench child report version invalid")
        clean_versions[key] = value
    isolation = report["isolation"]
    expected_iso = {
        "mode",
        "filesystem_boundary",
        "uid",
        "gid",
        "network_denied",
        "host_canary_denied",
        "no_new_privs",
        "stdin",
        "close_fds",
        "rlimits",
    }
    if not isinstance(isolation, dict) or set(isolation) != expected_iso:
        raise IsolationError("routerbench child report isolation invalid")
    if (
        isolation["mode"] != "chroot+dropuid+seccomp"
        or isolation["filesystem_boundary"] != "chroot"
    ):
        raise IsolationError("routerbench child report isolation mode invalid")
    for key in ("network_denied", "host_canary_denied", "no_new_privs", "close_fds"):
        if isolation[key] is not True:
            raise IsolationError("routerbench child report isolation proof invalid")
    if isolation["stdin"] != "DEVNULL":
        raise IsolationError("routerbench child report stdin invalid")
    if not isinstance(isolation["uid"], int) or not isinstance(isolation["gid"], int):
        raise IsolationError("routerbench child report identity invalid")
    rlimits = isolation["rlimits"]
    expected_limits = {"fsize_bytes", "nofile", "cpu_seconds", "nproc", "address_space_bytes"}
    if not isinstance(rlimits, dict) or set(rlimits) != expected_limits:
        raise IsolationError("routerbench child report rlimits invalid")
    for _key, value in rlimits.items():
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise IsolationError("routerbench child report rlimit invalid")
    return {
        "mode": report["mode"],
        "output_name": report["output_name"],
        "output_bytes": report["output_bytes"],
        "output_sha256": report["output_sha256"],
        "roundtrip": roundtrip,
        "versions": clean_versions,
        "isolation": {**isolation, "source": "child_reported_allowlisted"},
    }


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def write_jsonl_stream(frame: Any, output: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    if hasattr(frame, "columns") and hasattr(frame, "itertuples"):
        columns = [str(column) for column in frame.columns]
        rows = (
            {column: json_safe(value) for column, value in zip(columns, values, strict=True)}
            for values in frame.itertuples(index=False, name=None)
        )
    else:
        raw_rows = frame if isinstance(frame, list) else [frame]
        rows = (json_safe(row) for row in raw_rows)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            encoded = line.encode("utf-8")
            digest.update(encoded)
            stream.write(line)
            count += 1
    return {"row_count": count, "source_jsonsafe_sha256": digest.hexdigest()}


def read_memory_probe() -> dict[str, Any]:
    root = Path("/sys/fs/cgroup")
    probe: dict[str, Any] = {}
    current = root / "memory.current"
    maximum = root / "memory.max"
    events = root / "memory.events"
    stat = root / "memory.stat"
    if current.is_file():
        probe["memory_current_bytes"] = int(current.read_text().strip())
    if maximum.is_file():
        value = maximum.read_text().strip()
        probe["memory_max_bytes"] = None if value == "max" else int(value)
    if events.is_file():
        probe["memory_events"] = _read_cgroup_kv(events)
    if stat.is_file():
        values = _read_cgroup_kv(stat)
        probe["memory_stat"] = {key: values[key] for key in ("anon", "file") if key in values}
    return probe


def _read_cgroup_kv(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return values


def _read_process_memory(pid: int) -> dict[str, int]:
    status = Path(f"/proc/{pid}/status")
    values: dict[str, int] = {}
    if not status.is_file():
        return values
    for line in status.read_text(errors="ignore").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmPeak:")):
            key, rest = line.split(":", 1)
            amount = rest.strip().split()[0]
            try:
                values[f"{key.lower()}_bytes"] = int(amount) * 1024
            except ValueError:
                pass
    return values


def child_convert(
    *,
    jail_root: Path,
    host_canary: str,
    mode: str,
    max_output_bytes: int,
    cpu_seconds: int,
    memory_bytes: int,
    child_report: str,
) -> dict[str, Any]:
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise IsolationError("routerbench conversion requires Linux root chroot support")
    versions = _preload_reader_modules()
    limits = _set_child_limits(max_output_bytes, cpu_seconds, memory_bytes)
    if not jail_root.is_dir():
        raise IsolationError("missing conversion jail")
    install_network_deny_seccomp()
    os.chdir(jail_root)
    os.chroot(jail_root)
    os.chdir("/")
    _set_no_new_privs()
    _drop_to_nobody()
    secret_denied = _assert_secret_denied(host_canary)
    socket_denied = _assert_socket_denied()
    limits["address_space_bytes"] = _set_address_space_limit(memory_bytes)

    pd = cast(Any, __import__("pandas"))

    source = Path("/input/input.pkl")
    output = Path(f"/output/routerbench_{mode}.jsonl")
    frame = pd.read_pickle(source)
    if not hasattr(frame, "itertuples"):
        frame = pd.DataFrame(frame)
    source_jsonsafe = write_jsonl_stream(frame, output)
    if output.stat().st_size <= 0 or output.stat().st_size > max_output_bytes:
        raise IsolationError("routerbench converted output size is outside expected bounds")
    roundtrip = summarize_jsonl(output, max_bytes=max_output_bytes)
    if source_jsonsafe["row_count"] != roundtrip["row_count"]:
        raise IsolationError("routerbench jsonsafe row count mismatch")
    if source_jsonsafe["source_jsonsafe_sha256"] != roundtrip["row_jsonl_sha256"]:
        raise IsolationError("routerbench jsonsafe digest mismatch")
    return {
        "mode": mode,
        "output_name": output.name,
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "roundtrip": {
            "row_count": roundtrip["row_count"],
            "schema_sha256": roundtrip["schema_sha256"],
            "row_jsonl_sha256": roundtrip["row_jsonl_sha256"],
            "source_jsonsafe_sha256": source_jsonsafe["source_jsonsafe_sha256"],
        },
        "versions": versions,
        "isolation": {
            "mode": "chroot+dropuid+seccomp",
            "filesystem_boundary": "chroot",
            "uid": os.getuid(),
            "gid": os.getgid(),
            "network_denied": socket_denied,
            "host_canary_denied": secret_denied,
            "no_new_privs": True,
            "stdin": "DEVNULL",
            "close_fds": True,
            "rlimits": limits,
        },
    }


def run_child_process(
    *,
    reader_python: Path,
    jail: Path,
    host_canary: Path,
    mode: str,
    max_output_bytes: int,
    cpu_seconds: int,
    memory_bytes: int,
) -> dict[str, Any]:
    command = [
        str(reader_python),
        "-m",
        "scripts.prepare_routerbench",
        "_child-convert",
        "--jail-root",
        str(jail),
        "--host-canary",
        str(host_canary),
        "--mode",
        mode,
        "--max-output-bytes",
        str(max_output_bytes),
        "--cpu-seconds",
        str(cpu_seconds),
        "--memory-bytes",
        str(memory_bytes),
        "--child-report",
        "/output/child-report.json",
    ]
    env = {
        "PATH": f"{reader_python.parent}:/usr/bin:/bin",
        "PYTHONPATH": str(Path.cwd()),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    start_time = datetime.now(UTC).timestamp()
    probe: dict[str, Any] = {"before": read_memory_probe()}
    timeout_seconds = max(ISOLATED_CONVERSION_TIMEOUT_SECONDS, cpu_seconds + 60)
    with subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        env=env,
    ) as child:
        peak: dict[str, int] = {}
        try:
            while True:
                code = child.poll()
                current = _read_process_memory(child.pid)
                for key, value in current.items():
                    peak[key] = max(peak.get(key, 0), value)
                if code is not None:
                    break
                try:
                    child.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
                if datetime.now(UTC).timestamp() - start_time > timeout_seconds:
                    child.kill()
                    child.wait()
                    raise IsolationError("routerbench conversion child timed out")
        finally:
            probe["after"] = read_memory_probe()
            probe["child_peak"] = peak
        if child.returncode != 0:
            raise IsolationError(
                "routerbench conversion child failed "
                f"category=child_nonzero rc={child.returncode}; "
                f"resource_probe={json.dumps(probe, sort_keys=True)}"
            )
    return {"resource_probe": probe}


def isolated_convert_one(
    *,
    source: Path,
    output_dir: Path,
    mode: str,
    reader_python: Path,
    max_output_bytes: int = OUTPUT_LIMIT_BYTES,
    cpu_seconds: int = DEFAULT_CPU_SECONDS,
    memory_bytes: int = DEFAULT_CHILD_MEMORY_BYTES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"routerbench_{mode}.jsonl"
    if output.exists() or output.is_symlink():
        raise FileExistsError("routerbench_output_refuses_overwrite")
    if not reader_python.is_file():
        raise FileNotFoundError("reader_python_missing")
    with tempfile.TemporaryDirectory(prefix="routerbench-jail-") as temporary:
        jail = Path(temporary)
        input_dir = jail / "input"
        child_output_dir = jail / "output"
        jail.chmod(0o755)
        input_dir.mkdir()
        child_output_dir.mkdir()
        input_dir.chmod(0o755)
        child_output_dir.chmod(0o777)
        source_evidence = copy_source_to_jail(source, input_dir / "input.pkl", mode)
        host_canary = jail.parent / f"{jail.name}.host-canary"
        host_canary.write_text("routerbench-secret-canary")
        staging = output_dir / f".routerbench_{mode}.stage-{os.getpid()}"
        staging_report = output_dir / f".routerbench_{mode}.child-report.stage-{os.getpid()}"
        try:
            child_probe = run_child_process(
                reader_python=reader_python,
                jail=jail,
                host_canary=host_canary,
                mode=mode,
                max_output_bytes=max_output_bytes,
                cpu_seconds=cpu_seconds,
                memory_bytes=memory_bytes,
            )
            child_output = child_output_dir / f"routerbench_{mode}.jsonl"
            child_report_path = child_output_dir / "child-report.json"
            _copy_regular_nofollow(child_output, staging, max_bytes=max_output_bytes)
            _copy_regular_nofollow(
                child_report_path, staging_report, max_bytes=CHILD_REPORT_LIMIT_BYTES
            )
            child_report_data = json.loads(staging_report.read_text(encoding="utf-8"))
            child = validate_child_report(child_report_data, mode=mode)
            child.update(child_probe)
            summary = summarize_jsonl(staging, max_bytes=max_output_bytes)
            if child.get("output_sha256") != summary["sha256"]:
                raise IsolationError("copied routerbench output hash mismatch")
            if child["roundtrip"]["row_jsonl_sha256"] != summary["row_jsonl_sha256"]:
                raise IsolationError("copied routerbench output row digest mismatch")
            output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, output)
        except Exception:
            staging.unlink(missing_ok=True)
            staging_report.unlink(missing_ok=True)
            raise
        finally:
            host_canary.unlink(missing_ok=True)
            staging_report.unlink(missing_ok=True)
    return {
        "mode": mode,
        "source": source_evidence,
        "output_path": str(output),
        "plain_output": summary,
        "child": child,
    }


def convert_all(
    *,
    input_root: Path,
    output_dir: Path,
    reader_python: Path,
    modes: list[str],
    max_output_bytes: int,
    cpu_seconds: int,
    memory_bytes: int,
) -> dict[str, Any]:
    converted = []
    for mode in modes:
        filename = str(EXPECTED_PICKLES[mode]["filename"])
        converted.append(
            isolated_convert_one(
                source=input_root / filename,
                output_dir=output_dir,
                mode=mode,
                reader_python=reader_python,
                max_output_bytes=max_output_bytes,
                cpu_seconds=cpu_seconds,
                memory_bytes=memory_bytes,
            )
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "routerbench_plain_conversion_ready",
        "exit_code": 0,
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "reader_python": str(reader_python),
        "converted": converted,
        "training_admitted": False,
        "public_evaluation_only": True,
        "private_holdout_assigned": False,
        "sealed_used": False,
        "gpu_seconds": 0,
        "credentials_read": False,
    }


def probe_isolation() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "status": "blocked",
    }
    try:
        install_network_deny_seccomp()
        _assert_socket_denied()
        report.update(status="ready", network_denied=True)
    except Exception as exc:
        report.update(blocker=f"{type(exc).__name__}: {exc}", network_denied=False)
    return report


def _mode_list(value: str) -> list[str]:
    if value == "all":
        return ["0shot", "5shot"]
    if value not in EXPECTED_PICKLES:
        raise argparse.ArgumentTypeError("mode must be 0shot, 5shot, or all")
    return [value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert")
    convert.add_argument("--input-root", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument("--report", type=Path, required=True)
    convert.add_argument("--reader-python", type=Path, default=Path(sys.executable))
    convert.add_argument("--mode", type=_mode_list, default=["0shot", "5shot"])
    convert.add_argument("--max-output-bytes", type=int, default=OUTPUT_LIMIT_BYTES)
    convert.add_argument("--cpu-seconds", type=int, default=DEFAULT_CPU_SECONDS)
    convert.add_argument("--memory-bytes", type=int, default=DEFAULT_CHILD_MEMORY_BYTES)

    probe = subparsers.add_parser("probe-isolation")
    probe.add_argument("--report", type=Path, required=True)

    child = subparsers.add_parser("_child-convert")
    child.add_argument("--jail-root", type=Path, required=True)
    child.add_argument("--host-canary", required=True)
    child.add_argument("--mode", choices=sorted(EXPECTED_PICKLES), required=True)
    child.add_argument("--max-output-bytes", type=int, default=OUTPUT_LIMIT_BYTES)
    child.add_argument("--cpu-seconds", type=int, default=DEFAULT_CPU_SECONDS)
    child.add_argument("--memory-bytes", type=int, default=DEFAULT_CHILD_MEMORY_BYTES)
    child.add_argument("--child-report", default="/output/child-report.json")

    args = parser.parse_args(argv)
    if args.command == "_child-convert":
        try:
            report = child_convert(
                jail_root=args.jail_root,
                host_canary=args.host_canary,
                mode=args.mode,
                max_output_bytes=args.max_output_bytes,
                cpu_seconds=args.cpu_seconds,
                memory_bytes=args.memory_bytes,
                child_report=args.child_report,
            )
            Path(args.child_report).write_text(
                json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
            )
            return 0
        except Exception as exc:
            print(
                json.dumps({"status": "blocked", "reason": type(exc).__name__}),
                file=sys.stderr,
            )
            return 1

    if args.command == "probe-isolation":
        result = probe_isolation()
        write_json(args.report, result)
        print(json.dumps({"status": result["status"]}))
        return 0 if result["status"] == "ready" else 1

    try:
        result = convert_all(
            input_root=args.input_root,
            output_dir=args.output_dir,
            reader_python=args.reader_python,
            modes=args.mode,
            max_output_bytes=args.max_output_bytes,
            cpu_seconds=args.cpu_seconds,
            memory_bytes=args.memory_bytes,
        )
    except Exception as exc:
        result = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "blocked",
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "reason": (
                str(exc)
                if isinstance(exc, (ValueError, FileNotFoundError, FileExistsError, IsolationError))
                else "routerbench_conversion_failed"
            ),
            "training_admitted": False,
            "public_evaluation_only": True,
            "sealed_used": False,
            "gpu_seconds": 0,
            "credentials_read": False,
        }
    write_json(args.report, result)
    print(json.dumps({"status": result["status"], "exit_code": result["exit_code"]}))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
