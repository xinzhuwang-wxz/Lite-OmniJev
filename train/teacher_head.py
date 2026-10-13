"""Audited conversion for the Jev-Omni tensor-only decision head."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import json
import os
import platform
import pwd
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from train.checkpoint import file_sha256

SCHEMA_VERSION = 1
ALLOWED_SOURCE_REPO = "akhilaaa3/Jev-Omni"
PINNED_SOURCE_REVISION = "5addda86ddee081a68fb067477ea100c221b8917"
FORBIDDEN_SOURCE_NAMES = {"JEV-27B", "JEV-27B-VL"}
SAFETENSORS_HEADER_RESERVE_BYTES = 64 * 1024
ISOLATED_CONVERSION_TIMEOUT_SECONDS = 120


def tensor_bytes_sha256(tensor: Any) -> str:
    torch = importlib.import_module("torch")
    value = tensor.detach().cpu().contiguous()
    if value.numel() == 0:
        payload = b""
    else:
        payload = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _module_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "safetensors"):
        try:
            module = importlib.import_module(name)
        except Exception:
            versions[name] = "unavailable"
        else:
            versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def validate_source(repo: str, revision: str) -> None:
    if repo != ALLOWED_SOURCE_REPO or revision != PINNED_SOURCE_REVISION:
        raise ValueError("teacher head source is not the pinned Jev-Omni revision")
    if repo.rsplit("/", 1)[-1].upper() in FORBIDDEN_SOURCE_NAMES:
        raise ValueError("prohibited teacher source")


def load_expected_spec(path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    value = json.loads(path.read_text())
    tensors = value.get("tensors") if isinstance(value, dict) else None
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("expected spec must contain a non-empty tensors object")
    expected: dict[str, dict[str, Any]] = {}
    for key, item in tensors.items():
        if not isinstance(key, str) or not isinstance(item, dict):
            raise ValueError("expected tensor spec must map string keys to objects")
        shape = item.get("shape")
        dtype = item.get("dtype")
        valid_shape = isinstance(shape, list) and all(
            isinstance(dim, int) and dim >= 0 for dim in shape
        )
        if not valid_shape:
            raise ValueError(f"invalid expected tensor shape: {key}")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"invalid expected tensor dtype: {key}")
        expected[key] = {"shape": shape, "dtype": dtype}
    return expected


def expected_tensor_payload_bytes(expected: dict[str, dict[str, Any]]) -> int:
    dtype_sizes = {
        "torch.float64": 8,
        "torch.float32": 4,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.int64": 8,
        "torch.int32": 4,
        "torch.int16": 2,
        "torch.int8": 1,
        "torch.uint8": 1,
        "torch.bool": 1,
    }
    total = 0
    for key, item in expected.items():
        dtype = str(item["dtype"])
        if dtype not in dtype_sizes:
            raise ValueError(f"unsupported expected tensor dtype: {key}")
        numel = 1
        for dim in item["shape"]:
            numel *= int(dim)
        total += numel * dtype_sizes[dtype]
    return total


def _extract_tensor_mapping(value: Any) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    if not isinstance(value, Mapping):
        raise ValueError("teacher head must load to a tensor mapping")
    tensors: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("teacher head contains a non-string tensor key")
        if not isinstance(item, torch.Tensor):
            raise ValueError(f"teacher head contains a non-tensor value: {key}")
        tensors[key] = item.detach().cpu().contiguous()
    if not tensors:
        raise ValueError("teacher head contains no tensors")
    return tensors


def _audit_tensors(
    tensors: dict[str, Any], expected: dict[str, dict[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    torch = importlib.import_module("torch")
    if expected is not None and set(tensors) != set(expected):
        missing = sorted(set(expected) - set(tensors))
        extra = sorted(set(tensors) - set(expected))
        raise ValueError(f"teacher head tensor keys mismatch: missing={missing} extra={extra}")
    report: dict[str, dict[str, Any]] = {}
    for key in sorted(tensors):
        tensor = tensors[key]
        if tensor.layout != torch.strided:
            raise ValueError(f"teacher head tensor layout is unsupported: {key}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"teacher head tensor has nonfinite values: {key}")
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        if expected is not None:
            spec = expected[key]
            if shape != spec["shape"] or dtype != spec["dtype"]:
                raise ValueError(f"teacher head tensor spec mismatch: {key}")
        report[key] = {
            "shape": shape,
            "dtype": dtype,
            "numel": int(tensor.numel()),
            "sha256": tensor_bytes_sha256(tensor),
        }
    return report


def _load_pt_weights_only(path: Path) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ValueError("torch.load weights_only=True is unavailable") from exc
    except Exception as exc:
        raise ValueError("teacher head weights_only load failed") from exc
    return _extract_tensor_mapping(loaded)


def convert_head(
    source: Path,
    output: Path,
    *,
    expected_sha256: str,
    expected_size: int | None = None,
    expected_spec: dict[str, dict[str, Any]] | None = None,
    source_repo: str = ALLOWED_SOURCE_REPO,
    source_revision: str = PINNED_SOURCE_REVISION,
) -> dict[str, Any]:
    validate_source(source_repo, source_revision)
    if source.suffix != ".pt":
        raise ValueError("teacher head source must be a .pt file")
    if output.suffix != ".safetensors":
        raise ValueError("teacher head output must be a .safetensors file")
    if not source.is_file():
        raise ValueError(f"missing teacher head: {source}")
    if expected_size is not None and source.stat().st_size != expected_size:
        raise ValueError("teacher head size mismatch")
    actual_sha256 = file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError("teacher head SHA256 mismatch")
    tensors = _load_pt_weights_only(source)
    tensor_report = _audit_tensors(tensors, expected_spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    safetensors = importlib.import_module("safetensors.torch")
    safetensors.save_file(
        tensors,
        str(temporary),
        metadata={
            "source_repo": source_repo,
            "source_revision": source_revision,
            "source_sha256": actual_sha256,
            "schema_version": str(SCHEMA_VERSION),
        },
    )
    converted = safetensors.load_file(str(temporary), device="cpu")
    converted_report = _audit_tensors(converted, expected_spec)
    if converted_report != tensor_report:
        temporary.unlink(missing_ok=True)
        raise ValueError("teacher head safetensors round-trip metadata mismatch")
    for key, tensor in tensors.items():
        if not bool(importlib.import_module("torch").equal(tensor, converted[key])):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"teacher head tensor round-trip mismatch: {key}")
    temporary.replace(output)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "converted",
        "converted_at": datetime.now(UTC).isoformat(),
        "source_repo": source_repo,
        "source_revision": source_revision,
        "source_path": str(source),
        "source_size": source.stat().st_size,
        "source_sha256": actual_sha256,
        "output_path": str(output),
        "output_size": output.stat().st_size,
        "output_sha256": file_sha256(output),
        "tensor_count": len(tensor_report),
        "tensors": tensor_report,
        "versions": _module_versions(),
    }


class SeccompError(RuntimeError):
    """Raised when the platform cannot install the network-deny boundary."""


class IsolationError(RuntimeError):
    """Raised when conversion cannot run inside the no-secrets boundary."""


def install_network_deny_seccomp() -> None:
    if platform.system() != "Linux":
        raise SeccompError("seccomp network-deny boundary requires Linux")
    try:
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as exc:
        raise SeccompError("libseccomp.so.2 is unavailable") from exc

    scmp_act_allow = 0x7FFF0000
    scmp_act_errno = 0x00050000 | errno.EPERM
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int

    ctx = lib.seccomp_init(scmp_act_allow)
    if not ctx:
        raise SeccompError("seccomp_init failed")
    try:
        for name in (
            "socket",
            "socketpair",
            "connect",
            "bind",
            "listen",
            "accept",
            "accept4",
            "sendto",
            "recvfrom",
            "sendmsg",
            "recvmsg",
            "getsockopt",
            "setsockopt",
        ):
            syscall = int(lib.seccomp_syscall_resolve_name(name.encode()))
            if syscall < 0:
                continue
            result = int(lib.seccomp_rule_add(ctx, scmp_act_errno, syscall, 0))
            if result < 0:
                raise SeccompError(f"seccomp_rule_add failed for {name}: {result}")
        result = int(lib.seccomp_load(ctx))
        if result < 0:
            raise SeccompError(f"seccomp_load failed: {result}")
    finally:
        lib.seccomp_release(ctx)


def _scrubbed_env() -> dict[str, str]:
    keep = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "PYTHONHOME",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "UV_PROJECT_ENVIRONMENT",
        }
    }
    keep.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return keep


def _nobody_ids() -> tuple[int, int]:
    try:
        entry = pwd.getpwnam("nobody")
    except KeyError:
        return 65534, 65534
    return int(entry.pw_uid), int(entry.pw_gid)


def _drop_to_nobody() -> None:
    uid, gid = _nobody_ids()
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    if os.getuid() == 0 or os.geteuid() == 0:
        raise IsolationError("failed to drop root privileges")


def _set_no_new_privs() -> None:
    if platform.system() != "Linux":
        raise IsolationError("no_new_privs requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_no_new_privs = 38
    result = libc.prctl(pr_set_no_new_privs, 1, 0, 0, 0)
    if result != 0:
        error = ctypes.get_errno()
        raise IsolationError(f"prctl no_new_privs failed: {error}")


def _assert_secret_denied(host_canary: str) -> bool:
    try:
        Path(host_canary).read_bytes()
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        return True
    raise IsolationError("host canary remained readable after chroot")


def _assert_socket_denied() -> bool:
    import socket

    try:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        return True
    raise IsolationError("socket syscall was not denied")


def _copy_regular_nofollow(source: Path, destination: Path, *, max_bytes: int) -> None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise IsolationError("isolated conversion output is not a regular file") from exc
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode):
            raise IsolationError("isolated conversion output is not a regular file")
        if status.st_size <= 0 or status.st_size > max_bytes:
            raise IsolationError("isolated conversion output size is outside expected bounds")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        digest = hashlib.sha256()
        copied = 0
        with os.fdopen(fd, "rb", closefd=False) as src, temporary.open("xb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > max_bytes:
                    raise IsolationError("isolated conversion output exceeded expected bounds")
                digest.update(chunk)
                dst.write(chunk)
        if copied != status.st_size:
            temporary.unlink(missing_ok=True)
            raise IsolationError("isolated conversion output changed while copying")
        temporary.replace(destination)
    except Exception:
        destination.with_suffix(destination.suffix + ".tmp").unlink(missing_ok=True)
        raise
    finally:
        os.close(fd)


def child_chrooted_convert(argv: list[str], *, jail_root: Path, host_canary: str) -> dict[str, Any]:
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise IsolationError("chroot conversion requires Linux root")
    importlib.import_module("torch")
    importlib.import_module("torch.utils.serialization")
    importlib.import_module("numpy.ctypeslib")
    importlib.import_module("safetensors.torch")
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
    report = convert_from_argv(argv)
    report["isolation"] = {
        "mode": "chroot+dropuid+seccomp",
        "filesystem_boundary": "chroot",
        "uid": os.getuid(),
        "gid": os.getgid(),
        "network_denied": socket_denied,
        "host_canary_denied": secret_denied,
        "no_new_privs": True,
    }
    report["expected_spec_provided"] = True
    return report


def _jail_conversion_args(args: list[str]) -> tuple[list[str], Path, Path, Path]:
    parsed = parse_conversion_args(args)
    if parsed["expected_spec"] is None:
        raise IsolationError("isolated conversion requires a preapproved expected spec")
    source = cast(Path, parsed["source"])
    output = cast(Path, parsed["output"])
    expected_spec_path: Path | None = None
    for index, item in enumerate(args):
        if item == "--expected-spec":
            expected_spec_path = Path(args[index + 1])
            break
    if expected_spec_path is None:
        raise IsolationError("isolated conversion requires a preapproved expected spec")
    jail_args = args.copy()
    replacements = {
        "--source": "/input/head.pt",
        "--output": "/output/head.safetensors",
        "--expected-spec": "/input/expected-spec.json",
    }
    for index, item in enumerate(jail_args):
        if item in replacements:
            jail_args[index + 1] = replacements[item]
    return jail_args, source, output, expected_spec_path


def isolated_convert(args: list[str]) -> dict[str, Any]:
    jail_args, source, output, expected_spec_path = _jail_conversion_args(args)
    if platform.system() != "Linux" or os.geteuid() != 0:
        raise IsolationError("isolated conversion requires Linux root chroot support")
    expected_spec = load_expected_spec(expected_spec_path)
    if expected_spec is None:
        raise IsolationError("isolated conversion requires a preapproved expected spec")
    max_output_bytes = (
        expected_tensor_payload_bytes(expected_spec) + SAFETENSORS_HEADER_RESERVE_BYTES
    )
    spec_sha256 = file_sha256(expected_spec_path)
    source_evidence: dict[str, Any] = {"expected_spec_sha256": spec_sha256}
    for name in ("decision_config.json", "jev_omni.py"):
        candidate = source.parent / name
        if candidate.is_file():
            source_evidence[f"{name}_sha256"] = file_sha256(candidate)
    with tempfile.TemporaryDirectory(prefix="teacher-head-jail-") as temporary:
        jail = Path(temporary)
        input_dir = jail / "input"
        output_dir = jail / "output"
        jail.chmod(0o755)
        input_dir.mkdir()
        output_dir.mkdir()
        input_dir.chmod(0o755)
        output_dir.chmod(0o777)
        shutil.copy2(source, input_dir / "head.pt")
        shutil.copy2(expected_spec_path, input_dir / "expected-spec.json")
        host_canary = jail.parent / f"{jail.name}.host-canary"
        host_canary.write_text("secret-canary")
        command = [
            sys.executable,
            "-m",
            "scripts.teacher_head",
            "_child-convert",
            "--jail-root",
            str(jail),
            "--host-canary",
            str(host_canary),
            *jail_args,
        ]
        try:
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                    env=_scrubbed_env(),
                    timeout=ISOLATED_CONVERSION_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise IsolationError("isolated conversion child timed out") from exc
            if completed.returncode != 0:
                raise IsolationError("isolated conversion child failed")
            report = dict(json.loads(completed.stdout))
            converted = output_dir / "head.safetensors"
            _copy_regular_nofollow(converted, output, max_bytes=max_output_bytes)
            if file_sha256(output) != report.get("output_sha256"):
                output.unlink(missing_ok=True)
                raise IsolationError("copied safetensors hash mismatch")
            report["output_path"] = str(output)
            report["output_size"] = output.stat().st_size
            report["output_sha256"] = file_sha256(output)
            report.update(source_evidence)
            return report
        finally:
            host_canary.unlink(missing_ok=True)


def probe_isolation() -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "setpriv": shutil.which("setpriv"),
        "libseccomp": "unknown",
        "network_denied": False,
        "status": "blocked",
    }
    try:
        install_network_deny_seccomp()
        _assert_socket_denied()
        report.update(libseccomp="available", network_denied=True, status="ready")
    except Exception as exc:
        report.update(libseccomp="blocked", blocker=f"{type(exc).__name__}: {exc}")
    return report


def parse_conversion_args(argv: list[str]) -> dict[str, Any]:
    import argparse

    parser = argparse.ArgumentParser(description="Convert Jev-Omni head.pt to safetensors")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int)
    parser.add_argument("--expected-spec", type=Path)
    parser.add_argument("--source-repo", default=ALLOWED_SOURCE_REPO)
    parser.add_argument("--source-revision", default=PINNED_SOURCE_REVISION)
    args = parser.parse_args(argv)
    return {
        "source": args.source,
        "output": args.output,
        "expected_sha256": args.expected_sha256,
        "expected_size": args.expected_size,
        "expected_spec": load_expected_spec(args.expected_spec),
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
    }


def convert_from_argv(argv: list[str]) -> dict[str, Any]:
    kwargs = parse_conversion_args(argv)
    return convert_head(**cast(Any, kwargs))
