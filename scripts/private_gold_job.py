"""Root-supervised fixed inventory audit with an observed ordinary-UID boundary.

No namespace, network, host-visibility or root-security isolation is claimed.
Run trees and logs remain private; stdout contains only fixed failure categories.
"""

from __future__ import annotations

import ctypes
import errno
import grp
import json
import os
import pwd
import resource
import signal
import stat
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field

from data.schema import Contract, Sha256
from eval.gold_gate import IsolationSummary
from scripts.gold_inventory import Manifest, Receipt, audit_manifest, digest, publish

MAX_INPUT = 256 * 1024 * 1024
MAX_MANIFEST = 1024 * 1024
MAX_RECEIPT = 1024 * 1024
Category = Literal[
    "invalid_arguments",
    "boundary_unavailable",
    "input_rejected",
    "probe_failed",
    "timeout",
    "missing_receipt",
    "invalid_receipt",
    "job_failed",
    "report_io",
]


class JobRejected(Exception):
    def __init__(self, category: Category):
        self.category = category


class PublicReceipt(Contract):
    schema_version: Literal["private-gold-job-v1"] = "private-gold-job-v1"
    job: Literal["inventory-audit"] = "inventory-audit"
    outcome: Literal["isolated", "rejected"] = "rejected"
    categories: list[
        Category
        | Literal[
            "invalid_manifest",
            "input_io",
            "hash_mismatch",
            "invalid_sample",
            "invalid_component",
            "pool_mismatch",
            "modality_mismatch",
            "missing_component",
            "conflicting_identity",
            "pool_overlap",
            "invalid_metadata",
        ]
    ]
    source_manifest_sha256: Sha256
    translated_manifest_sha256: Sha256
    code_sha256: Sha256
    config_sha256: Sha256
    nonce_sha256: Sha256
    duration_seconds: float = Field(ge=0)
    exit_code: int
    ordinary_identity_boundary: bool
    namespace_enforced: Literal[False] = False
    network_isolation: Literal["not_enforced"] = "not_enforced"
    root_visible_workflow_boundary: Literal[True] = True
    isolation: IsolationSummary | None = None
    gold_provenance: Literal["not_validated"] = "not_validated"
    inventory_completeness: Literal["not_validated"] = "not_validated"
    lineage_completeness: Literal["not_validated"] = "not_validated"
    external_security_attestation: Literal["not_validated"] = "not_validated"


def safe_path(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise JobRejected("input_rejected")
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            raise JobRejected("input_rejected")


def bounded_read(path: Path, limit: int) -> bytes:
    safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise JobRejected("input_rejected")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            content = handle.read(limit + 1)
        if len(content) > limit:
            raise JobRejected("input_rejected")
        return content
    finally:
        os.close(fd)


def snapshot(source: Path, expected: str, inputs: Path, gid: int) -> tuple[Path, list[Path]]:
    content = bounded_read(source, MAX_MANIFEST)
    if digest(content) != expected:
        raise JobRejected("input_rejected")
    manifest = Manifest.model_validate_json(content)
    originals = [source]
    for index, artifact in enumerate([*manifest.samples, *manifest.component_ledgers]):
        original = Path(artifact.path)
        payload = bounded_read(original, MAX_INPUT)
        if digest(payload) != artifact.sha256:
            raise JobRejected("input_rejected")
        target = inputs / f"artifact-{index:06d}.jsonl"
        target.write_bytes(payload)
        os.chown(target, 0, gid)
        target.chmod(0o440)
        artifact.path = str(target)
        originals.append(original)
    target = inputs / "manifest.json"
    target.write_text(manifest.model_dump_json() + "\n")
    os.chown(target, 0, gid)
    target.chmod(0o440)
    return target, originals


def drop_identity(uid: int, gid: int) -> None:
    os.environ.clear()
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise JobRejected("boundary_unavailable")


def close_fds() -> None:
    # Linux supervisor only; close inherited sockets and source descriptors.
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        if fd > 2:
            try:
                os.close(fd)
            except OSError:
                pass


def denied_probe(uid: int, gid: int, reads: list[Path], writes: list[Path]) -> bool:
    pid = os.fork()
    if pid == 0:
        try:
            close_fds()
            drop_identity(uid, gid)
            os.chdir("/")
            for path, flags in [(p, os.O_RDONLY) for p in reads] + [
                (p, os.O_WRONLY | os.O_CREAT) for p in writes
            ]:
                try:
                    fd = os.open(path, flags | os.O_NOFOLLOW)
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EPERM):
                        os._exit(1)
                else:
                    os.close(fd)
                    os._exit(1)
            os._exit(0)
        except JobRejected:
            os._exit(2)
        except BaseException:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    code = os.waitstatus_to_exitcode(status)
    if code == 2:
        raise JobRejected("boundary_unavailable")
    return code == 0


def wait_child(pid: int, timeout: float) -> tuple[int, bool]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result, status = os.waitpid(pid, os.WNOHANG)
        if result:
            return os.waitstatus_to_exitcode(status), False
        time.sleep(0.02)
    # setsid occurs before private reads. Kill the process as well if setsid failed.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), True


def run_child(manifest: Path, output: Path, uid: int, timeout: int) -> tuple[int, bool]:
    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
            close_fds()
            drop_identity(uid, uid)
            os.chdir(output)
            null = os.open("/dev/null", os.O_RDONLY)
            os.dup2(null, 0)
            os.close(null)
            log = os.open(output / "private.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.dup2(log, 1)
            os.dup2(log, 2)
            os.close(log)
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_RECEIPT, MAX_RECEIPT))
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
            receipt = audit_manifest(manifest)
            publish(receipt, output / "receipt.json")
            os._exit(0 if receipt.outcome == "isolated" else 1)
        except BaseException:
            os._exit(2)
    return wait_child(pid, timeout)


def validate_receipt(path: Path, expected: str, exit_code: int) -> Receipt:
    if not path.exists():
        raise JobRejected("missing_receipt")
    try:
        receipt = Receipt.model_validate_json(bounded_read(path, MAX_RECEIPT))
        if receipt.manifest_sha256 != expected or exit_code != (
            0 if receipt.outcome == "isolated" else 1
        ):
            raise ValueError
        if (receipt.outcome == "isolated") != (not receipt.categories):
            raise ValueError
        return receipt
    except Exception:
        raise JobRejected("invalid_receipt") from None


def unused_uid(uid: int) -> bool:
    if uid <= 0 or uid >= 2**31:
        return False
    try:
        pwd.getpwuid(uid)
    except KeyError:
        pass
    else:
        return False
    try:
        grp.getgrgid(uid)
    except KeyError:
        return True
    return False


def execute(
    source: Path,
    expected: str,
    work_root: Path,
    report: Path,
    worker_uid: int,
    probe_uid: int,
    timeout: int,
) -> PublicReceipt:
    start = time.monotonic()
    if sys.platform != "linux" or os.geteuid() != 0:
        raise JobRejected("boundary_unavailable")
    if not unused_uid(worker_uid) or not unused_uid(probe_uid) or worker_uid == probe_uid:
        raise JobRejected("invalid_arguments")
    if not 1 <= timeout <= 3600:
        raise JobRejected("invalid_arguments")
    for path in (source, work_root, report.parent):
        safe_path(path)
    if not report.is_absolute() or report.exists() or report.is_symlink():
        raise JobRejected("report_io")
    for directory in (work_root, report.parent):
        info = directory.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise JobRejected("boundary_unavailable")
    if not work_root.is_relative_to(Path("/var/tmp")) or work_root == Path("/var/tmp"):
        raise JobRejected("invalid_arguments")
    nonce = uuid.uuid4().hex
    run = work_root / nonce
    run.mkdir(mode=0o700)
    os.chown(run, 0, worker_uid)
    inputs = run / "inputs"
    inputs.mkdir(mode=0o700)
    os.chown(inputs, 0, worker_uid)
    manifest, originals = snapshot(source, expected, inputs, worker_uid)
    translated = digest(manifest.read_bytes())
    inputs.chmod(0o550)
    output = run / "output"
    output.mkdir(mode=0o700)
    os.chown(output, worker_uid, worker_uid)
    export = run / "export-probe"
    export.write_bytes(b"")
    export.chmod(0o440)
    run.chmod(0o750)
    # Explicit source scope: every original manifest and JSONL path must deny reads.
    boundary = denied_probe(probe_uid, probe_uid, [manifest, output], []) and denied_probe(
        worker_uid, worker_uid, originals, [*inputs.iterdir(), export, report.parent / nonce]
    )
    if not boundary:
        raise JobRejected("probe_failed")
    exit_code, timed_out = run_child(manifest, output, worker_uid, timeout)
    category: Category | None = "timeout" if timed_out else None
    receipt = None
    if category is None:
        try:
            receipt = validate_receipt(output / "receipt.json", translated, exit_code)
        except JobRejected as error:
            category = error.category
    code = b"".join(
        Path(p).read_bytes()
        for p in (
            __file__,
            __import__("scripts.gold_inventory", fromlist=["__file__"]).__file__,
            __import__("eval.gold_gate", fromlist=["__file__"]).__file__,
        )
    )
    result = PublicReceipt(
        outcome=receipt.outcome if receipt else "rejected",
        categories=[category] if category else list(receipt.categories if receipt else []),
        source_manifest_sha256=expected,
        translated_manifest_sha256=translated,
        code_sha256=digest(code),
        config_sha256=digest(
            json.dumps(
                {
                    "timeout": timeout,
                    "worker_uid": worker_uid,
                    "probe_uid": probe_uid,
                    "max_input": MAX_INPUT,
                    "job": "inventory-audit",
                },
                sort_keys=True,
            ).encode()
        ),
        nonce_sha256=digest(nonce.encode()),
        duration_seconds=time.monotonic() - start,
        exit_code=exit_code,
        ordinary_identity_boundary=boundary,
        isolation=receipt.isolation if receipt else None,
    )
    publish(result, report)
    return result


def main(argv: list[str] | None = None) -> int:
    from scripts.gold_inventory import PrivateParser

    try:
        parser = PrivateParser(description=__doc__)
        parser.add_argument("--job", choices=["inventory-audit"], required=True)
        parser.add_argument("--inventory", type=Path, required=True)
        parser.add_argument("--sha256", required=True)
        parser.add_argument("--work-root", type=Path, required=True)
        parser.add_argument("--report", type=Path, required=True)
        parser.add_argument("--worker-uid", type=int, required=True)
        parser.add_argument("--probe-uid", type=int, required=True)
        parser.add_argument("--timeout", type=int, default=300)
        args = parser.parse_args(argv)
        result = execute(
            args.inventory,
            args.sha256,
            args.work_root,
            args.report,
            args.worker_uid,
            args.probe_uid,
            args.timeout,
        )
        return 0 if result.outcome == "isolated" else 1
    except JobRejected as error:
        category = error.category
    except Exception:
        category = "job_failed"
    print(json.dumps({"outcome": "rejected", "categories": [category]}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
