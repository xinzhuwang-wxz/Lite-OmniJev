"""Root-only synthetic Linux boundary evidence; never reads private dataset contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts import private_calibration as cal
from scripts import private_gold_job as broker
from scripts import private_stage_eval as stage
from scripts.s0_public_baselines import sha256_file

JOURNAL = Path("/var/tmp/omnijev-root-traverse-acl-recovery.json")
CASES = ("normal", "timeout", "term", "kill", "competition", "stale_recovery")
WORKER = r"""
import json, os, sys, time
from pathlib import Path
from scripts.private_calibration import verify_child_identity
verify_child_identity()
root = Path(sys.argv[1])
assert (root / 'public-model').read_bytes() == b'synthetic public model'
assert (root / 'private' / 'sample').read_bytes() == b'synthetic calibration'
for path, flags in [(Path('/root'), os.O_RDONLY), (root / 'secret', os.O_RDONLY),
                    (root / 'public-model', os.O_WRONLY), (root / 'source', os.O_WRONLY)]:
    try:
        fd = os.open(path, flags | os.O_NOFOLLOW)
    except PermissionError:
        continue
    else:
        os.close(fd)
        raise RuntimeError('denial_probe_failed')
fields = dict(line.split(':', 1)
              for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
print(json.dumps({'uid': os.getuid(), 'gid': os.getgid(), 'groups': os.getgroups(),
                  'no_new_privs': fields['NoNewPrivs'].strip(),
                  'caps': {key: fields[key].strip() for key in ['CapEff','CapPrm','CapAmb']},
                  'synthetic_denials': True}), flush=True)
time.sleep(float(sys.argv[2]))
"""


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def no_processes(uid: int) -> bool:
    try:
        broker.check_process_ids({uid})
        return True
    except broker.JobRejected:
        return False


def wait_until(predicate: Any, seconds: float = 8) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise RuntimeError("bounded_wait_expired")


def harness(args: argparse.Namespace) -> int:
    case_root = args.case_root
    ready = case_root / "ready.json"
    with cal.guarded_acl(args.uid, case_root / "acl-backup.json") as evidence:
        with (case_root / "worker.jsonl").open("xb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    WORKER,
                    str(case_root),
                    "0" if args.case == "normal" else "60",
                ],
                env={
                    **stage.safe_environment(),
                    "PYTHONPATH": str(Path(cal.__file__).resolve().parents[1]),
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                cwd="/",
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                user=args.uid,
                group=args.uid,
                extra_groups=[],
                preexec_fn=cal.restrict_child,
                start_new_session=True,
            )
            wait_until(lambda: (case_root / "worker.jsonl").stat().st_size > 0)
            cal.durable_json(ready, {"worker_pid": child.pid, **evidence})
            if args.case == "timeout":
                try:
                    child.wait(timeout=0.2)
                    raise RuntimeError("timeout_not_observed")
                except subprocess.TimeoutExpired:
                    stage.terminate(child)
            else:
                child.wait(timeout=65)
                if args.case == "normal":
                    cal.require(child.returncode == 0, "worker_failed")
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    cal.require(sys.platform == "linux" and os.geteuid() == 0, "linux_root_required")
    cal.require(
        not JOURNAL.exists() and not JOURNAL.is_symlink(), "existing_job_or_recovery_required"
    )
    root = args.work_root
    broker.safe_path(root)
    root.mkdir(mode=0o755, parents=True, exist_ok=True)
    info = root.stat()
    cal.require(info.st_uid == 0 and not info.st_mode & 0o022, "unsafe_work_root")
    cal.require(root.is_relative_to("/var/tmp") and root != Path("/var/tmp"), "wrong_work_root")
    api = cal.PosixACL()
    before = api.read(Path("/root"))
    cal.require(
        set(before.splitlines()) == {"user::rwx", "group::---", "other::---"}, "unexpected_acl"
    )
    cal.durable_json(root / "outer-original-acl.json", {"acl_before": before})
    result: dict[str, Any] = {
        "schema_version": "calibration-linux-boundary-probe-v1",
        "status": "failed",
        "private_data_read": False,
        "gpu_used": False,
        "before_acl_sha256": digest(before),
        "cases": [],
        "producer_sha256": sha256_file(Path(cal.__file__)),
        "probe_sha256": sha256_file(Path(__file__)),
        "code_sha256": stage.code_hash(),
    }
    used: list[int] = []
    parents: list[subprocess.Popen[Any]] = []
    started = time.monotonic()
    old_alarm = signal.getsignal(signal.SIGALRM)
    old_term = signal.getsignal(signal.SIGTERM)

    def expired(signum: int, frame: Any) -> None:
        raise TimeoutError("probe_hardcap_180_seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.alarm(170)  # Reserve ten seconds for bounded cleanup and the outer 180s cap.
    try:
        for index, case in enumerate(CASES):
            uid = args.first_uid + index * 2
            cal.require(broker.unused_uid(uid) and broker.unused_uid(uid + 1), "uid_occupied")
            broker.reserve_ids(root, uid, uid + 1)
            used.extend([uid, uid + 1])
            case_root = root / case
            case_root.mkdir(mode=0o755)
            for name, content, mode in [
                ("secret", b"synthetic credential", 0o600),
                ("public-model", b"synthetic public model", 0o444),
                ("source", b"synthetic source", 0o444),
            ]:
                target = case_root / name
                target.write_bytes(content)
                target.chmod(mode)
            private = case_root / "private"
            private.mkdir(mode=0o700)
            os.chown(private, uid, uid)
            (private / "sample").write_bytes(b"synthetic calibration")
            (private / "sample").chmod(0o600)
            os.chown(private / "sample", uid, uid)
            cal.require(
                broker.denied_probe(uid + 1, uid + 1, [private / "sample"], []),
                "unrelated_uid_read_private_synthetic",
            )
            if case == "stale_recovery":
                # Simulated crash residue contains only this freshly leased synthetic UID.
                cal.durable_json(JOURNAL, {"acl_before": before, "uid": uid + 1, "parent_pid": 0})
                api.write(Path("/root"), before + f"user:{uid + 1}:--x\nmask::--x\n")
            with (case_root / "harness.log").open("xb") as log:
                parent = subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--harness",
                        "--case",
                        "normal" if case == "stale_recovery" else case,
                        "--case-root",
                        str(case_root),
                        "--uid",
                        str(uid),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                parents.append(parent)
                wait_until(
                    lambda case_root=case_root, parent=parent: (
                        (case_root / "ready.json").exists() or parent.poll() is not None
                    )
                )
                cal.require((case_root / "ready.json").exists(), "harness_failed_before_ready")
                ready = json.loads((case_root / "ready.json").read_text())
                if case == "competition":
                    blocked = False
                    try:
                        with cal.guarded_acl(uid + 1, case_root / "competitor.json"):
                            raise RuntimeError("global_lock_failed")
                    except BlockingIOError:
                        blocked = True
                    cal.require(blocked, "competition_not_blocked")
                if case in ("term", "competition"):
                    parent.send_signal(signal.SIGTERM)
                elif case == "kill":
                    parent.kill()
                parent.wait(timeout=10)
                wait_until(
                    lambda uid=uid: (
                        api.read(Path("/root")) == before
                        and not JOURNAL.exists()
                        and no_processes(uid)
                    )
                )
                identity = json.loads((case_root / "worker.jsonl").read_text().splitlines()[0])
                cal.require(
                    identity["uid"] == uid
                    and identity["gid"] == uid
                    and identity["groups"] == []
                    and identity["no_new_privs"] == "1"
                    and all(int(v, 16) == 0 for v in identity["caps"].values())
                    and identity["synthetic_denials"] is True,
                    "identity_probe_failed",
                )
                if case in ("normal", "timeout", "stale_recovery"):
                    cal.require(parent.returncode == 0, "unexpected_exit")
                result["cases"].append(
                    {
                        "case": case,
                        "passed": True,
                        "worker_identity": identity,
                        "parent_exit_code": parent.returncode,
                        "guardian_pid": ready["guardian_pid"],
                        "before_acl_sha256": digest(before),
                        "after_acl_sha256": digest(api.read(Path("/root"))),
                        "no_lingering_uid_processes": True,
                        "journal_removed": not JOURNAL.exists(),
                    }
                )
        result["status"] = "passed"
        result["exit_code"] = 0
    except BaseException as error:
        result["failure_category"] = type(error).__name__
        result["exit_code"] = 1
    finally:
        signal.alarm(0)
        for parent in parents:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=2)
        for uid in used:
            cal.revoke_uid_processes(uid)
        # Every guardian gets EOF before fallback. Never race an active production lease.
        wait_until(lambda: all(parent.poll() is not None for parent in parents), 2)
        import fcntl

        lock = os.open(
            "/var/tmp/omnijev-root-traverse-acl.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            wait_until(lambda: acquire(lock, fcntl), 5)
            api.write(Path("/root"), before)
            cal.require(api.read(Path("/root")) == before, "outer_restore_failed")
            if JOURNAL.exists():
                residue = json.loads(broker.bounded_read(JOURNAL, 16384))
                cal.require(residue["uid"] in used, "foreign_recovery_journal")
                JOURNAL.unlink()
        finally:
            os.close(lock)
            signal.signal(signal.SIGALRM, old_alarm)
            signal.signal(signal.SIGTERM, old_term)
        result["after_acl_sha256"] = digest(api.read(Path("/root")))
        result["acl_restored"] = result["after_acl_sha256"] == result["before_acl_sha256"]
        result["no_lingering_uid_processes"] = all(no_processes(uid) for uid in used)
        if not result["acl_restored"] or not result["no_lingering_uid_processes"]:
            result.update(status="failed", exit_code=1, failure_category="cleanup_evidence_failed")
        result["wall_seconds"] = time.monotonic() - started
        cal.write_private(args.report, result)
    return result


def acquire(fd: int, fcntl: Any) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--work-root", type=Path, default=Path("/var/tmp/omnijev-calibration-probes-001")
    )
    parser.add_argument("--first-uid", type=int, default=64001)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--harness", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--case-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--uid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.harness:
        return harness(args)
    if not args.execute:
        print(
            json.dumps({"status": "prepared_not_executed", "cases": CASES, "hardcap_seconds": 180})
        )
        return 0
    cal.require(args.report is not None and not args.report.exists(), "new_report_required")
    result = run(args)
    print(json.dumps(result))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
