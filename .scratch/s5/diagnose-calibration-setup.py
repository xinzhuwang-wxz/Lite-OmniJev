"""Fixed metadata-only setup diagnostics: never open private data or worker logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts import private_calibration as cal
from scripts import private_gold_job as broker
from scripts.s0_public_baselines import sha256_file

ALLOWED = {"public_readonly_asset_required", "unsafe_asset_ancestor"}


def classify_assets(directory: str, files: dict[str, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for name in files:
        try:
            cal.public_readonly(Path(directory) / name)  # stat/path only; never content
            counts["accessible"] += 1
        except ValueError as error:
            # Compare only complete predefined internal codes; no arbitrary exception text exported.
            code = error.args[0] if len(error.args) == 1 else None
            counts[code if isinstance(code, str) and code in ALLOWED else "unclassified"] += 1
        except Exception:
            counts["unclassified"] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--worker-uid", type=int, required=True)
    parser.add_argument("--probe-uid", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cal.require(sha256_file(args.spec) == args.spec_sha256, "spec_changed")
    spec = cal.JobSpec.model_validate_json(args.spec.read_bytes())
    first = spec.datasets[0]
    assert first.adapter_dir and first.adapter_files
    backups = []
    for path in args.work_root.glob("*-root-acl.json"):
        record = json.loads(broker.bounded_read(path, 16384))  # root ACL metadata only
        if record["uid"] == args.worker_uid:
            backups.append((path, record))
    cal.require(len(backups) == 1, "one_uid_acl_record_required")
    backup, record = backups[0]
    run = args.work_root / backup.name.removesuffix("-root-acl.json")
    before = record["acl_before"]
    after = cal.PosixACL().read(Path("/root"))
    clean = True
    try:
        broker.check_process_ids({args.worker_uid, args.probe_uid})
    except broker.JobRejected:
        clean = False
    result = {
        "schema_version": "calibration-setup-diagnostic-v1",
        "spec_sha256": args.spec_sha256,
        "producer_sha256": sha256_file(Path(cal.__file__)),
        "base_asset_checks": classify_assets(first.model_dir, first.model_files),
        "adapter_asset_checks": classify_assets(first.adapter_dir, first.adapter_files),
        "snapshot_metadata": {
            "data_files": len(list((run / "inputs/data").glob("*.jsonl"))),
            "state_created": (run / "inputs/state.json").exists(),
            "runtime_started": (run / "output/_runtime_probe.log").exists(),
            "plan_started": (run / "output/_plan.log").exists(),
            "inference_started": (run / "output/_fit.log").exists(),
        },
        "acl_before_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "acl_after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "acl_restored": before == after,
        "fresh_uid_processes_absent": clean,
        "recovery_journal_absent": not Path(
            "/var/tmp/omnijev-root-traverse-acl-recovery.json"
        ).exists(),
        "private_logs_opened": False,
        "private_rows_opened": False,
        "gpu_started": False,
    }
    cal.write_private(args.report, result)
    print(json.dumps(result))
    return 0 if result["acl_restored"] and clean and result["recovery_journal_absent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
