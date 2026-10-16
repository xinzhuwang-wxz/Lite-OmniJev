"""Create a root-owned public base view without changing original assets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from pathlib import Path

from scripts import private_calibration as cal
from scripts import private_gold_job as broker
from scripts.s0_public_baselines import sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cal.require(os.geteuid() == 0, "root_required")
    cal.require(sha256_file(args.spec) == args.spec_sha256, "spec_changed")
    spec = cal.JobSpec.model_validate_json(args.spec.read_bytes())
    source = spec.datasets[0]
    broker.safe_path(args.output)
    cal.require(not args.output.exists(), "new_view_required")
    # Validate every source hash/size before creating the view.
    records = []
    for name, digest in source.model_files.items():
        cal.require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "relative_asset_name_required",
        )
        path = Path(source.model_dir) / name
        broker.safe_path(path)
        info = path.stat()
        cal.require(
            stat.S_ISREG(info.st_mode) and sha256_file(path) == digest, "model_asset_changed"
        )
        large = info.st_size > 256 * 1024**2
        if large:
            cal.require(
                info.st_uid == 0 and bool(info.st_mode & 4) and not info.st_mode & 0o022,
                "large_weight_must_already_be_root_readonly",
            )
            cal.require(info.st_dev == args.output.parent.stat().st_dev, "same_filesystem_required")
        else:
            cal.require(info.st_size <= 64 * 1024**2, "metadata_copy_budget_exceeded")
        records.append((name, path, digest, large, info.st_size))
    args.output.mkdir(mode=0o755)
    for name, path, digest, large, _size in records:
        target = args.output / name
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if large:
            os.link(path, target)  # Never chmod/chown a hardlink to an original.
            cal.require(target.stat().st_ino == path.stat().st_ino, "hardlink_identity_changed")
        else:
            shutil.copyfile(path, target)
            os.chown(target, 0, 0)
            target.chmod(0o444)
        cal.require(sha256_file(target) == digest, "view_asset_changed")
        cal.public_readonly(target)
    result = {
        "status": "prepared",
        "exit_code": 0,
        "view": str(args.output),
        "source_spec_sha256": args.spec_sha256,
        "model_files": source.model_files,
        "hardlinked_weight_bytes": sum(size for _, _, _, large, size in records if large),
        "copied_metadata_bytes": sum(size for _, _, _, large, size in records if not large),
        "original_permissions_unchanged": True,
        "gpu_started": False,
        "private_data_read": False,
    }
    cal.write_private(args.report, result)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
