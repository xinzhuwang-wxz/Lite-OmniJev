"""Public CLI behavior: pending output and tampering must never report a complete inventory."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CLI = Path(__file__).with_name("check_inventory.py")
ROOT = Path(__file__).resolve().parents[3]


def run_cli(tmp_path, entries):
    manifest = tmp_path / "inventory.json"
    manifest.write_text(
        json.dumps({"schema": "lite-omnijev.release-inventory.v1", "files": entries})
    )
    return subprocess.run(
        [sys.executable, str(CLI), "--manifest", str(manifest), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
    )


def asset(tmp_path):
    payload = b"release receipt\n"
    (tmp_path / "receipt.json").write_bytes(payload)
    return {
        "id": "receipt",
        "state": "available",
        "path": "receipt.json",
        "size": len(payload),
        "actual_sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_complete_only_claims_integrity(tmp_path):
    result = run_cli(tmp_path, [asset(tmp_path)])
    assert result.returncode == 0
    assert json.loads(result.stdout)["status"] == "inventory_verified"
    assert "not S4/S5/S6/FINAL" in json.loads(result.stdout)["claim"]


def test_generated_output_is_pending(tmp_path):
    result = run_cli(
        tmp_path,
        [
            asset(tmp_path),
            {
                "id": "calibration",
                "state": "pending",
                "reason": "S5 not passed",
            },
        ],
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["files"][-1]["status"] == "pending"


def test_tampered_and_missing_files_are_invalid(tmp_path):
    entry = asset(tmp_path)
    (tmp_path / "receipt.json").write_bytes(b"wrong checksum!!")
    result = run_cli(tmp_path, [entry])
    assert result.returncode == 1
    assert json.loads(result.stdout)["files"][0]["status"] == "invalid"
    (tmp_path / "receipt.json").unlink()
    assert json.loads(run_cli(tmp_path, [entry]).stdout)["files"][0]["status"] == "invalid"


def test_escaping_path_rejected(tmp_path):
    entry = asset(tmp_path)
    entry["path"] = "../receipt.json"
    assert run_cli(tmp_path, [entry]).returncode != 0
