"""Verify prepared artifacts, preflight, then run one bounded gold-only candidate."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
source = Path("/root/autodl-tmp/s4-claim-pair-code-001")
prepared_dir = root / "artifacts/s4-native-event-condition-pilot-001"
prepared_path = prepared_dir / "report.json"
prepared = json.loads(prepared_path.read_text())
assert prepared["status"] == "data_ready_only" and prepared["exit_code"] == 0
assert prepared["video_families"] == prepared["video_rows"] == 60
assert prepared["gold_counts"] == {"no": 30, "yes": 30}
assert prepared["parent_model"] is None
assert (
    prepared["original_official_gold"]
    == "unchanged; proxy is never a target or admission/quality gate"
)
assert (
    prepared["output_sha256"]["train.jsonl"]
    == "fa3a16c9b0b2343fcc69fae21e7505e421c610aa5dbc6993d096498ba6789a72"
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


assert prepared["pending_parent_selection"] is True and prepared["training_authorized"] is False
parent_report_path = root / "reports/s4-temporal-coverage-training-001.json"
parent = json.loads(parent_report_path.read_text())
assert parent["status"] == "completed" and parent["steps_completed"] == 10
assert parent["adapter_sha256"] == (
    "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
)
retention_path = root / "artifacts/s4-temporal-coverage-old-retention-001/receipt.json"
retention = json.loads(retention_path.read_text())
assert retention["status"] == "completed" and retention["exit_code"] == 0
assert retention["training_report_sha256"] == sha(parent_report_path)
for cohort, minimum in {"audio": 38, "image": 39, "screenshot": 49, "text": 48}.items():
    assert retention["results"][cohort]["count"] == 50
    assert retention["results"][cohort]["correct"] >= minimum
video_path = root / "artifacts/s4-temporal-coverage-video150-001/receipt.json"
video = json.loads(video_path.read_text())
assert video["status"] == "completed" and video["results"]["video"]["correct"] == 116
cross_path = root / "artifacts/s4-temporal-coverage-original100-001/receipt-recovered-002.json"
cross = json.loads(cross_path.read_text())
assert cross["exit_code"] == 0 and cross["paired_inputs_verified"] == 100
assert cross["adapter_sha256"] == parent["adapter_sha256"] and cross["metrics"]["accuracy"] == 0.74
parent_dir = root / "artifacts/s4-temporal-coverage-train-001/adapter"
chosen_parent = {
    "path": str(parent_dir),
    "weights_sha256": parent["adapter_sha256"],
    "config_sha256": sha(parent_dir / "adapter_config.json"),
}
assert sha(parent_dir / "adapter_model.safetensors") == chosen_parent["weights_sha256"]
for name, expected in prepared["output_sha256"].items():
    assert sha(prepared_dir / name) == expected, name
assert sha(prepared_path) == "401fb4248c079c37e25cff1827b9470b3f3105ee8a3182997aee3c4de0fc7696"
exposure_path = root / "artifacts/s4-pilot-exposure-015/report.json"
assert sha(exposure_path) == prepared["exposure_base"]["report_sha256"]
exposure = json.loads(exposure_path.read_text())
assert exposure["exit_code"] == 0 and not exposure["unknown_update_bearing_inputs"]
full_path = root / "artifacts/s4-temporal-coverage-video500-001/receipt.json"
full = json.loads(full_path.read_text())
assert full["status"] == "completed" and full["exit_code"] == 0
assert full["adapter_sha256"] == parent["adapter_sha256"]
assert full["metrics"]["count"] == 500 and full["metrics"]["accuracy"] == 0.708
receipt_path = root / "reports/s4-native-event-condition-launch-001.json"
assert not receipt_path.exists()
receipt = {
    "status": "preflight_running",
    "source_commit": "3fe0ab2",
    "source_archive_sha256": sha(Path("/root/autodl-tmp/omnijev-s4-claim-pair-001.tar")),
    "prepared_report_sha256": sha(prepared_path),
    "quality_passed": False,
    "latest_exposure_report_sha256": sha(exposure_path),
    "parent_full_video_receipt_sha256": sha(full_path),
    "experiment": (
        "Native in/out-before/after original official gold; one ten-step corrective curriculum"
    ),
    "screening_stop": "DEV50 before expansion; no automatic next segment",
    "parent_selection": "root_selected_f2db_balanced_retention_and_full_video_evidence",
    "selected_init_adapter": chosen_parent,
    "placeholder_init_overridden": "data-only preparation had no selected parent",
    "retention_report_sha256": sha(retention_path),
    "video_screen_sha256": sha(video_path),
    "cross_screen_sha256": sha(cross_path),
    "optimizer_boundary": "new ten-step segment resets AdamW state; continues adapter weights",
    "cost_usd": None,
}
env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
start = time.monotonic()
try:
    assert prepared["inventory_isolation"] == "isolated"
    assert (
        receipt["source_archive_sha256"]
        == "9132c2d552af09aa4569c33df3b370fc61abe5c61eb34fa2e235b795a28f2378"
    )
    python = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
    common = [
        "--config",
        str(prepared_dir / "config-ce.json"),
        "--model-dir",
        "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2",
        "--asset-manifest",
        "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json",
        "--data",
        str(prepared_dir / "train.jsonl"),
        "--allocation-ledger",
        str(prepared_dir / "allocation-ledger.jsonl"),
        "--output-dir",
        str(root / "artifacts/s4-native-event-condition-train-001"),
        "--init-adapter",
        chosen_parent["path"],
        "--init-adapter-sha256",
        chosen_parent["weights_sha256"],
    ]
    preflight = [
        python,
        "-m",
        "scripts.train_s1",
        "preflight",
        *common,
        "--report",
        str(root / "reports/s4-native-event-condition-preflight-001.json"),
    ]
    preflight[preflight.index("--report") + 1] = str(
        root / "reports/s4-native-event-condition-preflight-001.json"
    )
    with (prepared_dir / "preflight.log").open("x") as log:
        result = subprocess.run(
            preflight, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
        )
    receipt["preflight_exit_code"] = result.returncode
    assert result.returncode == 0, "CPU preflight failed"
    p = json.loads((root / "reports/s4-native-event-condition-preflight-001.json").read_text())
    assert p["status"] == "ready" and not p.get("blockers")
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    env.update(
        SWANLAB_API_KEY=Path("/root/.config/lite-omnijev/swanlab.key").read_text().strip(),
        LITE_OMNIJEV_SWANLAB_MODE="online",
    )
    command = [
        "timeout",
        "--signal=INT",
        "--kill-after=20s",
        "240",
        python,
        "-m",
        "scripts.train_s1",
        "run",
        *common,
        "--report",
        str(root / "reports/s4-native-event-condition-training-001.json"),
    ]
    receipt.update(status="training", command=command)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    train_start = time.monotonic()
    with (prepared_dir / "train.log").open("x") as log:
        result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(
        status="completed" if result.returncode == 0 else "failed",
        exit_code=result.returncode,
        training_wall_seconds=time.monotonic() - train_start,
    )
except Exception as error:
    receipt.update(status="failed", exit_code=1, error=str(error))
finally:
    receipt["total_wall_seconds"] = time.monotonic() - start
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
raise SystemExit(receipt["exit_code"])
