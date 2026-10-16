"""Root-only cross-video ranking first10; CPU command preparation default, no autoresume."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = Path("/root/autodl-tmp/s4-video-pair-code-7b75f9f")
ARCHIVE_SHA = "49b11d244dc483a59895695468697e05ee4d45ff8b491cad377724137b15d32d"
PREPARED_SHA = "371bb26d6d2dce38784c889cb58b2aa5cecc654233c859557959ae79442bd70e"
CONFIG_SHA = "f38c6672de3cd1b4d9e0dfff72a98c93c9c20ed94c0a17cd772bfe6b0310e621"
SIDECAR_SHA = "c1ba3471df27ce6b7df7df954f7b3b5377727cf9e1902f1f811386453d18f3c8"
DATA_SHA = "17cb9a0066639dafd1feb21e3819cc38a367c86e78ce2a2907a5047d6cb437aa"
LEDGER_SHA = "1d18f75d395a6935b1fdececd7530d543144b4fa0e2a0974c31541b3d9c5fbb3"
INIT_SHA = "62920299c7a1ac33b2a874b586a9d7f3ea3eaa4e38198bc20388642482318fc6"
INIT_CONFIG_SHA = "8d79bb86dc522bc6da2b3de170dc704e51b60e2d2518582b07e8ffbfcdd6bc0a"
BASELINE_SHA = "f05ae672360c869998120ac1452bff8ad1096e39e98d7a1acf7f533590f8aa51"
CE_REPORT_SHA = "11e14016bfe5e546fe67bb0b7b1a21dbc5b89a082253ffbb743ee3bbd4e90a05"
CE_DEV_SHA = "5a43069e962446387a5e7d171c3f17197b727b111644eaa482ec8ea968bc22df"
EXPOSURE_SHA = "24b2bcb85e7067c3759cba3f007782c46b174695fd77399da15bc468cfb3a35e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate_decision(decision):
    expected = {
        "decision": "approve_first10",
        "initializer_sha256": INIT_SHA,
        "prepared_report_sha256": PREPARED_SHA,
        "config_sha256": CONFIG_SHA,
        "baseline_report_sha256": BASELINE_SHA,
        "control_training_report_sha256": CE_REPORT_SHA,
        "max_steps": 10,
        "max_seconds": 240,
    }
    require(
        all(decision.get(k) == value for k, value in expected.items()),
        "explicit root ranking decision does not bind this experiment/budget",
    )
    require(
        not decision.get("parity_validated", False)
        and not decision.get("parent_quality_inherited", False),
        "no inherited parity/quality",
    )


def verify_projection(source, archive):
    require(sha(archive) == ARCHIVE_SHA, "immutable source archive changed")
    with tarfile.open(archive) as bundle:
        for entry in bundle.getmembers():
            if entry.isfile():
                path = Path(entry.name)
                require(not path.is_absolute() and ".." not in path.parts, "unsafe archive path")
                stream = bundle.extractfile(entry)
                require(
                    stream is not None
                    and sha(source / path) == hashlib.sha256(stream.read()).hexdigest(),
                    f"immutable deployed source changed: {path}",
                )


def main(argv=None):
    require(__debug__, "Python -O disables required provenance assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256", required=True)
    args = parser.parse_args(argv)
    root, source = ROOT, SOURCE
    require(sha(args.policy) == args.policy_sha256, "root decision SHA mismatch")
    validate_decision(read(args.policy))
    pilot_dir = root / "artifacts/s4-video-pair-objective-pilot-001"
    prepared_dir = root / "artifacts/s4-continuous-matched-pool-001"
    prepared_path = pilot_dir / "report.json"
    exposure_path = root / "artifacts/s4-pilot-exposure-021/report.json"
    baseline_path = root / "reports/s4-rank64-initializer-dev50-001.json"
    ce_path = root / "reports/s4-rank64-matched-pilot-training-001.json"
    ce_dev_path = root / "reports/s4-rank64-matched-pilot-dev50-001.json"
    initializer = root / "artifacts/s4-rank64-initializer-001"
    archive = Path("/root/autodl-tmp/omnijev-video-pair-code-7b75f9f.tar")
    for path, expected in [
        (prepared_path, PREPARED_SHA),
        (pilot_dir / "config-video-pair.json", CONFIG_SHA),
        (pilot_dir / "video-pairs.json", SIDECAR_SHA),
        (prepared_dir / "train.jsonl", DATA_SHA),
        (prepared_dir / "allocation-ledger.jsonl", LEDGER_SHA),
        (initializer / "adapter_model.safetensors", INIT_SHA),
        (initializer / "adapter_config.json", INIT_CONFIG_SHA),
        (exposure_path, EXPOSURE_SHA),
        (baseline_path, BASELINE_SHA),
        (ce_path, CE_REPORT_SHA),
        (ce_dev_path, CE_DEV_SHA),
    ]:
        require(sha(path) == expected, f"bound experiment file changed: {path}")
    verify_projection(source, archive)
    prepared = read(prepared_path)
    require(
        prepared["status"] == "cpu_prepared_only" and prepared["exit_code"] == 0,
        "actual prepared artifacts required",
    )
    exposure = read(exposure_path)
    require(
        exposure["exit_code"] == 0 and not exposure["unknown_update_bearing_inputs"],
        "unknown exposure inputs",
    )
    config = read(pilot_dir / "config-video-pair.json")
    require(
        config["video_pair_sidecar_sha256"] == SIDECAR_SHA
        and config["video_pair_loss_weight"] == 0.1
        and config["checkpoint_every"] == config["midway_step"] == 10,
        "fixed objective and matched CE checkpoint cadence",
    )
    output = root / "artifacts/s4-video-pair-objective-pilot-train-001"
    require(not output.exists(), "training output already exists; no resume")
    chosen_parent = {
        "path": str(initializer),
        "weights_sha256": INIT_SHA,
        "config_sha256": INIT_CONFIG_SHA,
    }
    receipt_path = root / "reports/s4-video-pair-objective-launch-001.json"
    if not args.execute:
        receipt_path = receipt_path.with_name("s4-video-pair-objective-launch-cpu-001.json")
    require(not receipt_path.exists(), "launch receipt already exists")
    preflight_report = root / "reports/s4-video-pair-objective-preflight-001.json"
    receipt = {
        "status": "preflight_running",
        "exit_code": 1,
        "source_commit": "7b75f9f",
        "source_archive_sha256": ARCHIVE_SHA,
        "prepared_report_sha256": PREPARED_SHA,
        "config_sha256": CONFIG_SHA,
        "sidecar_sha256": SIDECAR_SHA,
        "data_sha256": DATA_SHA,
        "allocation_ledger_sha256": LEDGER_SHA,
        "policy_sha256": args.policy_sha256,
        "baseline_report_sha256": BASELINE_SHA,
        "control_training_report_sha256": CE_REPORT_SHA,
        "control_dev50_report_sha256": CE_DEV_SHA,
        "latest_exposure_report_sha256": EXPOSURE_SHA,
        "selected_init_adapter": chosen_parent,
        "parent_quality_inherited": False,
        "parity_validated": False,
        "stage_gate": False,
        "cost_usd": None,
        "training_hard_cap_seconds": 240,
        "invocation_steps": 10,
        "hypothesis": "Conditional ranking regularization; observed CE gradients already work",
        "no_resume": True,
        "wrapper_sha256": sha(Path(__file__)),
    }
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    start = time.monotonic()
    try:
        require(prepared["audited_pairs"] == 300, "actual300 pair audit required")
        assert (
            receipt["source_archive_sha256"]
            == "49b11d244dc483a59895695468697e05ee4d45ff8b491cad377724137b15d32d"
        )
        python = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
        common = [
            "--config",
            str(pilot_dir / "config-video-pair.json"),
            "--model-dir",
            "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2",
            "--asset-manifest",
            "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json",
            "--data",
            str(prepared_dir / "train.jsonl"),
            "--allocation-ledger",
            str(prepared_dir / "allocation-ledger.jsonl"),
            "--output-dir",
            str(output),
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
            str(root / "reports/s4-video-pair-objective-preflight-001.json"),
        ]
        preflight[preflight.index("--report") + 1] = str(
            root / "reports/s4-video-pair-objective-preflight-001.json"
        )
        if not args.execute:
            receipt.update(
                status="cpu_prepared_only",
                exit_code=0,
                training_authorized=False,
                root_review_pending=False,
                preflight_command=preflight,
            )
            return 0
        with (pilot_dir / "preflight.log").open("x") as log:
            result = subprocess.run(
                preflight, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
            )
        receipt["preflight_exit_code"] = result.returncode
        assert result.returncode == 0, "CPU preflight failed"
        p = json.loads((root / "reports/s4-video-pair-objective-preflight-001.json").read_text())
        assert p["status"] == "ready" and not p.get("blockers")
        require(
            p["video_pairs"]["pairs"] == 300 and p["video_pairs"]["sidecar_sha256"] == SIDECAR_SHA,
            "preflight pair contract changed",
        )
        receipt["preflight_report_sha256"] = sha(preflight_report)
        receipt["media_validation"] = p.get("preparation", {})
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
            "220",
            python,
            "-m",
            "scripts.train_s1",
            "run",
            *common,
            "--invocation-steps",
            "10",
            "--report",
            str(root / "reports/s4-video-pair-objective-training-001.json"),
        ]
        receipt.update(status="training", command=command)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        train_start = time.monotonic()
        with (pilot_dir / "train.log").open("x") as log:
            result = subprocess.run(
                command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        receipt.update(
            status="pending_runtime_snapshot" if result.returncode == 0 else "failed",
            exit_code=result.returncode,
            training_wall_seconds=time.monotonic() - train_start,
        )
        final_path = root / "reports/s4-video-pair-objective-training-001.json"
        runtime_path = output / "run.json"
        snapshot = root / "reports/s4-video-pair-objective-runtime-step0010.json"
        if runtime_path.exists():
            with snapshot.open("xb") as stream:
                stream.write(runtime_path.read_bytes())
            raw = json.loads(snapshot.read_bytes())
            final_payload = json.loads(final_path.read_bytes())
            assert final_payload.pop("allocation_ledger_check") == {
                "ledger_rows": 1200,
                "verified_train": 1200,
            }
            assert final_payload == raw
            receipt["status"] = raw["status"]
            receipt["exposure_binding"] = {
                "base_exposure_report": {"path": str(exposure_path), "sha256": sha(exposure_path)},
                "prepared_report": {"path": str(prepared_path), "sha256": sha(prepared_path)},
                "config_file": {
                    "path": str(pilot_dir / "config-video-pair.json"),
                    "sha256": sha(pilot_dir / "config-video-pair.json"),
                },
                "runtime_data": {
                    "path": str(runtime_path.parent / "data.jsonl"),
                    "sha256": sha(runtime_path.parent / "data.jsonl"),
                },
                "final_report": {"path": str(final_path), "sha256": sha(final_path)},
                "runtime_report_snapshot": {"path": str(snapshot), "sha256": sha(snapshot)},
                "runtime_report_path": str(runtime_path),
                "source_files": [
                    {"path": str(source / name), "sha256": sha(source / name)}
                    for name in [
                        "train/trainer.py",
                        "train/checkpoint.py",
                        "scripts/train_s1.py",
                        "train/video_pairs.py",
                        "train/video_pair_objective.py",
                    ]
                ],
                "start_step": 0,
                "end_step": raw.get("steps_completed", 0),
                "resume_from": None,
            }
            if result.returncode == 0:
                assert raw["status"] == "paused" and raw["steps_completed"] == 10
                assert raw["invocation"]["next_draw"] == 120
                assert raw["invocation"]["global_steps_target"] == 100
                assert (
                    sha(Path(raw["adapter_dir"]) / "adapter_model.safetensors")
                    == raw["adapter_sha256"]
                )

    except Exception as error:
        receipt.update(status="failed", exit_code=1, error=str(error))
    finally:
        receipt["total_wall_seconds"] = time.monotonic() - start
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
