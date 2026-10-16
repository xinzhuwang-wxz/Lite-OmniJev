"""CPU preparation default; root-only explicit first10 rank64 pilot, no autoresume."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import tarfile
import time
from pathlib import Path

PARENT_SHA = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
INIT_SHA = "62920299c7a1ac33b2a874b586a9d7f3ea3eaa4e38198bc20388642482318fc6"
INIT_CONFIG_SHA = "8d79bb86dc522bc6da2b3de170dc704e51b60e2d2518582b07e8ffbfcdd6bc0a"
EXPANSION_SHA = "05e97013ac5d474866fdf208611bfef618bdfb65f5770c18f2924d21d37b2be7"
EXPOSURE_SHA = "eb7502b939da6339d4944421cfafae297e2251101f88c9327a4466f9226cc81f"
ARCHIVE_SHA = "636aca42044ab7d3e87b4928a99a3deb0fe10abd9a8fa60ac91d8061f9e64427"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank64_config(original):
    expected = {
        "rank": 16,
        "alpha": 32,
        "sample_count": 1200,
        "steps": 100,
        "accumulation": 12,
        "seed": 42,
        "learning_rate": 2e-5,
        "video_num_frames": 16,
        "image_max_soft_tokens": 1120,
    }
    require(all(original.get(k) == v for k, v in expected.items()), "fixed matched config changed")
    require(
        not any(
            original.get(k, 0) for k in ["pair_loss_weight", "reference_kl_weight", "brier_weight"]
        ),
        "CE only",
    )
    return {**original, "rank": 64, "alpha": 128}


POLICY_SHA = "0cc0d96a2f77edc857dc2904cff78af95d3c1fd5daa56110637bb0d695292ff1"


def validate_baseline_review(baseline, baseline_sha, decision):
    require(
        baseline.get("status") == "completed"
        and baseline.get("exit_code") == 0
        and baseline.get("gpu_requested") is True,
        "actual own GPU baseline required",
    )
    require(
        baseline.get("initializer_sha256") == INIT_SHA
        and baseline.get("initializer_config_sha256") == INIT_CONFIG_SHA
        and baseline.get("expansion_report_sha256") == EXPANSION_SHA
        and baseline.get("policy_sha256") == POLICY_SHA,
        "own baseline identity changed",
    )
    require(
        baseline.get("parity_validated") is False
        and baseline.get("parent_quality_inherited") is False,
        "failed parity must remain failed; no inherited parent quality",
    )
    metrics = baseline["candidate"]
    require(
        baseline.get("paired_input_hashes_verified") == 50 and metrics["count"] == 50,
        "fifty actual paired baseline inputs required",
    )
    require(
        math.isfinite(metrics["accuracy"])
        and math.isfinite(metrics["brier"])
        and 0.64 <= metrics["accuracy"] <= 1
        and 0 <= metrics["brier"] <= 0.46748601999097937,
        "own baseline missed shortlist; root review cannot bypass",
    )
    expected = {
        "decision": "approve_first10",
        "initializer_sha256": INIT_SHA,
        "baseline_report_sha256": baseline_sha,
        "policy_sha256": POLICY_SHA,
        "max_steps": 10,
        "max_seconds": 360,
    }
    require(
        all(decision.get(k) == v for k, v in expected.items()),
        "explicit root review does not bind this baseline/policy/first10 budget",
    )
    require(
        not decision.get("parity_validated", False)
        and not decision.get("parent_quality_inherited", False),
        "root review must not claim failed parity or inherited quality",
    )


def verify_source_projection(source, archive):
    require(sha(archive) == ARCHIVE_SHA, "immutable source archive mismatch")
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if member.isfile():
                path = Path(member.name)
                require(not path.is_absolute() and ".." not in path.parts, "unsafe source path")
                stream = bundle.extractfile(member)
                require(stream is not None, "missing source archive member")
                require(
                    sha(source / path) == hashlib.sha256(stream.read()).hexdigest(),
                    f"deployed source changed: {path}",
                )


def main(argv=None):
    require(__debug__, "run without Python -O; provenance assertions are mandatory")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--baseline-review", type=Path)
    parser.add_argument("--baseline-review-sha256")
    parser.add_argument("--control-receipt-sha256", required=True)
    args = parser.parse_args(argv)
    root = Path("/root/autodl-tmp/s1-text-lora-001")
    source = Path("/root/autodl-tmp/s4-rank64-code-b97edc0")
    prepared_dir = root / "artifacts/s4-continuous-matched-pool-001"
    prepared_path = prepared_dir / "report.json"
    prepared = json.loads(prepared_path.read_text())
    assert prepared["status"] == "data_ready_only" and prepared["exit_code"] == 0
    assert prepared["video_rows"] + prepared["replay_draws"] == 1200
    assert prepared["global_steps"] == 100
    assert prepared["video_families"] == 600
    assert prepared["parent_model"] is None

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
    assert (
        cross["adapter_sha256"] == parent["adapter_sha256"] and cross["metrics"]["accuracy"] == 0.74
    )
    parent_dir = root / "artifacts/s4-temporal-coverage-train-001/adapter"
    chosen_parent = {
        "path": str(parent_dir),
        "weights_sha256": parent["adapter_sha256"],
        "config_sha256": sha(parent_dir / "adapter_config.json"),
    }
    assert sha(parent_dir / "adapter_model.safetensors") == chosen_parent["weights_sha256"]
    for name, expected in prepared["output_sha256"].items():
        assert sha(prepared_dir / name) == expected, name
    assert sha(prepared_path) == "2c51e16cb88ab91dfb45dca84a2325b371ad259dce522a01fa50e042090cfa87"
    exposure_path = root / "artifacts/s4-pilot-exposure-020/report.json"
    assert sha(exposure_path) == EXPOSURE_SHA
    exposure = json.loads(exposure_path.read_text())
    assert exposure["exit_code"] == 0 and not exposure["unknown_update_bearing_inputs"]
    full_path = root / "artifacts/s4-temporal-coverage-video500-001/receipt.json"
    full = json.loads(full_path.read_text())
    assert full["status"] == "completed" and full["exit_code"] == 0
    assert full["adapter_sha256"] == parent["adapter_sha256"]
    assert full["metrics"]["count"] == 500 and full["metrics"]["accuracy"] == 0.708

    pilot_dir = root / "artifacts/s4-rank64-matched-pilot-001"
    require(
        not (root / "artifacts/s4-rank64-matched-pilot-train-001").exists(),
        "training output exists; no autoresume",
    )
    initializer = root / "artifacts/s4-rank64-initializer-001"
    require(
        sha(initializer / "adapter_model.safetensors") == INIT_SHA, "initializer weights changed"
    )
    require(
        sha(initializer / "adapter_config.json") == INIT_CONFIG_SHA, "initializer config changed"
    )
    require(
        sha(initializer / "rank-expansion.json") == EXPANSION_SHA, "initializer receipt changed"
    )
    expansion = json.loads((initializer / "rank-expansion.json").read_text())
    require(expansion["source_weights_sha256"] == PARENT_SHA, "derivative parent changed")
    require(
        expansion["source_config_sha256"] == chosen_parent["config_sha256"], "source config changed"
    )
    source_parent = chosen_parent
    chosen_parent = {
        "path": str(initializer),
        "weights_sha256": INIT_SHA,
        "config_sha256": INIT_CONFIG_SHA,
    }
    config = rank64_config(json.loads((prepared_dir / "config-ce.json").read_text()))
    verify_source_projection(source, Path("/root/autodl-tmp/omnijev-rank64-code-b97edc0.tar"))
    policy = root / "reports/explore/rank64-independent-candidate-policy-001.md"
    require(sha(policy) == POLICY_SHA, "independent capacity policy changed")
    control_path = root / "artifacts/s4-rank-runtime-control-001/receipt.json"
    require(sha(control_path) == args.control_receipt_sha256, "controlled failure receipt SHA")
    control = json.loads(control_path.read_text())
    require(
        control.get("status") == "completed"
        and control.get("exit_code") == 0
        and control.get("parity_validated") is False
        and control.get("expanded_weights_sha256") == INIT_SHA
        and control.get("expanded_config_sha256") == INIT_CONFIG_SHA
        and control.get("source_weights_sha256") == PARENT_SHA,
        "preserve actual failed controlled parity for this initializer",
    )
    baseline_path = root / "reports/s4-rank64-initializer-dev50-001.json"
    if args.execute:
        require(
            args.baseline_review is not None and args.baseline_review_sha256 is not None,
            "training requires explicit root review of own baseline",
        )
        require(
            sha(args.baseline_review) == args.baseline_review_sha256, "root review SHA mismatch"
        )
        baseline = json.loads(baseline_path.read_text())
        require(
            baseline["controlled_failure_receipt_sha256"] == sha(control_path),
            "baseline controlled failure changed",
        )
        validate_baseline_review(
            baseline, sha(baseline_path), json.loads(args.baseline_review.read_text())
        )
        baseline_artifact = root / "artifacts/s4-rank64-initializer-dev50-001/uniform"
        require(
            sha(baseline_artifact / "receipt.json") == baseline["candidate_receipt_sha256"]
            and sha(baseline_artifact / "records.jsonl") == baseline["candidate_records_sha256"],
            "actual baseline evidence changed",
        )
    pilot_dir.mkdir(exist_ok=True)
    config_path = pilot_dir / "config-ce.json"
    if config_path.exists():
        require(json.loads(config_path.read_text()) == config, "prepared rank64 config changed")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")

    receipt_path = root / "reports/s4-rank64-matched-pilot-launch-001.json"
    if not args.execute:
        receipt_path = receipt_path.with_name("s4-rank64-matched-pilot-cpu-001.json")
    assert not receipt_path.exists()
    receipt = {
        "status": "preflight_running",
        "source_commit": "b97edc0",
        "source_archive_sha256": sha(Path("/root/autodl-tmp/omnijev-rank64-code-b97edc0.tar")),
        "prepared_report_sha256": sha(prepared_path),
        "original_config_sha256": sha(prepared_dir / "config-ce.json"),
        "derived_config_sha256": sha(config_path),
        "derived_config_path": str(config_path),
        "data_sha256": sha(prepared_dir / "train.jsonl"),
        "allocation_ledger_sha256": sha(prepared_dir / "allocation-ledger.jsonl"),
        "quality_passed": False,
        "parent_quality_inherited": False,
        "source_parent": source_parent,
        "expansion_report_sha256": EXPANSION_SHA,
        "policy_sha256": POLICY_SHA,
        "controlled_failure_receipt_sha256": sha(control_path),
        "baseline_review_sha256": args.baseline_review_sha256,
        "baseline_report_sha256": sha(baseline_path) if args.execute else None,
        "parity_validated": False,
        "latest_exposure_report_sha256": sha(exposure_path),
        "parent_full_video_receipt_sha256": sha(full_path),
        "experiment": "Fixed exact-pair1200 pool; FIRST ten-step invocation only",
        "screening_stop": "DEV50 before expansion; no automatic next segment",
        "parent_selection": "root_selected_f2db_balanced_retention_and_full_video_evidence",
        "selected_init_adapter": chosen_parent,
        "placeholder_init_overridden": "data-only preparation had no selected parent",
        "retention_report_sha256": sha(retention_path),
        "video_screen_sha256": sha(video_path),
        "cross_screen_sha256": sha(cross_path),
        "optimizer_boundary": (
            "First invocation initializes AdamW once; "
            "no autoresume or next-segment authorization in this launcher"
        ),
        "cost_usd": None,
        "observation_window_seconds": 360,
        "usual_max_steps_before_reconsideration": 10,
        "hypothesis": "Test standard decoder LoRA capacity with only rank/alpha changed",
        "screening_policy": "First10 only; no autoresume; root review; unchanged formal gates",
    }
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    start = time.monotonic()
    try:
        assert prepared["inventory_isolation"] == "isolated"
        assert (
            receipt["source_archive_sha256"]
            == "636aca42044ab7d3e87b4928a99a3deb0fe10abd9a8fa60ac91d8061f9e64427"
        )
        python = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
        common = [
            "--config",
            str(pilot_dir / "config-ce.json"),
            "--model-dir",
            "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2",
            "--asset-manifest",
            "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json",
            "--data",
            str(prepared_dir / "train.jsonl"),
            "--allocation-ledger",
            str(prepared_dir / "allocation-ledger.jsonl"),
            "--output-dir",
            str(root / "artifacts/s4-rank64-matched-pilot-train-001"),
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
            str(root / "reports/s4-rank64-matched-pilot-preflight-001.json"),
        ]
        preflight[preflight.index("--report") + 1] = str(
            root / "reports/s4-rank64-matched-pilot-preflight-001.json"
        )
        if not args.execute:
            receipt.update(
                status="cpu_prepared_only",
                exit_code=0,
                training_authorized=False,
                root_review_pending=True,
                preflight_command=preflight,
            )
            return 0
        with (pilot_dir / "preflight.log").open("x") as log:
            result = subprocess.run(
                preflight, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
            )
        receipt["preflight_exit_code"] = result.returncode
        assert result.returncode == 0, "CPU preflight failed"
        p = json.loads((root / "reports/s4-rank64-matched-pilot-preflight-001.json").read_text())
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
            "340",
            python,
            "-m",
            "scripts.train_s1",
            "run",
            *common,
            "--invocation-steps",
            "10",
            "--report",
            str(root / "reports/s4-rank64-matched-pilot-training-001.json"),
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
        final_path = root / "reports/s4-rank64-matched-pilot-training-001.json"
        runtime_path = root / "artifacts/s4-rank64-matched-pilot-train-001/run.json"
        snapshot = root / "reports/s4-rank64-matched-pilot-runtime-step0010.json"
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
                    "path": str(pilot_dir / "config-ce.json"),
                    "sha256": sha(pilot_dir / "config-ce.json"),
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
                    for name in ["train/trainer.py", "train/checkpoint.py", "scripts/train_s1.py"]
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
