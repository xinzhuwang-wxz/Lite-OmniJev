"""Root-reviewed ten-step CE invocations inside one immutable twenty-minute window."""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = Path("/root/autodl-tmp/s4-video-pair-code-7b75f9f")
ARCHIVE_SHA = "49b11d244dc483a59895695468697e05ee4d45ff8b491cad377724137b15d32d"
STRATEGY_SHA = "43234bfcdc1cadbf80087cdc3fe55c238549000124f0ab35c0439dc9d1f3a657"
EXPOSURE_SHA = "960a6b661f6414a514fa3274979b3a979669544979632251f52838f1f882edcb"
INIT_SHA = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
INIT_CONFIG_SHA = "a4fc244afa7a308262478aad466e250592fc15176d447ea12390d8bf321539ad"
PYTHON = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {"path": str(path), "sha256": sha(path)}


def paths(root, end):
    return {
        key: root / f"reports/s4-program-stratified-{key}-step{end:04d}.json"
        for key in ["launch", "training", "runtime", "preflight", "dev50"]
    }


def utc_seconds(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(
        parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0, "UTC required"
    )
    return parsed.timestamp()


def validate_decision(decision, prepared_sha, now, frozen=None):
    start = decision.get("start_step")
    require(type(start) is int and start in (0, 10, 20, 30), "explicit bounded start step")
    expected = {
        "decision": "approve_next10",
        "end_step": start + 10,
        "max_total_steps": 40,
        "invocation_steps": 10,
        "max_seconds": 240,
        "prepared_report_sha256": prepared_sha,
        "initializer_sha256": INIT_SHA,
        "strategy_sha256": STRATEGY_SHA,
    }
    require(
        all(decision.get(k) == value for k, value in expected.items()),
        "root decision binding/budget",
    )
    window = {
        "window_start_utc": decision["window_start_utc"],
        "window_deadline_utc": decision["window_deadline_utc"],
        "prepared_report_sha256": prepared_sha,
        "initializer_sha256": INIT_SHA,
        "max_total_steps": 40,
    }
    beginning, deadline = [
        utc_seconds(window[k]) for k in ["window_start_utc", "window_deadline_utc"]
    ]
    require(deadline - beginning == 1200 and now >= beginning, "one fixed twenty-minute window")
    if frozen is not None:
        require(window == frozen, "continuation cannot reset or extend original window")
    reserve = 120 if start == 0 else 300
    if start >= 20:
        old_reserve = decision.get("old_retention_reserve_seconds")
        require(
            type(old_reserve) is int and old_reserve > 0,
            "explicit final old-retention time reserve",
        )
        reserve += old_reserve
    require(now + 240 + reserve <= deadline, "insufficient window for training and required review")
    if start:
        require(
            decision.get("learning_observed") is True, "fresh root review must identify learning"
        )
        require(
            bool(decision.get("previous_launch_sha256"))
            and bool(decision.get("screening_report_sha256")),
            "continuation requires immutable prior launch and fresh screen",
        )
    return window


def continuation_screen(screen, start, decision):
    require(
        screen.get("status") == "completed"
        and screen.get("exit_code") == 0
        and screen.get("step") == start
        and screen.get("paired_input_hashes_verified") == 50,
        "completed same-step paired DEV50 required",
    )
    metric = screen["candidate"]
    require(
        metric["count"] == 50
        and math.isfinite(metric["accuracy"])
        and math.isfinite(metric["brier"])
        and 0.64 <= metric["accuracy"] <= 1
        and 0 <= metric["brier"] <= 0.46748601999097937,
        "DEV50 does not permit continuation; gray31 needs a separate scope decision",
    )
    if start >= 20:
        require(
            decision.get("wide_retention_passed") is True
            and bool(decision.get("video150_report_sha256")),
            "step20 wide-retention review required",
        )


def validate_prepared(report, config):
    require(
        report["status"] == "data_ready_only"
        and report["exit_code"] == 0
        and report["inventory_isolation"] == "isolated",
        "actual isolated prepared pool required",
    )
    require(
        report["video_families"] == report["video_rows"] == report["replay_draws"] == 600,
        "fixed600 video plus exact600 replay",
    )
    require(
        len(report["stratum_counts"]) == 4
        and all(counts == {"yes": 75, "no": 75} for counts in report["stratum_counts"].values()),
        "four150 original-gold program strata",
    )
    expected = {
        "sample_count": 1200,
        "steps": 100,
        "accumulation": 12,
        "seed": 42,
        "rank": 16,
        "alpha": 32,
        "learning_rate": 2e-5,
        "video_num_frames": 16,
        "image_max_soft_tokens": 1120,
        "modality": "mixed",
        "checkpoint_every": 5,
    }
    require(
        all(config.get(k) == value for k, value in expected.items()), "fixed program-course config"
    )
    require(
        not any(
            config.get(k, 0)
            for k in [
                "brier_weight",
                "reference_kl_weight",
                "pair_loss_weight",
                "video_pair_loss_weight",
            ]
        ),
        "original CE only",
    )


def verify_projection():
    archive = Path("/root/autodl-tmp/omnijev-video-pair-code-7b75f9f.tar")
    require(sha(archive) == ARCHIVE_SHA, "immutable source archive changed")
    with tarfile.open(archive) as bundle:
        for entry in bundle.getmembers():
            if entry.isfile():
                name = Path(entry.name)
                require(not name.is_absolute() and ".." not in name.parts, "safe source path")
                stream = bundle.extractfile(entry)
                require(
                    stream is not None
                    and sha(SOURCE / name) == hashlib.sha256(stream.read()).hexdigest(),
                    "deployed source changed",
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-report-sha256", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    require(sha(args.decision) == args.decision_sha256, "decision SHA changed")
    decision = read(args.decision)
    start = decision["start_step"]
    end = start + 10
    files = paths(ROOT, end)
    if not args.execute:
        files["launch"] = files["launch"].with_name(f"s4-program-stratified-cpu-step{end:04d}.json")
    require(
        not any(files[k].exists() for k in ["launch", "training", "runtime"]),
        "immutable invocation output exists",
    )
    prepared_dir = ROOT / "artifacts/s4-program-stratified-pool-001"
    prepared_path = prepared_dir / "report.json"
    require(sha(prepared_path) == args.prepared_report_sha256, "prepared report SHA changed")
    prepared = read(prepared_path)
    for name, expected in prepared["output_sha256"].items():
        require(
            Path(name).name == name and sha(prepared_dir / name) == expected,
            "prepared artifact changed",
        )
    config_path = prepared_dir / "config-ce.json"
    config = read(config_path)
    validate_prepared(prepared, config)
    verify_projection()
    initializer = ROOT / "artifacts/s4-temporal-coverage-train-001/adapter"
    require(
        sha(initializer / "adapter_model.safetensors") == INIT_SHA
        and sha(initializer / "adapter_config.json") == INIT_CONFIG_SHA,
        "f2db initializer changed",
    )
    window_path = ROOT / "reports/s4-program-stratified-window-001.json"
    frozen = read(window_path) if window_path.exists() else None
    require(not start or frozen is not None, "resume requires original frozen window")
    window = validate_decision(decision, args.prepared_report_sha256, time.time(), frozen)
    output = ROOT / "artifacts/s4-program-stratified-pool-train-001"
    raw_path = output / "run.json"
    checkpoint, previous_raw = None, None
    exposure_path = ROOT / "artifacts/s4-pilot-exposure-022/report.json"
    exposure_sha = EXPOSURE_SHA
    previous = None
    if start:
        previous_files = paths(ROOT, start)
        require(
            sha(previous_files["launch"]) == decision["previous_launch_sha256"]
            and sha(previous_files["dev50"]) == decision["screening_report_sha256"],
            "fresh root review bindings",
        )
        previous = read(previous_files["launch"])
        require(
            previous["status"] == "paused"
            and previous["exit_code"] == 0
            and previous["prepared_report_sha256"] == args.prepared_report_sha256,
            "previous accepted invocation",
        )
        require(
            sha(raw_path)
            == previous["exposure_binding"]["runtime_report_snapshot"]["sha256"]
            == sha(previous_files["runtime"]),
            "live state changed since immutable snapshot",
        )
        previous_raw = read(raw_path)
        screen = read(previous_files["dev50"])
        continuation_screen(screen, start, decision)
        require(
            screen["adapter_sha256"] == previous_raw["adapter_sha256"]
            and previous_raw["steps_completed"] == start
            and previous_raw["status"] == "paused",
            "screen/checkpoint step or adapter mismatch",
        )
        checkpoint = Path(previous_raw["checkpoint"])
        require(
            checkpoint.resolve().parent == (output / "checkpoints").resolve(),
            "local full checkpoint only",
        )
        sys.path.insert(0, str(SOURCE))
        from train.checkpoint import load_checkpoint_metadata

        metadata = load_checkpoint_metadata(checkpoint)
        order = list(range(1200))
        random.Random(42).shuffle(order)
        require(
            metadata["step"] == start
            and metadata["config"] == previous_raw["config"]
            and metadata["data_sha256"] == prepared["output_sha256"]["train.jsonl"]
            and metadata["order"] == order
            and metadata["losses"] == previous_raw["losses"],
            "full AdamW/RNG checkpoint trajectory mismatch",
        )
        exposure_path = Path(decision["latest_exposure_report"])
        exposure_sha = decision["latest_exposure_report_sha256"]
        if start >= 20:
            wide_path = Path(decision["video150_report"])
            require(
                sha(wide_path) == decision["video150_report_sha256"], "wide-retention receipt SHA"
            )
            wide = read(wide_path)
            require(
                wide.get("status") == "completed"
                and wide.get("exit_code") == 0
                and wide.get("training_report_sha256") == sha(previous_files["training"])
                and wide.get("results", {}).get("video", {}).get("count") == 150
                and wide["results"]["video"]["correct"] >= 116
                and decision.get("video150_correct") == wide["results"]["video"]["correct"]
                and decision.get("probabilities_nonworse") is True,
                "bound root wide-retention review required",
            )
    else:
        require(not output.exists(), "first invocation cannot overwrite or reset training")
    require(sha(exposure_path) == exposure_sha, "current exposure report SHA")
    exposure = read(exposure_path)
    require(
        exposure["exit_code"] == 0 and not exposure["unknown_update_bearing_inputs"],
        "unknown exposure",
    )
    common = [
        "--config",
        str(config_path),
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
        str(initializer),
        "--init-adapter-sha256",
        INIT_SHA,
    ]
    if checkpoint is not None:
        common.extend(["--resume-from", str(checkpoint)])
    preflight = [
        PYTHON,
        "-m",
        "scripts.train_s1",
        "preflight",
        *common,
        "--report",
        str(files["preflight"]),
    ]
    command = [
        "timeout",
        "--signal=INT",
        "--kill-after=20s",
        "220",
        PYTHON,
        "-m",
        "scripts.train_s1",
        "run",
        *common,
        "--invocation-steps",
        "10",
        "--report",
        str(files["training"]),
    ]
    receipt = {
        "status": "cpu_prepared_only",
        "exit_code": 0,
        "stage_gate": False,
        "parent_quality_inherited": False,
        "prepared_report_sha256": args.prepared_report_sha256,
        "config_sha256": sha(config_path),
        "decision_sha256": args.decision_sha256,
        "strategy_sha256": STRATEGY_SHA,
        "source_commit": "7b75f9f",
        "source_archive_sha256": ARCHIVE_SHA,
        "selected_init_adapter": {
            "path": str(initializer),
            "weights_sha256": INIT_SHA,
            "config_sha256": INIT_CONFIG_SHA,
        },
        "window": window,
        "start_step": start,
        "end_step_target": end,
        "command": command,
        "preflight_command": preflight,
        "cost_usd": None,
        "wrapper_sha256": sha(Path(__file__)),
        "optimizer_boundary": "fresh AdamW" if start == 0 else "restore full AdamW/RNG checkpoint",
    }
    begun = time.monotonic()
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    try:
        if args.execute:
            if frozen is None:
                with window_path.open("x") as stream:
                    json.dump(window, stream, indent=2)
            with (prepared_dir / f"preflight-step{end:04d}.log").open("x") as log:
                result = subprocess.run(
                    preflight,
                    cwd=SOURCE,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
            require(
                result.returncode == 0 and read(files["preflight"])["status"] == "ready",
                "CPU preflight failed",
            )
            receipt["preflight_report"] = binding(files["preflight"])
            require(
                not subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
                ).strip(),
                "GPU busy",
            )
            validate_decision(decision, args.prepared_report_sha256, time.time(), window)
            env.update(
                SWANLAB_API_KEY=Path("/root/.config/lite-omnijev/swanlab.key").read_text().strip(),
                LITE_OMNIJEV_SWANLAB_MODE="online",
            )
            receipt.update(status="running", exit_code=1)
            files["launch"].write_text(json.dumps(receipt, indent=2) + "\n")
            with (prepared_dir / f"train-step{end:04d}.log").open("x") as log:
                result = subprocess.run(
                    command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            receipt["exit_code"] = result.returncode
            with files["runtime"].open("xb") as stream:
                stream.write(raw_path.read_bytes())
            raw = read(files["runtime"])
            final = read(files["training"])
            require(
                final.pop("allocation_ledger_check")
                == {"ledger_rows": 1200, "verified_train": 1200}
                and final == raw,
                "final/runtime report mismatch",
            )
            receipt["status"] = raw["status"]
            receipt["exposure_binding"] = {
                "base_exposure_report": binding(exposure_path),
                "prepared_report": binding(prepared_path),
                "config_file": binding(config_path),
                "runtime_data": binding(output / "data.jsonl"),
                "final_report": binding(files["training"]),
                "runtime_report_snapshot": binding(files["runtime"]),
                "runtime_report_path": str(raw_path),
                "source_files": [
                    binding(SOURCE / name)
                    for name in ["train/trainer.py", "train/checkpoint.py", "scripts/train_s1.py"]
                ],
                "start_step": start,
                "end_step": raw.get("steps_completed", 0),
                "resume_from": str(checkpoint) if checkpoint else None,
            }
            require(
                result.returncode == 0
                and raw["status"] == "paused"
                and raw["steps_completed"] == end
                and raw["invocation"]["start_step"] == start
                and raw["invocation"]["next_draw"] == end * 12,
                "incomplete bounded invocation",
            )
            require(
                not previous_raw or raw["losses"][:start] == previous_raw["losses"],
                "prior trajectory changed",
            )
            require(
                sha(Path(raw["adapter_dir"]) / "adapter_model.safetensors")
                == raw["adapter_sha256"],
                "saved adapter SHA",
            )
    except Exception as exc:
        receipt.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        receipt["wall_seconds"] = time.monotonic() - begun
        files["launch"].write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
