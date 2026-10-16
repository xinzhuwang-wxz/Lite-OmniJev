"""Validate one polarity arm by default; explicit execution is one bounded train+DEV50 job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
CORE = Path("/root/autodl-tmp/s4-claim-pair-code-001")
ARCHIVE = Path("/root/autodl-tmp/omnijev-s4-claim-pair-001.tar")
ARCHIVE_SHA = "9132c2d552af09aa4569c33df3b370fc61abe5c61eb34fa2e235b795a28f2378"
PREP_SHA = "1d14c7780892fe10eda4b2606dece555fbbd73179574f60b42ba46784411ab80"
PARENT_SHA = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
MODEL = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
PYTHON = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
MOTION = Path("/root/autodl-tmp/s4-motion-code-142e744")
RUNNER_SHA = "e392f10b7738f9cf776087dfb7cd40b35377d974f986fc5cd9ad47bb77b51246"
DATA = {
    "control": "f7cbd1a5ee86c92cb13fc9377d9942b4e6907e0182d685431103b4d20ddc85bc",
    "polarity": "0235ff3f039c28229c897bf79a5f353958d34d4bda486035cc72a21df691cb84",
}


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p: Path) -> Any:
    return json.loads(p.read_bytes())


def write(p: Path, value: Any, *, new: bool = False) -> None:
    if new:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
    else:
        temp = p.with_suffix(".tmp")
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2)
            f.write("\n")
        temp.replace(p)


def arm_contract(arm: str, prepared: dict[str, Any], config: dict[str, Any]) -> None:
    from train.trainer import LoopConfig

    require(arm in DATA, "explicit_single_arm")
    require(
        prepared.get("status") == "cpu_prepared_pending_root_training_decision"
        and prepared.get("exit_code") == 0
        and prepared.get("training_authorized") is False
        and prepared.get("families") == 30,
        "prepared_prototype_identity",
    )
    a = prepared["arms"][arm]
    require(
        a["inventory_isolation"] == "isolated"
        and a["output_sha256"]["train.jsonl"] == DATA[arm]
        and a["ledger_check"]["verified_train"] == 120,
        "arm_isolation_data",
    )
    c = LoopConfig.model_validate(config)
    require(
        c.seed == 42
        and c.sample_count == 120
        and c.steps == 10
        and c.accumulation == 12
        and c.midway_step == 5
        and c.checkpoint_every == 5
        and c.rank == 16
        and c.alpha == 32
        and c.dtype == "bfloat16"
        and c.video_num_frames == 16
        and c.image_max_soft_tokens == 1120
        and c.pair_loss_weight == c.reference_kl_weight == c.brier_weight == 0,
        "fixed_ce120_budget",
    )


def paths(arm: str) -> dict[str, Path]:
    require(arm in DATA, "explicit_single_arm")
    name = "s4-video-polarity-" + arm
    return {
        "prep": ROOT / "artifacts/s4-video-polarity-pilot-001" / arm,
        "train": ROOT / "artifacts" / (name + "-train-001"),
        "training": ROOT / "reports" / (name + "-training-001.json"),
        "preflight": ROOT / "reports" / (name + "-preflight-001.json"),
        "launch": ROOT / "reports" / (name + "-launch-001.json"),
        "dev": ROOT / "artifacts" / (name + "-dev50-001"),
        "dev_report": ROOT / "reports" / (name + "-dev50-001.json"),
    }


def validate(arm: str) -> dict[str, Any]:
    from scripts.train_s1 import verify_data_is_train_pool
    from train.trainer import LoopConfig, smoke_samples

    p = paths(arm)
    prep_root = p["prep"].parent
    require(sha(prep_root / "report.json") == PREP_SHA, "frozen_prototype_report")
    prepared = read(prep_root / "report.json")
    config = read(p["prep"] / "config-ce.json")
    arm_contract(arm, prepared, config)
    bindings = {}

    def bound(path: Path, expected: str | None = None) -> str:
        value = sha(path)
        require(expected is None or value == expected, "input_hash_changed:" + path.name)
        bindings[str(path)] = value
        return value

    bound(ARCHIVE, ARCHIVE_SHA)
    # Read trusted source archive only; never extract or execute archive content.
    with tarfile.open(ARCHIVE) as archive:
        for m in archive.getmembers():
            if (
                m.isfile()
                and m.name.endswith(".py")
                and m.name.startswith(("train/", "data/", "scripts/"))
            ):
                f = archive.extractfile(m)
                assert f is not None
                bound(CORE / m.name, hashlib.sha256(f.read()).hexdigest())
    bound(prep_root / "report.json")
    for name, expected in prepared["arms"][arm]["output_sha256"].items():
        require(Path(name).name == name, "safe_prepared_filename")
        bound(p["prep"] / name, expected)
    audit = read(p["prep"] / "inventory-audit.json")
    require(
        audit["outcome"] == "isolated"
        and audit["manifest_sha256"] == sha(p["prep"] / "inventory-manifest.json"),
        "bound_inventory_isolation",
    )
    verify_data_is_train_pool(p["prep"] / "train.jsonl", [p["prep"] / "allocation-ledger.jsonl"])
    smoke_samples(
        LoopConfig.model_validate(config),
        p["prep"] / "train.jsonl",
        expected_data_versions=config["expected_data_versions"],
    )
    parent = ROOT / "artifacts/s4-temporal-coverage-train-001/adapter"
    bound(parent / "adapter_model.safetensors", PARENT_SHA)
    parent_config = bound(parent / "adapter_config.json")
    parent_report = ROOT / "reports/s4-temporal-coverage-training-001.json"
    r = read(parent_report)
    require(
        r["status"] == "completed"
        and r["steps_completed"] == 10
        and r["adapter_sha256"] == PARENT_SHA,
        "parent_terminal_training",
    )
    bound(parent_report)
    for path in [
        ROOT / "artifacts/s4-temporal-coverage-old-retention-001/receipt.json",
        ROOT / "artifacts/s4-temporal-coverage-video150-001/receipt.json",
        ROOT / "artifacts/s4-temporal-coverage-original100-001/receipt-recovered-002.json",
    ]:
        bound(path)
    bound(
        MODEL / "chat_template.jinja",
        "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
    )
    bound(MOTION / ".scratch/s4/evaluate-video-motion-coverage.py", RUNNER_SHA)
    baseline_dir = ROOT / "artifacts/s4-motion-coverage-pilot-001/uniform"
    baseline_receipt = read(baseline_dir / "receipt.json")
    require(
        baseline_receipt["status"] == "completed"
        and baseline_receipt["identity"]["passes"] == 1
        and baseline_receipt["identity"]["runner_sha256"] == RUNNER_SHA,
        "fixed_dev_baseline",
    )
    bound(baseline_dir / "receipt.json")
    bound(baseline_dir / "records.jsonl")
    for name, expected in baseline_receipt["identity"]["code_sha256"].items():
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts, "safe_eval_code_path"
        )
        bound(MOTION / name, expected)
    bound(
        ROOT / "reports/s3-dev-eval-video-stable-001.json",
        baseline_receipt["identity"]["source_report_sha256"],
    )

    common = [
        "--config",
        str(p["prep"] / "config-ce.json"),
        "--model-dir",
        str(MODEL),
        "--asset-manifest",
        "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json",
        "--data",
        str(p["prep"] / "train.jsonl"),
        "--allocation-ledger",
        str(p["prep"] / "allocation-ledger.jsonl"),
        "--output-dir",
        str(p["train"]),
        "--init-adapter",
        str(parent),
        "--init-adapter-sha256",
        PARENT_SHA,
    ]
    return {
        "status": "dry_validated",
        "arm": arm,
        "gpu_used": False,
        "stage_gate": False,
        "selected_init_adapter": {
            "path": str(parent),
            "weights_sha256": PARENT_SHA,
            "config_sha256": parent_config,
        },
        "bound_inputs_sha256": bindings,
        "prepared_report_sha256": sha(prep_root / "report.json"),
        "core_commit": "3fe0ab2",
        "core_archive_sha256": ARCHIVE_SHA,
        "preflight_argv": [
            PYTHON,
            "-m",
            "scripts.train_s1",
            "preflight",
            *common,
            "--report",
            str(p["preflight"]),
        ],
        "training_argv": [
            "timeout",
            "--signal=INT",
            "--kill-after=20s",
            "240",
            PYTHON,
            "-m",
            "scripts.train_s1",
            "run",
            *common,
            "--report",
            str(p["training"]),
        ],
        "dev50_argv": [
            "timeout",
            "--signal=INT",
            "--kill-after=20s",
            "180",
            PYTHON,
            str(Path(__file__).resolve()),
            "--arm",
            arm,
            "--phase",
            "dev",
            "--execute",
        ],
        "training_limit_seconds": 240,
        "dev50_limit_seconds": 180,
        "launcher_sha256": sha(Path(__file__)),
        "cost": None,
    }


def eval_dev(arm: str) -> dict[str, Any]:
    p = paths(arm)
    trained = read(p["training"])
    run = read(p["train"] / "run.json")
    require(
        trained["status"] == run["status"] == "completed"
        and trained["steps_completed"] == run["steps_completed"] == 10
        and trained["exit_code"] == run["exit_code"] == 0,
        "terminal_training_before_dev",
    )
    require(
        trained["data_sha256"] == run["data_sha256"] == DATA[arm]
        and trained["init_adapter"]["weights_sha256"]
        == run["init_adapter"]["weights_sha256"]
        == PARENT_SHA,
        "trained_arm_identity",
    )
    adapter = p["train"] / "adapter"
    require(
        sha(adapter / "adapter_model.safetensors")
        == trained["adapter_sha256"]
        == run["adapter_sha256"],
        "trained_weights",
    )
    runner = MOTION / ".scratch/s4/evaluate-video-motion-coverage.py"
    require(sha(runner) == RUNNER_SHA, "fixed_dev50_runner")
    os.chdir(MOTION)
    sys.path.insert(0, str(MOTION))
    spec = importlib.util.spec_from_file_location("motion", runner)
    assert spec and spec.loader
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ADAPTER = adapter
    module.ADAPTER_SHA = trained["adapter_sha256"]
    for flags in [[], ["--execute", "--policy", "uniform"]]:
        sys.argv = [str(runner), "--output", str(p["dev"]), *flags]
        require(module.main() == 0, "dev50_failed")
    baseline = ROOT / "artifacts/s4-motion-coverage-pilot-001/uniform"
    candidate = p["dev"] / "uniform"
    old, new = read(baseline / "receipt.json"), read(candidate / "receipt.json")
    require(
        old["status"] == new["status"] == "completed" and new["identity"]["passes"] == 1,
        "dev50_terminal_single_pass",
    )
    for key, value in old["identity"].items():
        if key not in ["adapter_sha256", "adapter_config_sha256"]:
            require(new["identity"][key] == value, "same_dev_identity:" + key)
    before = [json.loads(line) for line in (baseline / "records.jsonl").read_bytes().splitlines()]
    after = [json.loads(line) for line in (candidate / "records.jsonl").read_bytes().splitlines()]
    require(len(before) == len(after) == 50, "exact_dev50")
    for a, b in zip(before, after, strict=True):
        require(
            all(
                a[k] == b[k]
                for k in ["id", "input_sha256", "input_summary_sha256", "answer_token_ids"]
            ),
            "same_dev_input",
        )
    result = {
        "status": "completed",
        "exit_code": 0,
        "stage_gate": False,
        "scope": "original fixed heuristic temporal DEV50, single pass",
        "arm": arm,
        "training_report_sha256": sha(p["training"]),
        "adapter_sha256": trained["adapter_sha256"],
        "same_input_rows": 50,
        "baseline": old["metrics"],
        "candidate": new["metrics"],
        "parent_adapter_sha256": PARENT_SHA,
        "parent_metrics": read(ROOT / "reports/s4-temporal-coverage-dev50-001.json")["candidate"],
        "candidate_receipt_sha256": sha(candidate / "receipt.json"),
        "baseline_receipt_sha256": sha(baseline / "receipt.json"),
        "cost": None,
    }
    write(p["dev_report"], result, new=True)
    return result


def execute(arm: str, validation: dict[str, Any]) -> dict[str, Any]:
    p = paths(arm)
    require(
        all(
            not p[k].exists()
            for k in ["train", "training", "preflight", "launch", "dev", "dev_report"]
        ),
        "refuse_overwrite",
    )
    other = paths("polarity" if arm == "control" else "control")["launch"]
    require(
        not other.exists() or read(other).get("status") in {"completed", "failed"},
        "other_arm_nonterminal_no_queue",
    )
    receipt = {
        **validation,
        "status": "preflight_running",
        "gpu_used": False,
        "exposure_reconciliation": "required_after_any_actual_training_updates_even_if_dev_fails",
    }
    write(p["launch"], receipt, new=True)
    start = time.monotonic()
    env = dict(
        os.environ,
        PYTHONPATH=str(CORE),
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    try:
        with (p["prep"] / "launcher-preflight.log").open("x") as log:
            r = subprocess.run(
                validation["preflight_argv"],
                cwd=CORE,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        require(r.returncode == 0, "preflight_exit")
        check = read(p["preflight"])
        require(check["status"] == "ready" and not check.get("blockers"), "preflight_not_ready")
        require(
            not subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip(),
            "gpu_busy_no_queue",
        )
        for path, expected in validation["bound_inputs_sha256"].items():
            require(sha(Path(path)) == expected, "input_mutated_before_train")
        # Key remains outside repository and serialized receipts.
        # Core telemetry contains real step metrics only.
        env.update(
            SWANLAB_API_KEY=Path("/root/.config/lite-omnijev/swanlab.key").read_text().strip(),
            LITE_OMNIJEV_SWANLAB_MODE="online",
        )
        receipt.update(status="training", gpu_used=True)
        write(p["launch"], receipt)
        with (p["prep"] / "launcher-train.log").open("x") as log:
            r = subprocess.run(
                validation["training_argv"], cwd=CORE, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        receipt["training_exit_code"] = r.returncode
        require(r.returncode == 0, "training_failed_no_dev")
        trained = read(p["training"])
        require(
            trained["status"] == "completed"
            and trained["steps_completed"] == 10
            and trained["exit_code"] == 0,
            "training_not_complete",
        )
        receipt.update(
            status="dev50_running",
            training_report_sha256=sha(p["training"]),
            exposure_reconciliation="pending_terminal_arm_reconcile",
        )
        write(p["launch"], receipt)
        with (p["prep"] / "launcher-dev50.log").open("x") as log:
            r = subprocess.run(
                validation["dev50_argv"], cwd=CORE, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        require(r.returncode == 0, "dev50_failed")
        receipt.update(status="completed", exit_code=0, dev50_report_sha256=sha(p["dev_report"]))
    except Exception as error:
        receipt.update(status="failed", exit_code=1, error=type(error).__name__ + ":" + str(error))
    finally:
        if p["training"].exists():
            actual = read(p["training"])
            receipt["actual_training_receipt"] = {
                "path": str(p["training"]),
                "sha256": sha(p["training"]),
                "status": actual.get("status"),
                "steps_completed": actual.get("steps_completed"),
            }
        receipt["duration_seconds"] = time.monotonic() - start
        write(p["launch"], receipt)
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["control", "polarity"], required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--phase", choices=["launch", "dev"], default="launch")
    ap.add_argument("--validation-report", type=Path)
    a = ap.parse_args()
    os.umask(0o077)
    try:
        if a.phase == "dev":
            require(a.execute and a.validation_report is None, "explicit_internal_dev_execution")
            r = eval_dev(a.arm)
        else:
            r = validate(a.arm)
            if a.validation_report:
                write(a.validation_report, r, new=True)
            if a.execute:
                r = execute(a.arm, r)
        print(json.dumps(r))
        return int(r.get("exit_code", 0))
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "exit_code": 1,
                    "error": type(error).__name__ + ":" + str(error),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
