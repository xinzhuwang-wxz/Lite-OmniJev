"""Fixed DEV50 for actual rank64 first10 candidate; CPU default, no automatic expansion."""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

HELPER = Path(__file__).with_name("evaluate-rank64-initializer-dev50.py")
spec = importlib.util.spec_from_file_location("rank64_dev50", HELPER)
assert spec and spec.loader
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ROOT = helper.ROOT
RANK16_SHA = "f6e9db3ffe88644a932e85eb6832918a42657fdb7c87628bcb7d4e09024e0fe2"


def validate_candidate(trained, prepared, launch):
    require = helper.require
    require(
        trained["status"] == "paused"
        and trained["exit_code"] == 0
        and trained["steps_completed"] == 10,
        "actual first10 candidate required",
    )
    require(
        trained["init_adapter"]["weights_sha256"] == helper.INIT_SHA
        and trained["init_adapter"]["config_sha256"] == helper.CONFIG_SHA,
        "candidate must use exact independent rank64 initializer",
    )
    require(
        trained["data_sha256"] == prepared["output_sha256"]["train.jsonl"],
        "same fixed matched training data required",
    )
    config = trained["config"]
    expected = {
        "rank": 64,
        "alpha": 128,
        "sample_count": 1200,
        "steps": 100,
        "accumulation": 12,
        "seed": 42,
        "learning_rate": 2e-5,
        "video_num_frames": 16,
        "image_max_soft_tokens": 1120,
    }
    require(all(config.get(k) == v for k, v in expected.items()), "candidate config changed")
    require(
        not any(
            config.get(k, 0) for k in ["pair_loss_weight", "reference_kl_weight", "brier_weight"]
        ),
        "CE only",
    )
    require(
        trained["invocation"]["next_draw"] == 120
        and trained["invocation"]["global_steps_target"] == 100,
        "first120 actual draws",
    )
    require(
        launch["exit_code"] == 0
        and launch["status"] == "paused"
        and launch["selected_init_adapter"]["weights_sha256"] == helper.INIT_SHA
        and launch["parity_validated"] is False
        and launch["parent_quality_inherited"] is False,
        "actual independent candidate launch",
    )


def passes_screen(own, candidate):
    return (
        own["count"] == candidate["count"] == 50
        and all(math.isfinite(m[k]) for m in [own, candidate] for k in ["accuracy", "brier"])
        and round(candidate["accuracy"] * 50) >= round(own["accuracy"] * 50) + 2
        and candidate["accuracy"] >= 0.68
        and candidate["brier"] < own["brier"]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-rank16-control", action="store_true")
    args = parser.parse_args(argv)
    report_path = ROOT / "reports/s4-rank64-matched-pilot-dev50-001.json"
    if not args.execute:
        report_path = report_path.with_name("s4-rank64-matched-pilot-dev50-cpu-001.json")
    helper.require(not report_path.exists(), "evaluation report already exists")
    report = {
        "status": "failed",
        "exit_code": 1,
        "stage_gate": False,
        "training_authorized": False,
        "parity_validated": False,
        "parent_quality_inherited": False,
        "budget_seconds": 180,
        "gpu_requested": args.execute,
        "cost_usd": None,
    }
    start = time.monotonic()
    try:
        trained_path = ROOT / "reports/s4-rank64-matched-pilot-training-001.json"
        trained = helper.read(trained_path)
        prepared = helper.read(ROOT / "artifacts/s4-continuous-matched-pool-001/report.json")
        launch_path = ROOT / "reports/s4-rank64-matched-pilot-launch-001.json"
        launch = helper.read(launch_path)
        validate_candidate(trained, prepared, launch)
        helper.require(
            launch["exposure_binding"]["final_report"]["sha256"] == helper.sha(trained_path),
            "launch final report binding changed",
        )
        adapter = Path(trained["adapter_dir"])
        helper.require(
            helper.sha(adapter / "adapter_model.safetensors") == trained["adapter_sha256"],
            "actual candidate weights changed",
        )
        own_path = ROOT / "artifacts/s4-rank64-initializer-dev50-001/uniform"
        own_summary_path = ROOT / "reports/s4-rank64-initializer-dev50-001.json"
        own_summary = helper.read(own_summary_path)
        helper.require(
            launch["baseline_report_sha256"] == helper.sha(own_summary_path)
            and own_summary["candidate_receipt_sha256"] == helper.sha(own_path / "receipt.json")
            and own_summary["candidate_records_sha256"] == helper.sha(own_path / "records.jsonl"),
            "own baseline evidence changed since root review",
        )
        out = ROOT / "artifacts/s4-rank64-matched-pilot-dev50-001"
        helper.run_evaluator(adapter, trained["adapter_sha256"], out, args.execute)
        report.update(
            status="cpu_prepared_only",
            exit_code=0,
            adapter_sha256=trained["adapter_sha256"],
            training_report_sha256=helper.sha(trained_path),
            launch_receipt_sha256=helper.sha(launch_path),
            own_baseline_report_sha256=helper.sha(own_summary_path),
            wrapper_sha256=helper.sha(Path(__file__)),
            helper_sha256=helper.sha(HELPER),
        )
        if args.execute:
            candidate_path = out / "uniform"
            own, candidate = helper.paired_metrics(
                own_path, candidate_path, helper.INIT_SHA, trained["adapter_sha256"]
            )
            parent_path = ROOT / "artifacts/s4-temporal-coverage-dev50-001/uniform"
            parent, _ = helper.paired_metrics(
                parent_path, candidate_path, helper.PARENT_SHA, trained["adapter_sha256"]
            )
            report.update(
                status="completed",
                own_initializer=own,
                parent=parent,
                candidate=candidate,
                paired_input_hashes_verified=50,
                shortlist_passed=passes_screen(own, candidate),
                candidate_receipt_sha256=helper.sha(candidate_path / "receipt.json"),
                candidate_records_sha256=helper.sha(candidate_path / "records.jsonl"),
                next_step="Root review; video150 >116 plus cross/old retention before expansion",
            )
            if args.include_rank16_control:
                control_path = ROOT / "artifacts/s4-continuous-matched-pool-dev50-001/uniform"
                control, _ = helper.paired_metrics(
                    control_path, candidate_path, RANK16_SHA, trained["adapter_sha256"]
                )
                report["rank16_matched_control"] = control
                report["rank16_control_adapter_sha256"] = RANK16_SHA
                report["rank16_control_receipt_sha256"] = helper.sha(control_path / "receipt.json")
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        report["wall_seconds"] = time.monotonic() - start
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
