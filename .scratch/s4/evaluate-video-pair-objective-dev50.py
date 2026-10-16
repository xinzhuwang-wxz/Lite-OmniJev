"""Screen actual ranking first10 against its own initializer and matched rank64 CE."""

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

HELPER = Path(__file__).with_name("evaluate-rank64-initializer-dev50.py")
spec = importlib.util.spec_from_file_location("video_pair_dev50", HELPER)
assert spec and spec.loader
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ROOT = helper.ROOT
HELPER_SHA = "1b944ca8eba834c2c3afa48cc331344751db6245d16b0c244126edae31cdf714"
CONFIG_SHA = "f38c6672de3cd1b4d9e0dfff72a98c93c9c20ed94c0a17cd772bfe6b0310e621"
SIDECAR_SHA = "c1ba3471df27ce6b7df7df954f7b3b5377727cf9e1902f1f811386453d18f3c8"
DATA_SHA = "17cb9a0066639dafd1feb21e3819cc38a367c86e78ce2a2907a5047d6cb437aa"
OWN_REPORT_SHA = "f05ae672360c869998120ac1452bff8ad1096e39e98d7a1acf7f533590f8aa51"
CE_SHA = "07c2d4dc397c75b0d030d3aa2822e1dfeac27f3653b633dc42aa92dcb4803bc4"
CE_REPORT_SHA = "5a43069e962446387a5e7d171c3f17197b727b111644eaa482ec8ea968bc22df"
POLICY_SHA = "6b7ecc322ccc001db63490811f04beea6d2b1bed81b96100a5215b1b34b8e51f"


def passes_screen(candidate):
    return (
        candidate["count"] == 50
        and math.isfinite(candidate["accuracy"])
        and math.isfinite(candidate["brier"])
        and 0.72 <= candidate["accuracy"] <= 1
        and 0 <= candidate["brier"] < 0.4401167682281881
    )


def validate_training(trained, launch):
    require = helper.require
    require(
        trained["status"] == "paused"
        and trained["exit_code"] == 0
        and trained["steps_completed"] == 10
        and trained["data_sha256"] == DATA_SHA,
        "actual fixed first10 ranking run required",
    )
    require(
        trained["init_adapter"]["weights_sha256"] == helper.INIT_SHA
        and trained["init_adapter"]["config_sha256"] == helper.CONFIG_SHA,
        "must start at same rank64 initializer, not trained CE",
    )
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
        "video_pair_loss_weight": 0.1,
        "video_pair_sidecar_sha256": SIDECAR_SHA,
        "midway_step": 10,
        "checkpoint_every": 10,
    }
    require(all(trained["config"].get(k) == v for k, v in expected.items()), "ranking config drift")
    require(
        not any(
            trained["config"].get(k, 0)
            for k in ["pair_loss_weight", "reference_kl_weight", "brier_weight"]
        ),
        "no extra objectives",
    )
    require(
        trained["video_pairs"]["pairs"] == 300
        and trained["invocation"]["next_draw"] == 120
        and trained["invocation"]["global_steps_target"] == 100,
        "same audited pairs and first120 draws",
    )
    require(
        launch["status"] == "paused"
        and launch["exit_code"] == 0
        and launch["config_sha256"] == CONFIG_SHA
        and launch["policy_sha256"] == POLICY_SHA,
        "bound actual ranking launch required",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    helper.require(helper.sha(HELPER) == HELPER_SHA, "official evaluation helper changed")
    report_path = ROOT / "reports/s4-video-pair-objective-dev50-001.json"
    if not args.execute:
        report_path = report_path.with_name("s4-video-pair-objective-dev50-cpu-001.json")
    helper.require(not report_path.exists(), "evaluation report already exists")
    started = time.monotonic()
    report = {
        "status": "failed",
        "exit_code": 1,
        "stage_gate": False,
        "training_authorized": False,
        "gpu_requested": args.execute,
        "parity_validated": False,
        "parent_quality_inherited": False,
        "budget_seconds": 180,
        "cost_usd": None,
        "policy_sha256": POLICY_SHA,
    }
    try:
        trained_path = ROOT / "reports/s4-video-pair-objective-training-001.json"
        launch_path = ROOT / "reports/s4-video-pair-objective-launch-001.json"
        trained, launch = helper.read(trained_path), helper.read(launch_path)
        validate_training(trained, launch)
        helper.require(
            launch["exposure_binding"]["final_report"]["sha256"] == helper.sha(trained_path),
            "actual training report changed",
        )
        adapter = Path(trained["adapter_dir"])
        helper.require(
            helper.sha(adapter / "adapter_model.safetensors") == trained["adapter_sha256"],
            "candidate adapter changed",
        )
        own_path = ROOT / "artifacts/s4-rank64-initializer-dev50-001/uniform"
        ce_path = ROOT / "artifacts/s4-rank64-matched-pilot-dev50-001/uniform"
        own_summary_path = ROOT / "reports/s4-rank64-initializer-dev50-001.json"
        ce_summary_path = ROOT / "reports/s4-rank64-matched-pilot-dev50-001.json"
        helper.require(
            helper.sha(own_summary_path) == OWN_REPORT_SHA
            and helper.sha(ce_summary_path) == CE_REPORT_SHA,
            "fixed baseline/CE summaries",
        )
        for summary_path, folder in [(own_summary_path, own_path), (ce_summary_path, ce_path)]:
            summary = helper.read(summary_path)
            helper.require(
                summary["candidate_receipt_sha256"] == helper.sha(folder / "receipt.json")
                and summary["candidate_records_sha256"] == helper.sha(folder / "records.jsonl"),
                "actual baseline/control records changed",
            )
        out = ROOT / "artifacts/s4-video-pair-objective-dev50-001"
        helper.run_evaluator(adapter, trained["adapter_sha256"], out, args.execute)
        report.update(
            status="cpu_prepared_only",
            exit_code=0,
            adapter_sha256=trained["adapter_sha256"],
            training_report_sha256=helper.sha(trained_path),
            launch_receipt_sha256=helper.sha(launch_path),
            own_baseline_report_sha256=OWN_REPORT_SHA,
            control_dev50_report_sha256=CE_REPORT_SHA,
            wrapper_sha256=helper.sha(Path(__file__)),
            helper_sha256=HELPER_SHA,
        )
        if args.execute:
            candidate_path = out / "uniform"
            own, candidate = helper.paired_metrics(
                own_path, candidate_path, helper.INIT_SHA, trained["adapter_sha256"]
            )
            ce, _ = helper.paired_metrics(
                ce_path, candidate_path, CE_SHA, trained["adapter_sha256"]
            )
            helper.require(
                own["accuracy"] == 0.68
                and ce["accuracy"] == 0.62
                and abs(ce["brier"] - 0.4401167682281881) < 1e-12,
                "fixed own34/control31 metrics changed",
            )
            report.update(
                status="completed",
                own_initializer=own,
                control_ce=ce,
                candidate=candidate,
                paired_input_hashes_verified=50,
                shortlist_passed=passes_screen(candidate),
                candidate_receipt_sha256=helper.sha(candidate_path / "receipt.json"),
                candidate_records_sha256=helper.sha(candidate_path / "records.jsonl"),
                next_step="Root review only; full stage gates remain unchanged",
            )
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        report["wall_seconds"] = time.monotonic() - started
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
