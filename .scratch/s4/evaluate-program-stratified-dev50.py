"""Fixed paired DEV50 inside the program-course window; review signals, never stage acceptance."""

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HELPER = Path(__file__).with_name("evaluate-rank64-initializer-dev50.py")
spec = importlib.util.spec_from_file_location("program_dev50", HELPER)
assert spec and spec.loader
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ROOT = helper.ROOT
HELPER_SHA = "1b944ca8eba834c2c3afa48cc331344751db6245d16b0c244126edae31cdf714"
PREPARED_SHA = "ac2f17fc3f5417e93e04d15292763384858f5e3ba1c4de533ca6f5e57d0aaf99"
DATA_SHA = "795f69fe9474523009bd953a0436bc505e85efdd4fb0f5e7c41e4d86c3bb028f"
CONFIG_SHA = "326b406eb23ce47c4854afe64d535a95ad91be94a68844109329e43872c6aaea"


def screening_disposition(metrics, step):
    if (
        metrics["count"] != 50
        or not all(math.isfinite(metrics[k]) for k in ["accuracy", "brier"])
        or not 0 <= metrics["accuracy"] <= 1
        or metrics["brier"] < 0
    ):
        return "stop_invalid_metrics"
    correct = round(metrics["accuracy"] * 50)
    if correct <= 30 or metrics["brier"] > 0.46748601999097937:
        return "stop"
    if correct == 31:
        return "gray_root_review_no_automatic_continuation"
    return (
        "root_learning_review_before_step20"
        if step == 10
        else "video150_and_probability_review_required"
    )


def record_statistics(folder):
    rows = [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
    helper.require(len(rows) == 50, "fixed50 records")
    joint_nll, conditional_nll = [], []
    for row in rows:
        helper.require(row["target_index"] in (0, 1), "noul targets only")
        gold = ["yes", "no"][row["target_index"]]
        for distribution, destination in [
            ("joint_k_plus_u", joint_nll),
            ("conditional_without_u", conditional_nll),
        ]:
            probability = row[distribution][gold]
            helper.require(
                math.isfinite(probability) and 0 < probability <= 1,
                "finite positive gold probability",
            )
            destination.append(-math.log(probability))
    return {
        "joint_nll": math.fsum(joint_nll) / 50,
        "conditional_nll": math.fsum(conditional_nll) / 50,
        "mean_p_U": math.fsum(row["p_U"] for row in rows) / 50,
        "yes_predictions": sum(row["prediction"] == "yes" for row in rows),
    }, rows


def run_evaluation(adapter, digest, out, execute):
    runner = helper.SOURCE / ".scratch/s4/evaluate-video-motion-coverage.py"
    helper.require(helper.sha(runner) == helper.RUNNER_SHA, "official runner SHA")
    code = """import importlib.util,sys
from pathlib import Path
runner,adapter,digest,out,execute=sys.argv[1:]
spec=importlib.util.spec_from_file_location("motion",runner)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ADAPTER=Path(adapter)
module.ADAPTER_SHA=digest
stages=[] if (Path(out)/"preparation.json").exists() else [[]]
if execute=="1": stages.append(["--execute","--policy","uniform"])
for flags in stages:
    sys.argv=[runner,"--output",out,*flags]
    if module.main()!=0: raise SystemExit(1)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(runner),
            str(adapter),
            digest,
            str(out),
            "1" if execute else "0",
        ],
        cwd=helper.SOURCE,
        timeout=120,
        check=False,
    )
    helper.require(result.returncode == 0, "official evaluation failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=[10, 20, 30, 40], required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    helper.require(helper.sha(HELPER) == HELPER_SHA, "evaluation helper changed")
    tag = f"step{args.step:04d}"
    report_path = ROOT / f"reports/s4-program-stratified-dev50-{tag}.json"
    if not args.execute:
        report_path = report_path.with_name(f"s4-program-stratified-dev50-cpu-{tag}.json")
    helper.require(not report_path.exists(), "immutable screen already exists")
    report = {
        "status": "failed",
        "exit_code": 1,
        "stage_gate": False,
        "training_authorized": False,
        "gpu_requested": args.execute,
        "budget_seconds": 120,
        "step": args.step,
        "cost": None,
    }
    begun = time.monotonic()
    try:
        trained_path = ROOT / f"reports/s4-program-stratified-training-{tag}.json"
        launch_path = ROOT / f"reports/s4-program-stratified-launch-{tag}.json"
        trained, launch = helper.read(trained_path), helper.read(launch_path)
        helper.require(
            trained["status"] == launch["status"] == "paused"
            and trained["exit_code"] == launch["exit_code"] == 0
            and trained["steps_completed"] == args.step
            and trained["data_sha256"] == DATA_SHA,
            "actual same-step fixed course required",
        )
        helper.require(
            launch["prepared_report_sha256"] == PREPARED_SHA
            and launch["config_sha256"] == CONFIG_SHA
            and launch["exposure_binding"]["final_report"]["sha256"] == helper.sha(trained_path),
            "bound actual training report required",
        )
        helper.require(
            trained["init_adapter"]["weights_sha256"] == helper.PARENT_SHA
            and trained["config"]["rank"] == 16
            and trained["config"]["alpha"] == 32
            and not any(
                trained["config"].get(k, 0)
                for k in [
                    "pair_loss_weight",
                    "video_pair_loss_weight",
                    "brier_weight",
                    "reference_kl_weight",
                ]
            ),
            "unchanged f2db rank16 CE",
        )
        adapter = Path(trained["adapter_dir"])
        helper.require(
            helper.sha(adapter / "adapter_model.safetensors") == trained["adapter_sha256"],
            "actual candidate changed",
        )
        frozen = helper.read(ROOT / "reports/s4-program-stratified-window-001.json")
        helper.require(launch["window"] == frozen, "window changed")
        deadline = datetime.fromisoformat(
            frozen["window_deadline_utc"].replace("Z", "+00:00")
        ).timestamp()
        helper.require(time.time() + 120 <= deadline, "insufficient original window for DEV50")
        out = ROOT / f"artifacts/s4-program-stratified-dev50-{tag}"
        run_evaluation(adapter, trained["adapter_sha256"], out, args.execute)
        report.update(
            status="cpu_prepared_only",
            exit_code=0,
            adapter_sha256=trained["adapter_sha256"],
            training_report_sha256=helper.sha(trained_path),
            launch_report_sha256=helper.sha(launch_path),
            window=frozen,
            wrapper_sha256=helper.sha(Path(__file__)),
        )
        if args.execute:
            candidate_path = out / "uniform"
            parent_path = ROOT / "artifacts/s4-temporal-coverage-dev50-001/uniform"
            parent, candidate = helper.paired_metrics(
                parent_path, candidate_path, helper.PARENT_SHA, trained["adapter_sha256"]
            )
            helper.require(
                parent["accuracy"] == 0.66 and abs(parent["brier"] - 0.44748601999097937) < 1e-12,
                "fixed f2db parent33 probability baseline",
            )
            parent_stats, parent_rows = record_statistics(parent_path)
            candidate_stats, candidate_rows = record_statistics(candidate_path)
            corrected = regressed = 0
            for left, right in zip(parent_rows, candidate_rows, strict=True):
                gold = ["yes", "no"][left["target_index"]]
                helper.require(
                    left["target_index"] == right["target_index"], "paired gold mismatch"
                )
                corrected += left["prediction"] != gold and right["prediction"] == gold
                regressed += left["prediction"] == gold and right["prediction"] != gold
            report.update(
                status="completed",
                parent=parent,
                candidate=candidate,
                parent_probability_diagnostics=parent_stats,
                candidate_probability_diagnostics=candidate_stats,
                paired_input_hashes_verified=50,
                corrected_vs_parent=corrected,
                regressed_vs_parent=regressed,
                disposition=screening_disposition(candidate, args.step),
                actual_training_loss_first=trained["losses"][0],
                actual_training_loss_last=trained["losses"][-1],
                learning_observed="root must review actual loss/gradient/exposure evidence",
                candidate_receipt_sha256=helper.sha(candidate_path / "receipt.json"),
                candidate_records_sha256=helper.sha(candidate_path / "records.jsonl"),
                next_step="No automatic continuation or quality promotion",
            )
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        report["wall_seconds"] = time.monotonic() - begun
        with report_path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
