"""Root-reviewed fixed video150 with cached f2db comparison; no parent GPU rerun."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = Path("/root/autodl-tmp/s4-monitored-eval-code-001")
PARENT_SHA = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
TRAIN_DATA_SHA = "795f69fe9474523009bd953a0436bc505e85efdd4fb0f5e7c41e4d86c3bb028f"
CONFIG_SHA = "326b406eb23ce47c4854afe64d535a95ad91be94a68844109329e43872c6aaea"
PREPARED_SHA = "ac2f17fc3f5417e93e04d15292763384858f5e3ba1c4de533ca6f5e57d0aaf99"
DEV_DATA_SHA = "e9b91fedecf2ccf7dc30e837ac11bcc4b233624d1f6c16812a40b1319c15b080"
PARENT_REPORT_SHA = "7d2109fcab1034efb711a89f0cc084fe350e84d9ab80ff6c17d715d5616a0918"
PARENT_RECORDS_SHA = "bdb6272f2d77c6a0fc02c95046c868a03be5e7c5a557ee7a70be7864aebc39e5"
SOURCE_FILES = {
    "scripts/s1_dev_eval.py": "12d387e778a685a8f49ef223f5fa2d26538e3e6a052c7c8e5013677186ad9c68",
    "train/trainer.py": "0de8f4c1e62b56f07611abf062810890cb04acf8c591b0c1b30d3bd9a3c1bf3e",
    "train/template.py": "be82e8a5973a8e3ebba81e5df6dac8478c7b4d9b0984c2320f05fe792eca4a36",
    "train/multimodal.py": "09c5df0cb4ea5917910282b0fd2ae3ba55778049dc8e33b36f9b1d3ee469f265",
    "train/readout.py": "fce1d13a6a1d024c0abc3b10926d3637a8c3a5ee3497a0d39da1f4d5e46a225b",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def validate_review(review, step, training_sha, dev50_sha, adapter_sha):
    expected = {
        "decision": "approve_video150",
        "step": step,
        "training_report_sha256": training_sha,
        "dev50_report_sha256": dev50_sha,
        "adapter_sha256": adapter_sha,
        "max_seconds": 180,
    }
    require(
        all(review.get(k) == v for k, v in expected.items()),
        "explicit same-checkpoint video150 review required",
    )


def original_records(path, samples):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        require(row["pass"] == 0 and row["variant"] == "original", "one original pass only")
        require(row["id"] in samples and row["id"] not in result, "unknown/duplicate video150 ID")
        sample = samples[row["id"]]
        require(
            row["pool"] == sample["meta"]["pool"] == "dev"
            and row["family_hash"] == sample["meta"]["family_id"]
            and row["target"] == sample["questions"][0]["target"]["value"],
            "record/source identity or gold mismatch",
        )
        require(
            set(row["raw_logits"]) == set(row["joint_probabilities"]) == {"yes", "no", "U"}
            and set(row["probabilities"]) == {"yes", "no"},
            "complete legal answer/U readout",
        )
        require(all(math.isfinite(v) for v in row["raw_logits"].values()), "nonfinite legal logits")
        result[row["id"]] = row
    require(set(result) == set(samples) and len(result) == 150, "complete fixed150 required")
    return result


def probability_summary(records):
    joint, conditional, p_u = [], [], []
    for row in records.values():
        for name, dest in [("joint_probabilities", joint), ("probabilities", conditional)]:
            p = row[name][row["target"]]
            require(math.isfinite(p) and 0 < p <= 1, "finite positive gold probability required")
            dest.append(-math.log(p))
        p_u.append(row["p_insufficient"])
    return {
        "joint_nll": math.fsum(joint) / len(joint),
        "conditional_nll": math.fsum(conditional) / len(conditional),
        "mean_p_U": math.fsum(p_u) / len(p_u),
    }


def compare_records(parent, candidate):
    require(parent.keys() == candidate.keys() and len(parent) == 150, "same150 paired IDs")
    for item_id in parent:
        require(
            all(
                parent[item_id][k] == candidate[item_id][k]
                for k in ["id", "family_hash", "pool", "target", "n_options"]
            ),
            "paired provenance changed",
        )
    return sum(row["prediction"] == row["target"] for row in candidate.values())


def group_scores(records, samples):
    groups = {}
    for item_id, row in records.items():
        words = set(samples[item_id]["questions"][0]["instruction"].lower().split())
        kind = (
            "temporal"
            if words & {"before", "after", "when", "while"}
            else "motion"
            if words & {"moving", "stationary"}
            else "existence"
        )
        group = groups.setdefault(kind, {"count": 0, "correct": 0})
        group["count"] += 1
        group["correct"] += int(row["prediction"] == row["target"])
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=list(range(10, 101, 10)), required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--decision-sha256", required=True)
    args = parser.parse_args(argv)
    tag = f"step{args.step:04d}"
    summary_path = ROOT / f"reports/s4-program-stratified-video150-{tag}.json"
    if not args.execute:
        summary_path = summary_path.with_name(f"s4-program-stratified-video150-cpu-{tag}.json")
    require(not summary_path.exists(), "immutable video150 summary exists")
    report = {
        "status": "failed",
        "exit_code": 1,
        "stage_gate": False,
        "results": {},
        "step": args.step,
        "gpu_requested": args.execute,
        "budget_seconds": 180,
        "parent_quality_inherited": False,
        "cost": None,
        "historical_tensor_parity": "not_recorded_in_cached_baseline",
        "historical_token_id_parity": "not_recorded_in_cached_baseline",
    }
    begun = time.monotonic()
    try:
        trained_path = ROOT / f"reports/s4-program-stratified-training-{tag}.json"
        launch_path = ROOT / f"reports/s4-program-stratified-launch-{tag}.json"
        snapshot_path = ROOT / f"reports/s4-program-stratified-runtime-{tag}.json"
        dev50_path = ROOT / f"reports/s4-program-stratified-dev50-{tag}.json"
        trained, launch, dev50 = read(trained_path), read(launch_path), read(dev50_path)
        require(
            trained["status"] == launch["status"] == ("completed" if args.step == 100 else "paused")
            and trained["exit_code"] == launch["exit_code"] == 0
            and trained["steps_completed"] == args.step
            and trained["data_sha256"] == TRAIN_DATA_SHA,
            "actual paused fixed-course checkpoint required",
        )
        require(
            launch["config_sha256"] == CONFIG_SHA
            and launch["prepared_report_sha256"] == PREPARED_SHA
            and launch["source_commit"] == "7b75f9f"
            and launch["exposure_binding"]["final_report"]["sha256"] == sha(trained_path)
            and launch["exposure_binding"]["runtime_report_snapshot"]["sha256"]
            == sha(snapshot_path),
            "actual launch/config/snapshot bindings",
        )
        raw = dict(trained)
        require(
            raw.pop("allocation_ledger_check") == {"ledger_rows": 1200, "verified_train": 1200}
            and raw == read(snapshot_path),
            "snapshot differs from final report",
        )
        require(
            trained["init_adapter"]["weights_sha256"] == PARENT_SHA
            and trained["config"]["rank"] == 16
            and trained["config"]["alpha"] == 32,
            "same f2db rank16 trajectory",
        )
        require(
            dev50["status"] == "completed"
            and dev50["exit_code"] == 0
            and dev50["step"] == args.step
            and dev50["adapter_sha256"] == trained["adapter_sha256"]
            and dev50["training_report_sha256"] == sha(trained_path),
            "DEV50 must complete before wide screen",
        )
        require(sha(args.decision) == args.decision_sha256, "root review SHA")
        validate_review(
            read(args.decision),
            args.step,
            sha(trained_path),
            sha(dev50_path),
            trained["adapter_sha256"],
        )
        full_policy = launch.get("policy_family") == "monitored_full_course"
        window_name = (
            "s4-program-stratified-full-window-001.json"
            if full_policy
            else "s4-program-stratified-window-001.json"
        )
        window = read(ROOT / "reports" / window_name)
        if full_policy:
            require(
                window.get("policy_family") == "monitored_full_course"
                and window.get("target_step") == 100
                and window.get("decision_sha256") == launch["decision_sha256"],
                "explicit full-course supersession window required",
            )
        require(window == launch["window"], "original window changed")
        deadline = datetime.fromisoformat(
            window["window_deadline_utc"].replace("Z", "+00:00")
        ).timestamp()
        require(time.time() + 180 <= deadline, "insufficient original window for video150")
        adapter = Path(trained["adapter_dir"])
        require(
            sha(adapter / "adapter_model.safetensors") == trained["adapter_sha256"],
            "actual candidate SHA",
        )
        for name, expected in SOURCE_FILES.items():
            require(sha(SOURCE / name) == expected, "frozen evaluator dependency changed")
        parent_dir = ROOT / "artifacts/s4-temporal-coverage-video150-001/video"
        parent_report_path, parent_records_path = (
            parent_dir / "report.json",
            parent_dir / "eval/records.jsonl",
        )
        require(
            sha(parent_report_path) == PARENT_REPORT_SHA
            and sha(parent_records_path) == PARENT_RECORDS_SHA,
            "cached actual f2db baseline changed",
        )
        parent_report = read(parent_report_path)
        require(
            parent_report["status"] == "passed"
            and parent_report["exit_code"] == 0
            and parent_report["adapter_sha256"] == PARENT_SHA,
            "actual cached parent evaluation",
        )
        command = list(parent_report["command"])
        require(command[1:3] == ["-m", "scripts.s1_dev_eval"], "frozen official command")
        data = Path(command[command.index("--data") + 1])
        require(sha(data) == DEV_DATA_SHA, "fixed official video150 data changed")
        samples = {row["id"]: row for row in map(json.loads, data.read_text().splitlines())}
        require(
            len(samples) == 150 and all(s["meta"]["pool"] == "dev" for s in samples.values()),
            "DEV150 only",
        )
        parent = original_records(parent_records_path, samples)
        require(
            sum(row["prediction"] == row["target"] for row in parent.values()) == 116,
            "cached parent116",
        )
        parent_probability = probability_summary(parent)
        out = ROOT / f"artifacts/s4-program-stratified-video150-{tag}"
        stage = out / "video"
        require(not out.exists(), "evaluation output exists")
        for flag, value in [
            ("--adapter", str(adapter)),
            ("--output-dir", str(stage / "eval")),
            ("--report", str(stage / "report.json")),
        ]:
            command[command.index(flag) + 1] = value
        require(
            command[command.index("--passes") + 1] == "1"
            and command[command.index("--video-frames") + 1] == "16",
            "frozen one-pass16-frame evaluation",
        )
        report.update(
            status="cpu_prepared_only",
            exit_code=0,
            adapter_sha256=trained["adapter_sha256"],
            training_report_sha256=sha(trained_path),
            decision_sha256=args.decision_sha256,
            launch_report_sha256=sha(launch_path),
            dev50_report_sha256=sha(dev50_path),
            source_commit="74e96e2",
            source_sha256=SOURCE_FILES,
            command=command,
            parent_report_sha256=PARENT_REPORT_SHA,
            parent_records_sha256=PARENT_RECORDS_SHA,
            parent_probability_metrics=parent_report["metrics"],
            parent_nll=parent_probability,
            parent_groups=group_scores(parent, samples),
            grouping="legacy whitespace-keyword heuristic, not official program strata",
            wrapper_sha256=sha(Path(__file__)),
        )
        if args.execute:
            stage.mkdir(parents=True, exist_ok=False)
            env = dict(
                os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
            )
            with (stage / "run.log").open("x") as log:
                result = subprocess.run(
                    command, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180
                )
            require(result.returncode == 0, "official video150 execution failed")
            actual = read(stage / "report.json")
            records_path = stage / "eval/records.jsonl"
            require(
                actual["status"] == "passed"
                and actual["exit_code"] == 0
                and actual["adapter_sha256"] == trained["adapter_sha256"]
                and actual["data_sha256"] == DEV_DATA_SHA
                and actual["sidecar_sha256"] == sha(records_path),
                "actual evaluation output identity",
            )
            for key in [
                "model_revision",
                "video_num_frames",
                "passes",
                "script_sha256",
                "image_max_soft_tokens",
            ]:
                require(
                    actual[key] == parent_report[key], f"frozen evaluation setting changed: {key}"
                )
            candidate = original_records(records_path, samples)
            correct = compare_records(parent, candidate)
            probability = probability_summary(candidate)
            metrics = actual["metrics"]
            require(
                metrics["count"] == 150 and abs(metrics["accuracy"] - correct / 150) < 1e-12,
                "postprocessing correct count mismatch",
            )
            nonworse = (
                metrics["brier"] <= parent_report["metrics"]["brier"]
                and probability["joint_nll"] <= parent_probability["joint_nll"]
                and probability["conditional_nll"] <= parent_probability["conditional_nll"]
            )
            report.update(
                status="completed",
                exit_code=0,
                paired_ids_and_gold_verified=150,
                probabilities_nonworse=nonworse,
                nonworse_probability_criteria=["brier", "joint_nll", "conditional_nll"],
                basic_retention_passed=correct >= 116 and nonworse,
                root_review_required=True,
                no_automatic_continuation=True,
            )
            report["results"]["video"] = {
                "count": 150,
                "correct": correct,
                "parent_correct": 116,
                "delta_correct": correct - 116,
                "groups": group_scores(candidate, samples),
                "metrics": metrics,
                "nll": probability,
                "report": str(stage / "report.json"),
                "report_sha256": sha(stage / "report.json"),
                "sidecar": str(records_path),
                "sidecar_sha256": sha(records_path),
                "data_sha256": DEV_DATA_SHA,
            }
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        report["wall_seconds"] = time.monotonic() - begun
        with summary_path.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if args.execute and "out" in locals() and out.exists():
            (out / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
