"""User-authorized full fixed1200-draw CE course, with serial checkpoint monitoring."""

import argparse
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HELPER = Path(__file__).with_name("launch-program-stratified-pool.py")
spec = importlib.util.spec_from_file_location("course_helpers", HELPER)
assert spec and spec.loader
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
HELPER_SHA = "8ff309225773413e3c2a8b0494c891db77216229b36ee787c125d521bab3a4bc"
ROOT, SOURCE = h.ROOT, h.SOURCE
PREPARED_SHA = "ac2f17fc3f5417e93e04d15292763384858f5e3ba1c4de533ca6f5e57d0aaf99"
CONFIG_SHA = "326b406eb23ce47c4854afe64d535a95ad91be94a68844109329e43872c6aaea"
DATA_SHA = "795f69fe9474523009bd953a0436bc505e85efdd4fb0f5e7c41e4d86c3bb028f"
LEDGER_SHA = "b5172781520546ef974a57825fa8ce98033b20e6d8a66daf8c1f550efc186322"
SEGMENTS = [20, 40, 60, 80, 100]


def segment_ends(start, stop_after20=False):
    h.require(
        type(start) is int and start in (10, 20, 40, 60, 80, 100),
        "resume from reviewed segment boundary",
    )
    return [end for end in SEGMENTS if end > start and (not stop_after20 or end <= 20)]


def validate_policy(policy, now):
    expected = {
        "policy_family": "monitored_full_course",
        "decision": "run_monitored_full_course",
        "user_authorized_supersession": True,
        "prepared_report_sha256": PREPARED_SHA,
        "config_sha256": CONFIG_SHA,
        "initializer_sha256": h.INIT_SHA,
        "target_step": 100,
    }
    h.require(
        all(policy.get(k) == v for k, v in expected.items()),
        "explicit full-course user supersession required",
    )
    beginning = h.utc_seconds(policy["window_start_utc"])
    deadline = h.utc_seconds(policy["window_deadline_utc"])
    h.require(
        deadline - beginning == 5400 and beginning <= now < deadline,
        "fixed90-minute inclusive window",
    )
    h.require(
        isinstance(policy.get("adr_sha256"), str) and len(policy["adr_sha256"]) == 64,
        "ADR0023 decision binding required",
    )
    return deadline


def validate_resume(raw, metadata, start):
    expected_order = list(range(1200))
    random.Random(42).shuffle(expected_order)
    h.require(
        raw["status"] == ("completed" if start == 100 else "paused")
        and raw["exit_code"] == 0
        and raw["steps_completed"] == start
        and raw["data_sha256"] == DATA_SHA
        and raw["init_adapter"]["weights_sha256"] == h.INIT_SHA,
        "actual successful same-course trajectory required",
    )
    h.require(
        metadata["step"] == start
        and metadata["config"] == raw["config"]
        and metadata["data_sha256"] == DATA_SHA
        and metadata["order"] == expected_order
        and metadata["losses"] == raw["losses"],
        "full AdamW/RNG checkpoint identity mismatch",
    )


def catastrophic(current, checked_best):
    """Pause only jointly large accuracy/probability degradation; not one-to-three-item noise."""
    for record in (current, checked_best):
        metric = record["candidate"]
        h.require(
            metric["count"] == 50 and all(math.isfinite(metric[k]) for k in ["accuracy", "brier"]),
            "nonfinite or incomplete checked evaluation",
        )
    lost = round(checked_best["candidate"]["accuracy"] * 50) - round(
        current["candidate"]["accuracy"] * 50
    )
    probability_worse = current["candidate"]["brier"] > checked_best["candidate"]["brier"] + 0.02
    old_nll = checked_best.get("candidate_probability_diagnostics", {}).get("joint_nll")
    new_nll = current.get("candidate_probability_diagnostics", {}).get("joint_nll")
    if old_nll is not None and new_nll is not None:
        h.require(math.isfinite(old_nll) and math.isfinite(new_nll), "nonfinite checked NLL")
        probability_worse = probability_worse or new_nll > old_nll + 0.10
    return lost >= 5 and probability_worse


def checked_screen(path, step, adapter_sha):
    screen = h.read(path)
    h.require(
        screen["status"] == "completed"
        and screen["exit_code"] == 0
        and screen["step"] == step
        and screen["paired_input_hashes_verified"] == 50
        and screen["adapter_sha256"] == adapter_sha,
        "actual checked same-checkpoint DEV50 required",
    )
    return screen


def monitor_process(command, log_path, env, timeout_seconds, raw_path=None, monitor_path=None):
    """One process group, bounded polling; stop the entire group on a hard failure."""
    begun = time.monotonic()
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            cwd=SOURCE,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                remaining = timeout_seconds - (time.monotonic() - begun)
                if remaining <= 0:
                    raise TimeoutError("monitored process hard timeout")
                try:
                    return process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    if raw_path is not None and raw_path.exists():
                        try:
                            raw = h.read(raw_path)
                        except (OSError, json.JSONDecodeError):
                            continue
                        losses = raw.get("losses", [])
                        h.require(
                            all(math.isfinite(value) for value in losses), "nonfinite runtime loss"
                        )
                        observation = {
                            "elapsed_seconds": time.monotonic() - begun,
                            "steps_completed": raw.get("steps_completed"),
                            "last_loss": losses[-1] if losses else None,
                            "last_step_metrics": raw.get("step_metrics", [])[-1:],
                        }
                        if monitor_path is not None:
                            with monitor_path.open("a") as stream:
                                stream.write(json.dumps(observation, allow_nan=False) + "\n")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-after20", action="store_true")
    parser.add_argument("--resume-after-analysis", action="store_true")
    args = parser.parse_args(argv)
    h.require(
        h.sha(HELPER) == HELPER_SHA and h.sha(args.decision) == args.decision_sha256,
        "bound helper/decision changed",
    )
    policy = h.read(args.decision)
    deadline = validate_policy(policy, time.time())
    adr_path = ROOT / "docs/adr/0023-measured-quality-stage-progression.md"
    h.require(h.sha(adr_path) == policy["adr_sha256"], "actual ADR0023 file binding changed")
    prepared_dir = ROOT / "artifacts/s4-program-stratified-pool-001"
    prepared_path = prepared_dir / "report.json"
    config_path, data_path, ledger_path = [
        prepared_dir / name for name in ["config-ce.json", "train.jsonl", "allocation-ledger.jsonl"]
    ]
    for path, digest in [
        (prepared_path, PREPARED_SHA),
        (config_path, CONFIG_SHA),
        (data_path, DATA_SHA),
        (ledger_path, LEDGER_SHA),
    ]:
        h.require(h.sha(path) == digest, "fixed full-course artifact changed")
    h.validate_prepared(h.read(prepared_path), h.read(config_path))
    h.verify_projection()
    initializer = ROOT / "artifacts/s4-temporal-coverage-train-001/adapter"
    h.require(
        h.sha(initializer / "adapter_model.safetensors") == h.INIT_SHA
        and h.sha(initializer / "adapter_config.json") == h.INIT_CONFIG_SHA,
        "same initializer lineage",
    )
    exposure_path = Path(policy["latest_exposure_report"])
    h.require(h.sha(exposure_path) == policy["latest_exposure_report_sha256"], "exposure binding")
    exposure = h.read(exposure_path)
    h.require(
        exposure["exit_code"] == 0 and not exposure["unknown_update_bearing_inputs"],
        "unknown exposure inputs",
    )
    output = ROOT / "artifacts/s4-program-stratified-pool-train-001"
    raw_path = output / "run.json"
    current = h.read(raw_path)
    start = current["steps_completed"]
    ends = segment_ends(start, args.stop_after20)
    previous_files = h.paths(ROOT, start)
    previous_launch = h.read(previous_files["launch"])
    h.require(
        h.sha(raw_path)
        == previous_launch["exposure_binding"]["runtime_report_snapshot"]["sha256"]
        == h.sha(previous_files["runtime"]),
        "live checkpoint differs from immutable prior segment",
    )
    sys.path.insert(0, str(SOURCE))
    from train.checkpoint import load_checkpoint_metadata

    checkpoint = Path(current["checkpoint"])
    h.require(
        checkpoint.resolve().parent == (output / "checkpoints").resolve(),
        "full local checkpoint required",
    )
    validate_resume(current, load_checkpoint_metadata(checkpoint), start)
    first_screen = checked_screen(previous_files["dev50"], start, current["adapter_sha256"])
    window_path = ROOT / "reports/s4-program-stratified-full-window-001.json"
    window = {
        "policy_family": "monitored_full_course",
        "window_start_utc": policy["window_start_utc"],
        "window_deadline_utc": policy["window_deadline_utc"],
        "decision_sha256": args.decision_sha256,
        "prepared_report_sha256": PREPARED_SHA,
        "target_step": 100,
        "adr_sha256": policy["adr_sha256"],
    }
    if window_path.exists():
        h.require(h.read(window_path) == window, "full-course window cannot slide or change policy")
    state_path = ROOT / "reports/s4-program-stratified-full-controller-001.json"
    state = (
        h.read(state_path)
        if state_path.exists()
        else {
            "status": "cpu_ready",
            "exit_code": 0,
            "policy_family": "monitored_full_course",
            "decision_sha256": args.decision_sha256,
            "segments": [],
            "best_checked": first_screen,
            "formal_quality_gate": "not_passed_by_this_controller",
            "full_course_scope": "all1200 fixed course draws; not all S4 data",
        }
    )
    h.require(state["decision_sha256"] == args.decision_sha256, "controller identity changed")
    h.require(
        state.get("status") != "paused_for_root_analysis" or args.resume_after_analysis,
        "analysis pause requires explicit root continuation flag",
    )
    env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if not args.execute:
        print(
            json.dumps(
                {"status": "cpu_ready", "start_step": start, "segment_ends": ends, "window": window}
            )
        )
        return 0
    if not window_path.exists():
        with window_path.open("x") as stream:
            json.dump(window, stream, indent=2)
    env.update(
        SWANLAB_API_KEY=Path("/root/.config/lite-omnijev/swanlab.key").read_text().strip(),
        LITE_OMNIJEV_SWANLAB_MODE="online",
    )
    try:
        for end in ends:
            if time.time() + 500 + 120 > deadline:
                state.update(status="budget_exhausted", exit_code=0)
                break
            files = h.paths(ROOT, end)
            h.require(
                not any(files[k].exists() for k in ["launch", "training", "runtime"]),
                "immutable segment already exists",
            )
            current = h.read(raw_path)
            checkpoint = Path(current["checkpoint"])
            validate_resume(current, load_checkpoint_metadata(checkpoint), start)
            common = [
                "--config",
                str(config_path),
                "--model-dir",
                "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2",
                "--asset-manifest",
                "/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json",
                "--data",
                str(data_path),
                "--allocation-ledger",
                str(ledger_path),
                "--output-dir",
                str(output),
                "--init-adapter",
                str(initializer),
                "--init-adapter-sha256",
                h.INIT_SHA,
                "--resume-from",
                str(checkpoint),
            ]
            if not state.get("preflight_completed"):
                preflight = [
                    h.PYTHON,
                    "-m",
                    "scripts.train_s1",
                    "preflight",
                    *common,
                    "--report",
                    str(files["preflight"]),
                ]
                code = monitor_process(preflight, prepared_dir / "full-preflight.log", env, 180)
                h.require(
                    code == 0 and h.read(files["preflight"])["status"] == "ready",
                    "first full-course preflight failed",
                )
                state["preflight_completed"] = h.binding(files["preflight"])
            if time.time() + 500 + 120 > deadline:
                state.update(status="budget_exhausted", exit_code=0)
                break
            h.require(
                not subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
                ).strip(),
                "GPU busy",
            )
            command = [
                "timeout",
                "--signal=INT",
                "--kill-after=20s",
                "480",
                h.PYTHON,
                "-m",
                "scripts.train_s1",
                "run",
                *common,
                "--invocation-steps",
                str(end - start),
                "--report",
                str(files["training"]),
            ]
            receipt = {
                "status": "running",
                "exit_code": 1,
                "policy_family": "monitored_full_course",
                "window": window,
                "decision_sha256": args.decision_sha256,
                "prepared_report_sha256": PREPARED_SHA,
                "config_sha256": CONFIG_SHA,
                "source_commit": "7b75f9f",
                "source_archive_sha256": h.ARCHIVE_SHA,
                "selected_init_adapter": current["init_adapter"],
                "start_step": start,
                "end_step_target": end,
                "command": command,
                "stage_gate": False,
                "optimizer_boundary": "restore full AdamW/RNG checkpoint",
                "wrapper_sha256": h.sha(Path(__file__)),
                "cost": None,
            }
            files["launch"].write_text(json.dumps(receipt, indent=2) + "\n")
            code = monitor_process(
                command,
                prepared_dir / f"full-train-step{end:04d}.log",
                env,
                500,
                raw_path,
                prepared_dir / "full-monitor.jsonl",
            )
            with files["runtime"].open("xb") as stream:
                stream.write(raw_path.read_bytes())
            raw, final = h.read(files["runtime"]), h.read(files["training"])
            h.require(
                final.pop("allocation_ledger_check")
                == {"ledger_rows": 1200, "verified_train": 1200}
                and final == raw,
                "actual final/runtime mismatch",
            )
            receipt.update(
                status=raw["status"],
                exit_code=code,
                exposure_binding={
                    "base_exposure_report": h.binding(exposure_path),
                    "prepared_report": h.binding(prepared_path),
                    "config_file": h.binding(config_path),
                    "runtime_data": h.binding(output / "data.jsonl"),
                    "final_report": h.binding(files["training"]),
                    "runtime_report_snapshot": h.binding(files["runtime"]),
                    "runtime_report_path": str(raw_path),
                    "source_files": [
                        h.binding(SOURCE / n)
                        for n in ["train/trainer.py", "train/checkpoint.py", "scripts/train_s1.py"]
                    ],
                    "start_step": start,
                    "end_step": raw.get("steps_completed", 0),
                    "resume_from": str(checkpoint),
                },
            )
            files["launch"].write_text(json.dumps(receipt, indent=2) + "\n")
            h.require(
                code == 0
                and raw["status"] == ("completed" if end == 100 else "paused")
                and raw["steps_completed"] == end
                and raw["invocation"]["start_step"] == start
                and raw["invocation"]["next_draw"] == end * 12
                and raw["losses"][:start] == current["losses"],
                "incomplete/changed full-course segment",
            )
            h.require(
                h.sha(Path(raw["adapter_dir"]) / "adapter_model.safetensors")
                == raw["adapter_sha256"],
                "actual saved adapter changed",
            )
            evaluator = Path(__file__).with_name("evaluate-program-stratified-dev50.py")
            eval_command = [h.PYTHON, str(evaluator), "--step", str(end), "--execute"]
            code = monitor_process(
                eval_command, prepared_dir / f"full-dev50-step{end:04d}.log", env, 150
            )
            h.require(code == 0, "actual full-course DEV50 failed")
            screen = checked_screen(files["dev50"], end, raw["adapter_sha256"])
            state["segments"].append(
                {
                    "start_step": start,
                    "end_step": end,
                    "launch": h.binding(files["launch"]),
                    "screen": h.binding(files["dev50"]),
                    "adapter_sha256": raw["adapter_sha256"],
                }
            )
            state["steps_completed"] = end
            if catastrophic(screen, state["best_checked"]):
                state.update(
                    status="paused_for_root_analysis",
                    exit_code=2,
                    reason="atleast5/50 loss with material probability degradation",
                )
                break
            if (screen["candidate"]["accuracy"], -screen["candidate"]["brier"]) > (
                state["best_checked"]["candidate"]["accuracy"],
                -state["best_checked"]["candidate"]["brier"],
            ):
                state["best_checked"] = screen
            start = end
            state.update(
                status="course_completed" if end == 100 else "monitored",
                exit_code=0,
                steps_completed=end,
            )
            state_path.write_text(json.dumps(state, indent=2) + "\n")
        if args.stop_after20 and state["status"] == "monitored":
            state["status"] = "stopped_after20_for_initial_root_observation"
    except Exception as exc:
        if "receipt" in locals() and receipt.get("status") == "running":
            receipt.update(
                status="failed",
                exit_code=1,
                error=f"{type(exc).__name__}: {exc}",
                possible_partial_step=True,
            )
            files["launch"].write_text(json.dumps(receipt, indent=2) + "\n")
        state.update(
            status="paused_for_root_analysis", exit_code=1, error=f"{type(exc).__name__}: {exc}"
        )
    finally:
        state["exposure_reconciliation_pending"] = True
        state["updated_at_utc"] = datetime.now().astimezone().isoformat()
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state))
    return state["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
