"""Own-baseline screen for the non-parity rank64 initializer; CPU default, root launches GPU."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = Path("/root/autodl-tmp/s4-motion-code-142e744")
RUNNER_SHA = "e392f10b7738f9cf776087dfb7cd40b35377d974f986fc5cd9ad47bb77b51246"
INIT_SHA = "62920299c7a1ac33b2a874b586a9d7f3ea3eaa4e38198bc20388642482318fc6"
CONFIG_SHA = "8d79bb86dc522bc6da2b3de170dc704e51b60e2d2518582b07e8ffbfcdd6bc0a"
EXPANSION_SHA = "05e97013ac5d474866fdf208611bfef618bdfb65f5770c18f2924d21d37b2be7"
PARENT_SHA = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def paired_metrics(reference, candidate, expected_reference_sha, expected_candidate_sha):
    old, new = read(reference / "receipt.json"), read(candidate / "receipt.json")
    for receipt, expected in [(old, expected_reference_sha), (new, expected_candidate_sha)]:
        require(
            receipt["status"] == "completed" and receipt["exit_code"] == 0,
            "completed actual evaluation required",
        )
        require(receipt["identity"]["adapter_sha256"] == expected, "evaluation adapter changed")
        require(receipt["metrics"]["count"] == 50, "fixed DEV50 required")
    ignored = {"adapter_sha256", "adapter_config_sha256"}
    require(
        {k: v for k, v in old["identity"].items() if k not in ignored}
        == {k: v for k, v in new["identity"].items() if k not in ignored},
        "evaluation input/environment identity changed",
    )
    rows = [
        [json.loads(line) for line in (folder / "records.jsonl").read_text().splitlines()]
        for folder in [reference, candidate]
    ]
    require(len(rows[0]) == len(rows[1]) == 50, "fifty paired records required")
    require(len({row["id"] for row in rows[0]}) == 50, "unique DEV50 IDs required")
    for left, right in zip(*rows, strict=True):
        require(
            all(
                left[k] == right[k]
                for k in ["id", "input_sha256", "input_summary_sha256", "answer_token_ids"]
            ),
            "complete evaluation inputs/mapping changed",
        )
    return old["metrics"], new["metrics"]


def shortlist(metrics):
    return (
        metrics["count"] == 50
        and metrics["accuracy"] >= 0.64
        and metrics["brier"] <= 0.4674860199909794
    )


def run_evaluator(adapter, adapter_sha, output, execute):
    runner = SOURCE / ".scratch/s4/evaluate-video-motion-coverage.py"
    require(sha(runner) == RUNNER_SHA, "official evaluator changed")
    # One direct child owns GPU allocations; timeout kills that child. No GPU calls in parent.
    code = """import importlib.util,sys
from pathlib import Path
runner,adapter,digest,out,execute=sys.argv[1:]
spec=importlib.util.spec_from_file_location("motion",runner)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ADAPTER=Path(adapter)
module.ADAPTER_SHA=digest
stages = ([] if (Path(out)/"preparation.json").exists() else [[]])
if execute=="1": stages.append(["--execute","--policy","uniform"])
for flags in stages:
    sys.argv=[runner,"--output",out,*flags]
    if module.main()!=0: raise SystemExit(1)
"""
    command = [
        sys.executable,
        "-c",
        code,
        str(runner),
        str(adapter),
        adapter_sha,
        str(output),
        "1" if execute else "0",
    ]
    result = subprocess.run(command, cwd=SOURCE, timeout=180, check=False)
    require(result.returncode == 0, "official evaluator failed")


def validate_failed_control(control):
    require(
        control.get("status") == "completed"
        and control.get("exit_code") == 0
        and control.get("gpu_requested") is True
        and control.get("same_process") is True,
        "actual controlled receipt required",
    )
    require(
        control.get("expanded_weights_sha256") == INIT_SHA
        and control.get("expanded_config_sha256") == CONFIG_SHA
        and control.get("source_weights_sha256") == PARENT_SHA
        and control.get("expansion_report_sha256") == EXPANSION_SHA,
        "controlled receipt initializer/source changed",
    )
    require(
        control.get("parity_validated") is False
        and control.get("verified_complete_input_mapping") == 12,
        "retain failed parity and complete twelve-input evidence",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--control-receipt-sha256", required=True)
    args = parser.parse_args(argv)
    destination = ROOT / "reports/s4-rank64-initializer-dev50-001.json"
    require(not destination.exists(), "baseline receipt already exists")
    started = time.monotonic()
    report = {
        "status": "failed",
        "exit_code": 1,
        "stage_gate": False,
        "training_authorized": False,
        "parity_validated": False,
        "parent_quality_inherited": False,
        "gpu_requested": args.execute,
        "budget_seconds": 180,
        "cost_usd": None,
        "scope": "Independent rank64 initializer own DEV50 baseline; root review required",
    }
    try:
        policy = ROOT / "reports/explore/rank64-independent-candidate-policy-001.md"
        require(
            sha(policy) == "0cc0d96a2f77edc857dc2904cff78af95d3c1fd5daa56110637bb0d695292ff1",
            "independent candidate policy changed",
        )
        report["policy_sha256"] = sha(policy)
        initializer = ROOT / "artifacts/s4-rank64-initializer-001"
        require(sha(initializer / "adapter_model.safetensors") == INIT_SHA, "initializer weights")
        require(sha(initializer / "adapter_config.json") == CONFIG_SHA, "initializer config")
        require(sha(initializer / "rank-expansion.json") == EXPANSION_SHA, "expansion receipt")
        control_path = ROOT / "artifacts/s4-rank-runtime-control-001/receipt.json"
        require(sha(control_path) == args.control_receipt_sha256, "controlled failure receipt SHA")
        control = read(control_path)
        validate_failed_control(control)
        output = ROOT / "artifacts/s4-rank64-initializer-dev50-001"
        run_evaluator(initializer, INIT_SHA, output, args.execute)
        report.update(
            status="cpu_prepared_only",
            exit_code=0,
            initializer_sha256=INIT_SHA,
            initializer_config_sha256=CONFIG_SHA,
            expansion_report_sha256=EXPANSION_SHA,
            controlled_failure_receipt_sha256=args.control_receipt_sha256,
            wrapper_sha256=sha(Path(__file__)),
        )
        if args.execute:
            parent_path = ROOT / "artifacts/s4-temporal-coverage-dev50-001/uniform"
            candidate_path = output / "uniform"
            parent, candidate = paired_metrics(parent_path, candidate_path, PARENT_SHA, INIT_SHA)
            require(
                parent["accuracy"] == 0.66 and abs(parent["brier"] - 0.44748601999097937) < 1e-12,
                "fixed parent baseline changed",
            )
            report.update(
                status="completed",
                candidate=candidate,
                parent=parent,
                paired_input_hashes_verified=50,
                shortlist_passed=shortlist(candidate),
                candidate_receipt_sha256=sha(candidate_path / "receipt.json"),
                candidate_records_sha256=sha(candidate_path / "records.jsonl"),
                parent_receipt_sha256=sha(parent_path / "receipt.json"),
                next_step="root review only; no automatic training",
            )
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        report["wall_seconds"] = time.monotonic() - started
        # CPU preparation is separate so it does not consume the actual baseline receipt path.
        if not args.execute:
            destination = destination.with_name("s4-rank64-initializer-dev50-cpu-001.json")
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
