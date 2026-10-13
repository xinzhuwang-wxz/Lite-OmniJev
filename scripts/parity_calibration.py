"""Run program calibration or preregistered shadow confirmation without changing stage gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import httpx

from data.parity_fixture import load_calibration as load_fixture
from data.schema import (
    HardTarget,
    InsufficientTarget,
    Sample,
)
from eval.parity_shadow import assess, load_registered_confirmation
from eval.program_truth import truth_metrics as truth_metrics
from scripts.check_tokens import OfficialTokenizerAdapter
from scripts.p0_parity import (
    compare_rows,
    digest,
    file_hash,
    hf_reference,
    validate_reference,
    vllm_rows,
)
from train.trainer import SmokeConfig, code_revision, prepare_examples, verified_assets


def load_calibration(directory: Path) -> list[Sample]:
    manifest = json.loads((directory / "manifest.json").read_text())
    entry = manifest["files"]["calibration.jsonl"]
    payload = (directory / "calibration.jsonl").read_bytes()
    if entry["sha256"] != hashlib.sha256(payload).hexdigest() or entry["pool"] != "calibration":
        raise ValueError("calibration manifest mismatch")
    samples = load_fixture(directory / "calibration.jsonl")
    if entry["rows"] != 100:
        raise ValueError("exactly 100 calibration cases required")
    for sample in samples:
        if not isinstance(sample.questions[0].target, (HardTarget, InsufficientTarget)):
            raise ValueError("program hard truth or explicit insufficient target required")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("hf-reference", "compare-vllm"))
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--server-record", type=Path)
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8002")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--cohort", choices=("calibration", "confirmation"), default="calibration")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--registration", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError("choose a new report path")
    start = time.monotonic()
    report: dict[str, Any] = {
        "action": args.action,
        "status": "blocked",
        "exit_code": 1,
        "scope": "independent program calibration only; original gate unchanged; not P0 acceptance",
        "code_revision": code_revision(),
        "cost": None,
    }
    try:
        policy = None
        if args.cohort == "confirmation":
            if args.policy is None or args.registration is None:
                raise ValueError("confirmation requires preregistered shadow policy")
            samples, policy = load_registered_confirmation(
                args.fixture_dir, args.policy, args.registration
            )
            if args.dtype != policy["dtype"]:
                raise ValueError("confirmation dtype differs from frozen policy")
            registered_assets = json.loads(args.registration.read_text())
            if args.model_dir.name != registered_assets.get("model_revision") or (
                args.adapter is not None
                and file_hash(args.adapter / "adapter_model.safetensors")
                != registered_assets.get("adapter_sha256")
            ):
                raise ValueError("confirmation model or adapter differs from registration")
            report["scope"] = "preregistered program confirmation; shadow only, no stage acceptance"
            report["policy_sha256"] = file_hash(args.policy)
            report["registration_sha256"] = file_hash(args.registration)
        else:
            if args.policy is not None or args.registration is not None:
                raise ValueError("shadow registration is only valid for confirmation")
            samples = load_calibration(args.fixture_dir)
        config = SmokeConfig()
        tokenizer = OfficialTokenizerAdapter(args.model_dir)
        examples = prepare_examples(samples, tokenizer, config.max_length)
        if any(len(e["input_ids"]) >= config.max_length for e in examples):
            raise ValueError("prompt must leave generation room")
        assets = verified_assets(args.model_dir, args.asset_manifest)
        provenance = {
            "purpose": "independent-" + args.cohort,
            "dtype": args.dtype,
            "thinking": False,
            "model_revision": args.model_dir.name,
            "asset_hashes": assets,
            "manifest_sha256": file_hash(args.asset_manifest),
            "fixture_manifest_sha256": file_hash(args.fixture_dir / "manifest.json"),
            "data_file_sha256": file_hash(args.fixture_dir / (args.cohort + ".jsonl")),
            "inputs_sha256": digest(examples),
            "template_sha256": hashlib.sha256(tokenizer.template.encode()).hexdigest(),
            "reference_settings": {"seed": config.seed, "max_length": config.max_length},
            "mode": "adapter" if args.adapter else "base",
            "adapter_hashes": {
                n: file_hash(args.adapter / n)
                for n in ("adapter_config.json", "adapter_model.safetensors")
            }
            if args.adapter
            else None,
            "implementation_hashes": {
                n: file_hash(Path(n))
                for n in (
                    "scripts/parity_calibration.py",
                    "data/parity_fixture.py",
                    "scripts/p0_parity.py",
                    "train/trainer.py",
                    "train/template.py",
                    "train/readout.py",
                    "scripts/check_tokens.py",
                    "eval/program_truth.py",
                    "eval/parity_shadow.py",
                )
            },
        }
        if policy is not None:
            provenance["policy_sha256"] = report["policy_sha256"]
            provenance["registration_sha256"] = report["registration_sha256"]
        report["provenance"] = provenance
        if args.action == "hf-reference":
            report["rows"] = hf_reference(
                config, args.model_dir, args.adapter, examples, report, dtype=args.dtype
            )
            report.update(status="reference-ready", exit_code=0)
        else:
            if args.reference is None or args.server_record is None:
                raise ValueError("reference and attested server record required")
            reference = json.loads(args.reference.read_text())
            left = validate_reference(reference, provenance, examples)
            server = json.loads(args.server_record.read_text())
            if (
                server.get("provenance") != provenance
                or server.get("dtype") != args.dtype
                or server.get("logprobs_mode") != "raw_logprobs"
                or not server.get("launch_command")
                or not server.get("served_model")
                or (policy is not None and server.get("attention_backend") != policy["backend"])
            ):
                raise ValueError("server attestation mismatch")
            with httpx.Client(timeout=120) as client:
                right = vllm_rows(client, args.upstream_url, server["served_model"], examples)
            report["rows"] = right
            report.update(compare_rows(left, right))
            score_deltas = []
            for sample, a, b in zip(samples, left, right, strict=True):
                if sample.questions[0].type == "score":
                    delta = abs(
                        math.fsum(
                            i * (x - y)
                            for i, (x, y) in enumerate(
                                zip(a["conditional"], b["conditional"], strict=True)
                            )
                        )
                    )
                    score_deltas.append(
                        {
                            "id_sha256": a["id_sha256"],
                            "expected_score_delta": delta,
                            "normalized_delta": delta / (len(a["conditional"]) - 1),
                            "reference_p_U": a["p_U"],
                            "candidate_p_U": b["p_U"],
                        }
                    )
            report["score_deltas"] = score_deltas
            report["reference_sha256"] = file_hash(args.reference)
            report["server_record_sha256"] = file_hash(args.server_record)
            report["server_identity_verification"] = "operator-attested; not remote attestation"
            if policy is not None:
                report["shadow_assessment"] = assess(samples, reference, report, policy)
        report["truth_metrics"] = truth_metrics(samples, report["rows"])
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="blocked", exit_code=1, error_type=type(exc).__name__)
    report["duration_seconds"] = time.monotonic() - start
    with args.report.open("x") as output:
        output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "exit_code", "action")}))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
