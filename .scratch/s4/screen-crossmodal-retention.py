"""Fixed public-dev reference-retention screening, never stage acceptance.

Without --execute, only prepare a receipt. --execute defaults to video only;
--through explicitly extends the frozen ordered budget. --resume continues only
hash-verified passed stages from the same candidate and source receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample, answer_tokens
from train.readout import read_probabilities

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
BASE = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
ORDER = ("video", "original", "supplemental", "audio", "image", "screenshot", "text")
MINIMUM = dict(zip(ORDER, (102, 35, 40, 37, 40, 50, 49), strict=True))
CROSS = {
    "original": (
        "8099d5c78af26fd0848dd36d06a52b68906e6789cb2c55f7c62d8773485298af",
        "1af868df1d442def73765d5cfda0532a8b86884f441d28f844a5f6ee9e3ad972",
    ),
    "supplemental": (
        "1d2d8e151a3bc282099764a5ce7d04e8190fd212684ca010582511c5b25cfde1",
        "f6a60b8f7a7cf88a303ed6a0765b043cf8c3ccc957eeef03013e96d2133df879",
    ),
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cross_source(name: str) -> tuple[Path, Path]:
    if name == "original":
        data = Path(
            "/root/autodl-tmp/s0-crossmodal-development-001/artifacts/crossmodal-dev-001-a/dev.jsonl"
        )
        return data, data.parent / "media"
    return (
        ROOT / "artifacts/s4-gold-inventory-001/crossmodal-dev-002/dev.jsonl",
        ROOT / "artifacts/s4-gold-inventory-001/crossmodal-dev-source-bundle-002",
    )


def cross_forward(name: str, adapter: Path, out: Path) -> dict:
    # Same fixed first-pilot selection and evaluator; imports/loads CUDA only on execution.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    from scripts import s0_development_baselines as baseline

    data, media = cross_source(name)
    require(sha(data) == CROSS[name][0], "crossmodal source SHA mismatch")
    samples = baseline.load_samples(data)
    baseline.validate_samples(samples, cohort="crossmodal")
    require(all(sample.meta.pool == "dev" for sample in samples), "crossmodal must be dev")
    samples = sorted(
        samples,
        key=lambda sample: hashlib.sha256(("42:s4-crossmodal:" + sample.id).encode()).hexdigest(),
    )[:50]
    selected = hashlib.sha256("\n".join(sample.id for sample in samples).encode()).hexdigest()
    require(len(samples) == 50 and selected == CROSS[name][1], "crossmodal selected ID mismatch")
    processor = baseline.default_processor_factory(BASE)

    class BudgetProcessor:
        def __getattr__(self, key):
            return getattr(processor, key)

        def __call__(self, **kwargs):
            if "images" in kwargs:
                kwargs["images_kwargs"] = {"max_soft_tokens": 1120}
            return processor(**kwargs)

    baseline.configure_torch_runtime(torch)
    model = AutoModelForImageTextToText.from_pretrained(
        str(BASE),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    model.to("cuda")
    try:

        def forward(prepared):
            inputs = baseline.move_inputs_to_cuda(prepared.inputs, torch, torch.bfloat16)
            with torch.inference_mode():
                result = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                logits = result.logits[0, -1].float()
                require(bool(torch.isfinite(logits).all()), "nonfinite crossmodal logits")
                logprobs = logits.log_softmax(-1)
                return {
                    int(i): float(logprobs[int(i)].item())
                    for i in prepared.answer_token_ids.values()
                }

        records = baseline.run_pass(
            samples,
            processor=BudgetProcessor(),
            media_root=media,
            pass_index=1,
            forward=forward,
            records_sidecar=out / "records.jsonl",
        )
        metrics = baseline.score_records(records)
        require(metrics["count"] == 50, "crossmodal result count mismatch")
        return {
            "status": "completed",
            "exit_code": 0,
            "metrics": metrics,
            "data_sha256": sha(data),
            "selected_ids_sha256": selected,
            "records_sha256": sha(out / "records.jsonl"),
            "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": "bfloat16",
            "image_max_soft_tokens": 1120,
            "code_sha256": {p: baseline.sha256_file(Path(p)) for p in baseline.REQUIRED_CODE_FILES},
        }
    finally:
        model = None
        torch.cuda.empty_cache()


def ordinary(name: str, adapter: Path, output: Path, source: dict, command: dict) -> dict:
    data = Path(source["data"]["path"])
    require(sha(data) == source["data"]["sha256"], "fixed dev data SHA mismatch")
    samples = [
        Sample.model_validate_json(line) for line in data.read_text().splitlines() if line.strip()
    ]
    require(all(sample.meta.pool == "dev" for sample in samples), "ordinary screen must be dev")
    require(len(samples) == (150 if name == "video" else 50), "fixed cohort count mismatch")
    by_id = {sample.id: sample for sample in samples}
    require(len(by_id) == len(samples), "duplicate dev IDs")
    argv = list(command["argv"])
    require(argv[1:4] == ["-u", "-m", "scripts.s1_dev_eval"], "unexpected evaluator command")
    for flag, expected in [
        ("--data", str(data)),
        ("--model-dir", str(BASE)),
        ("--passes", "1"),
        ("--video-frames", "16"),
    ]:
        require(argv[argv.index(flag) + 1] == expected, "changed frozen evaluator arguments")
    for flag, value in [
        ("--adapter", str(adapter)),
        ("--output-dir", str(output / "eval")),
        ("--report", str(output / "report.json")),
    ]:
        argv[argv.index(flag) + 1] = value
    with (output / "run.log").open("x") as log:
        proc = subprocess.run(
            argv, stdout=log, stderr=log, timeout=command.get("timeout_seconds", 300), check=False
        )
    require(proc.returncode == 0, "ordinary evaluator execution failed")
    report = read(output / "report.json")
    require(
        report["status"] == "passed" and report["exit_code"] == 0,
        "ordinary evaluator report failed",
    )
    require(
        report["data_sha256"] == source["data"]["sha256"]
        and report["adapter_sha256"] == sha(adapter / "adapter_model.safetensors"),
        "evaluator binding mismatch",
    )
    sidecar = output / "eval/records.jsonl"
    require(sha(sidecar) == report["sidecar_sha256"], "ordinary sidecar SHA mismatch")
    seen, correct, groups = set(), 0, {}
    for row in map(json.loads, sidecar.read_text().splitlines()):
        if row["pass"] != 0 or row["variant"] != "original":
            continue
        require(row["id"] not in seen, "duplicate original result")
        seen.add(row["id"])
        question = by_id[row["id"]].questions[0]
        require(isinstance(question.target, HardTarget), "screen requires existing hard gold")
        prediction = read_probabilities(
            row["raw_logits"], list(answer_tokens(question)), temperature=1
        ).prediction
        ok = prediction == question.target.value
        correct += int(ok)
        words = set(question.instruction.lower().split())
        kind = (
            "temporal"
            if words & {"before", "after", "when", "while"}
            else "motion"
            if words & {"moving", "stationary"}
            else "existence"
        )
        group = groups.setdefault(kind, {"count": 0, "correct": 0})
        group["count"] += 1
        group["correct"] += int(ok)
    require(seen == set(by_id), "missing or extra original results")
    return {
        "count": len(seen),
        "correct": correct,
        "groups": groups if name == "video" else None,
        "argv": argv,
        "report": str(output / "report.json"),
        "report_sha256": sha(output / "report.json"),
        "sidecar": str(sidecar),
        "sidecar_sha256": sha(sidecar),
        "data_sha256": sha(data),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(s.id for s in samples).encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--through", choices=ORDER, default="video")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/s4-crossmodal-retention-screen-001"
    )
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    start = time.monotonic()
    receipt = args.output / "receipt.json"
    if args.resume:
        result = read(receipt)
    else:
        args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        result = {
            "status": "preparing",
            "exit_code": 1,
            "results": {},
            "cost": None,
            "stage_gate": False,
        }

    def save():
        result["duration_seconds_this_invocation"] = time.monotonic() - start
        temp = receipt.with_suffix(".tmp")
        with temp.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        temp.replace(receipt)

    try:
        training = ROOT / "reports/s4-crossmodal-retention-training-001.json"
        adapter = ROOT / "artifacts/s4-crossmodal-retention-train-001/adapter"
        train = read(training)
        require(
            train["status"] == "completed"
            and train["exit_code"] == 0
            and train["steps_completed"] == 30,
            "candidate training has not completed",
        )
        require(
            train["config"].get("reference_kl_weight", 0) == 0
            and train["config"]["sample_count"] == 360
            and train["config"]["accumulation"] == 12
            and train["init_adapter"]["weights_sha256"]
            == "6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac",
            "candidate is not the bounded first-adapter crossmodal coverage run",
        )
        require(
            sha(adapter / "adapter_model.safetensors") == train["adapter_sha256"],
            "candidate adapter SHA mismatch",
        )
        prior = ROOT / "artifacts/s4-pilot-five-screen-001"
        manifest, commands = read(prior / "manifest.json"), read(prior / "commands.json")
        identity = {
            "training_report_sha256": sha(training),
            "adapter_sha256": train["adapter_sha256"],
            "adapter_config_sha256": sha(adapter / "adapter_config.json"),
            "prior_manifest_sha256": sha(prior / "manifest.json"),
            "prior_commands_sha256": sha(prior / "commands.json"),
            "runner_sha256": sha(Path(__file__)),
            "crossmodal_pins": CROSS,
        }
        # JSON round-trip keeps tuple-based pins identical after receipt reload.
        identity = json.loads(json.dumps(identity))
        if args.resume:
            require(result["identity"] == identity, "candidate/source/runner identity changed")
            require(
                result["status"] in ("prepared", "partial_pass", "completed"),
                "failed screens cannot resume",
            )
            require(
                list(result["results"]) == list(ORDER[: len(result["results"])]),
                "non-prefix screen history",
            )
            for item in result["results"].values():
                require(
                    item["passed"]
                    and sha(Path(item["report"])) == item["report_sha256"]
                    and sha(Path(item["sidecar"])) == item["sidecar_sha256"],
                    "prior stage evidence changed",
                )
        result.update(
            identity=identity,
            through=args.through,
            scope="frozen public-dev screening only; no stage acceptance",
            thresholds=MINIMUM,
            video_subgroups={"temporal": 60, "motion": 7},
        )
        if not args.execute:
            result.update(status="prepared", exit_code=0)
            save()
            return 0
        result.update(status="running", exit_code=1)
        save()
        for name in ORDER[: ORDER.index(args.through) + 1]:
            if name in result["results"]:
                continue
            result["active_stage"] = name
            stage_started = time.monotonic()
            stage = args.output / name
            stage.mkdir(exist_ok=False, mode=0o700)
            if name in CROSS:
                report = cross_forward(name, adapter, stage)
                path = stage / "report.json"
                with path.open("x") as stream:
                    json.dump(report, stream, indent=2)
                item = {
                    "count": 50,
                    "correct": round(report["metrics"]["accuracy"] * 50),
                    "selected_ids_sha256": report["selected_ids_sha256"],
                    "data_sha256": report["data_sha256"],
                    "report": str(path),
                    "report_sha256": sha(path),
                    "sidecar": str(stage / "records.jsonl"),
                    "sidecar_sha256": sha(stage / "records.jsonl"),
                }
            else:
                source = next(source for source in manifest["sources"] if source["cohort"] == name)
                command = next(command for command in commands if command["source"] == name)
                item = ordinary(name, adapter, stage, source, command)
            require(
                sha(adapter / "adapter_model.safetensors") == identity["adapter_sha256"]
                and sha(adapter / "adapter_config.json") == identity["adapter_config_sha256"],
                "candidate adapter changed during screen",
            )
            item.update(required_correct=MINIMUM[name], passed=item["correct"] >= MINIMUM[name])
            if name == "video":
                item["passed"] &= (
                    item["groups"].get("temporal", {}).get("correct", 0) >= 60
                    and item["groups"].get("motion", {}).get("correct", 0) >= 7
                )
            item["duration_seconds"] = time.monotonic() - stage_started
            result["results"][name] = item
            result.update(
                status="partial_pass" if item["passed"] else "screen_failed",
                exit_code=0 if item["passed"] else 2,
            )
            save()
            if not item["passed"]:
                return 2
        result.update(
            status="completed" if len(result["results"]) == len(ORDER) else "partial_pass",
            exit_code=0,
        )
        save()
        return 0
    except Exception as exc:
        result.update(status="execution_failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        save()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
