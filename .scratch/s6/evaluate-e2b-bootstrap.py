"""Fixed public DEV50, paired E2B base/adapter on identical tensors; CPU preparation default."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import signal
import time
from pathlib import Path

from scripts import s0_development_baselines as baseline
from scripts.assets import write_json
from scripts.check_tokens import probe
from train.trainer import reference_prepared_hash, verified_assets, verified_init_adapter

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
MODEL = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E2B-it/3e22461f65e89153144f8adb70e3b8c2cc9845a7"
)
ADAPTER_SHA = "e49accdf456f3f8599bb02583749b3d8a5d4fd59ade7e35939558c495961e8f2"
DATA_SHA = {
    "text": "8c7809e8b07673904b71bc4676a49a5e31abb22ee7126f2dc09e8fde45b4e070",
    "video": "79ed30be3a687a6c767c4d7c55c24dc7ac08d5a1698a6f49e3bc7d47e74e63d2",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def select_samples(samples, cohort):
    baseline.validate_samples(samples, cohort=cohort)
    if cohort == "video":
        helper = Path(__file__).parents[1] / "s4/evaluate-video-motion-coverage.py"
        spec = importlib.util.spec_from_file_location("fixed_motion_selection", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.select_samples(samples)
    # Predeclared original file prefix, independent of predictions/gold correctness.
    return samples[:50]


def validate_training(trained):
    require(
        trained["status"] == "paused"
        and trained["exit_code"] == 0
        and trained["steps_completed"] == 10,
        "completed E2B first10 required",
    )
    require(
        trained["adapter_sha256"] == ADAPTER_SHA and trained["model_revision"] == MODEL.name,
        "E2B candidate/model identity mismatch",
    )
    require("init_adapter" not in trained, "E4B or other adapter inheritance forbidden")
    require(
        trained["data_sha256"]
        == "795f69fe9474523009bd953a0436bc505e85efdd4fb0f5e7c41e4d86c3bb028f",
        "bootstrap TRAIN identity mismatch",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["text", "video"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if not args.execute:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    report = {
        "status": "failed",
        "exit_code": 1,
        "gpu_requested": args.execute,
        "stage_gate": False,
        "budget_seconds": 240,
        "cohort": args.cohort,
        "cost": None,
        "script_sha256": sha(__file__),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / ("report.json" if args.execute else "preparation.json")
    require(not destination.exists(), "refuse to overwrite evaluation evidence")

    def interrupted(signum, frame):
        raise TimeoutError(f"paired E2B screening signal {signum}")

    for sig in [signal.SIGALRM, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupted)
    signal.alarm(230)
    try:
        import torch
        from peft import PeftModel
        from transformers import Gemma4ForConditionalGeneration

        trained_path = ROOT / "reports/s6-e2b-gold-step0010.json"
        trained = json.loads(trained_path.read_text())
        validate_training(trained)
        adapter = ROOT / "artifacts/s6-e2b-gold-bootstrap-001/adapter-invocation-step-0010"
        adapter_identity = verified_init_adapter(adapter, ADAPTER_SHA)
        manifest = ROOT / "artifacts/s6-e2b-training-readiness-005/e2b-runtime-manifest.json"
        hashes = verified_assets(MODEL, manifest)
        require(hashes == trained["asset_hashes"], "training/eval E2B assets differ")
        tokens = probe(MODEL)
        require(tokens["exit_code"] == 0 and tokens["answer_count"] == 33, "E2B33token probe")
        cohorts = json.loads(Path("configs/s0-development-cohorts-h20.json").read_text())["cohorts"]
        cohort = next(c for c in cohorts if c["name"] == args.cohort)
        data, media = Path(cohort["data"]), Path(cohort["media_root"])
        require(sha(data) == DATA_SHA[args.cohort], "original public DEV500 SHA mismatch")
        samples = select_samples(baseline.load_samples(data), args.cohort)
        require(
            len(samples) == 50 and all(s.meta.pool == "dev" for s in samples), "public DEV50 only"
        )
        processor = baseline.default_processor_factory(MODEL)

        # Only video kwargs, never leak image options into the text processor.
        class VideoBudget:
            def __getattr__(self, key):
                return getattr(processor, key)

            def __call__(self, **kwargs):
                if "videos" in kwargs:
                    kwargs["videos_kwargs"] = {"max_soft_tokens": 70}
                return processor(**kwargs)

        selected_ids_sha = hashlib.sha256(json.dumps([s.id for s in samples]).encode()).hexdigest()
        identity = {
            "model_revision": MODEL.name,
            "asset_hashes": hashes,
            "adapter": adapter_identity,
            "training_report_sha256": sha(trained_path),
            "data_sha256": sha(data),
            "selected_ids_sha256": selected_ids_sha,
            "template_sha256": tokens["template_sha256"],
            "code_hashes": {
                name: sha(name)
                for name in [
                    "scripts/s0_development_baselines.py",
                    "train/multimodal.py",
                    "train/template.py",
                    "train/readout.py",
                    ".scratch/s4/evaluate-video-motion-coverage.py",
                ]
            },
            "script_sha256": sha(__file__),
        }
        prior = None
        if args.execute:
            prior = json.loads((args.output / "preparation.json").read_text())
            require(
                prior["status"] == "cpu_prepared" and prior["identity"] == identity,
                "CPU prepared source/model/input identity changed",
            )
            require(not (args.output / "records.jsonl").exists(), "records already exist")
            baseline.configure_torch_runtime(torch)
            base = Gemma4ForConditionalGeneration.from_pretrained(
                str(MODEL),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            model = (
                PeftModel.from_pretrained(base, str(adapter), is_trainable=False).to("cuda").eval()
            )
            model.requires_grad_(False)
        bindings = []
        records = {"base": [], "candidate": []}
        for index, sample in enumerate(samples):
            prepared = baseline.prepare_sample(sample, VideoBudget(), media)
            binding = {
                "id": sample.id,
                "input_sha256": prepared.input_sha256,
                "input_summary_sha256": baseline.sha256_json(prepared.input_summary),
                "answer_token_ids": prepared.answer_token_ids,
            }
            bindings.append(binding)
            if args.execute:
                require(
                    prior["bindings"][index] == binding, "actual prepared tensor differs from CPU"
                )
                inputs = baseline.move_inputs_to_cuda(prepared.inputs, torch, torch.bfloat16)
                moved_hash = reference_prepared_hash(inputs)
                for phase in ["base", "candidate"]:
                    require(reference_prepared_hash(inputs) == moved_hash, "forward input mutated")

                    def forward(_prepared, phase=phase, inputs=inputs, prepared=prepared):
                        context = (
                            model.disable_adapter() if phase == "base" else contextlib.nullcontext()
                        )
                        with context, torch.inference_mode():
                            out = model(
                                **inputs, use_cache=False, logits_to_keep=1, return_dict=True
                            )
                            logits = out.logits[0, -1].float()
                            require(
                                bool(torch.isfinite(logits).all()), "nonfinite vocabulary logits"
                            )
                            logprobs = logits.log_softmax(-1)
                            return {
                                i: float(logprobs[i]) for i in prepared.answer_token_ids.values()
                            }

                    row = baseline.make_record(prepared, pass_index=1, forward=forward)
                    row.update(phase=phase, actual_forward_inputs_sha256=moved_hash)
                    records[phase].append(row)
                    baseline.append_record_jsonl(args.output / "records.jsonl", row)
                require(reference_prepared_hash(inputs) == moved_hash, "candidate mutated input")
            if (index + 1) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "prepared": index + 1,
                            "gpu": args.execute,
                            "elapsed": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        report.update(identity=identity, bindings=bindings, status="cpu_prepared", exit_code=0)
        if args.execute:
            report.update(
                status="completed",
                paired_inputs=50,
                actual_forwards=100,
                base=baseline.score_records(records["base"]),
                candidate=baseline.score_records(records["candidate"]),
                records_sha256=sha(args.output / "records.jsonl"),
                baseline_method="same resident E2B with PEFT disable_adapter; never E4B scores",
            )
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        report["duration_seconds"] = time.monotonic() - started
        write_json(destination, report)
    print(
        json.dumps({k: report[k] for k in ["status", "exit_code", "duration_seconds"]}), flush=True
    )
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
