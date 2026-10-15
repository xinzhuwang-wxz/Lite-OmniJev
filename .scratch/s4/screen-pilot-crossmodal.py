"""Bounded public-development diagnostic; never a sealed or stage gate."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

from scripts import s0_development_baselines as baseline

parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=["original", "supplemental"], required=True)
args = parser.parse_args()
start = time.monotonic()
root = Path("/root/autodl-tmp/s1-text-lora-001")
base = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
adapter = root / "artifacts/s4-joint-pilot-train-001/adapter"
training = json.loads((root / "reports/s4-joint-pilot-train-001.json").read_text())
assert (
    training["status"] == "completed"
    and training["exit_code"] == 0
    and training["steps_completed"] == 30
)
data = root / "artifacts/s4-gold-inventory-001/crossmodal-dev-002/dev.jsonl"
media = root / "artifacts/s4-gold-inventory-001/crossmodal-dev-source-bundle-002"
if args.source == "original":
    data = Path(
        "/root/autodl-tmp/s0-crossmodal-development-001/artifacts/crossmodal-dev-001-a/dev.jsonl"
    )
    media = data.parent / "media"
out = root / f"artifacts/s4-pilot-crossmodal-{args.source}-001"
report_path = root / f"reports/s4-pilot-crossmodal-{args.source}-001.json"
assert not report_path.exists()
out.mkdir(exist_ok=False)
report = {
    "scope": f"50 fixed {args.source} dev items, one-pass S4 pilot, not stage acceptance",
    "status": "failed",
    "exit_code": 1,
    "seed": 42,
    "samples": 50,
    "passes": 1,
    "dtype": "bfloat16",
    "image_max_soft_tokens": 1120,
    "base_revision": base.name,
    "cost_usd": None,
    "runtime": platform.python_version(),
}
try:
    report["data_sha256"] = baseline.sha256_file(data)
    assert (
        report["data_sha256"]
        == (
            {
                "original": "8099d5c78af26fd0848dd36d06a52b68906e6789cb2c55f7c62d8773485298af",
                "supplemental": "1d2d8e151a3bc282099764a5ce7d04e8190fd212684ca010582511c5b25cfde1",
            }[args.source]
        )
    )
    report["adapter_sha256"] = baseline.sha256_file(adapter / "adapter_model.safetensors")
    assert report["adapter_sha256"] == training["adapter_sha256"]
    report["training_report_sha256"] = baseline.sha256_file(
        root / "reports/s4-joint-pilot-train-001.json"
    )
    report["adapter_config_sha256"] = baseline.sha256_file(adapter / "adapter_config.json")
    samples = baseline.load_samples(data)
    baseline.validate_samples(samples, cohort="crossmodal")
    samples = sorted(
        samples, key=lambda s: hashlib.sha256(("42:s4-crossmodal:" + s.id).encode()).hexdigest()
    )[:50]
    report["selected_ids_sha256"] = hashlib.sha256(
        "\n".join(s.id for s in samples).encode()
    ).hexdigest()
    processor = baseline.default_processor_factory(base)

    class BudgetProcessor:
        def __getattr__(self, name):
            return getattr(processor, name)

        def __call__(self, **kwargs):
            if "images" in kwargs:
                kwargs["images_kwargs"] = {"max_soft_tokens": 1120}
            return processor(**kwargs)

    baseline.configure_torch_runtime(torch)
    model = AutoModelForImageTextToText.from_pretrained(
        str(base),
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
    torch.cuda.reset_peak_memory_stats()

    def forward(prepared):
        inputs = baseline.move_inputs_to_cuda(prepared.inputs, torch, torch.bfloat16)
        with torch.inference_mode():
            result = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
            logits = result.logits[0, -1].float()
            assert bool(torch.isfinite(logits).all().item())
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            return {
                int(i): float(logprobs[int(i)].item()) for i in prepared.answer_token_ids.values()
            }

    records = baseline.run_pass(
        samples,
        processor=BudgetProcessor(),
        media_root=media,
        pass_index=1,
        forward=forward,
        records_sidecar=out / "records.jsonl",
    )
    report["metrics"] = baseline.score_records(records)
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["records_sha256"] = baseline.sha256_file(out / "records.jsonl")
    report["torch"] = torch.__version__
    report["gpu"] = torch.cuda.get_device_name()
    report.update(status="completed", exit_code=0)
except Exception as exc:
    # Detailed traces stay in the remote private job log; aggregate is category only.
    report["failure_category"] = type(exc).__name__
    import traceback

    with (out / "failure.log").open("x") as h:
        traceback.print_exc(file=h)
finally:
    report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    report["runner_sha256"] = baseline.sha256_file(Path(__file__))
    report["code_sha256"] = {p: baseline.sha256_file(Path(p)) for p in baseline.REQUIRED_CODE_FILES}
    with report_path.open("x") as h:
        json.dump(report, h, indent=2)
print(json.dumps(report))
raise SystemExit(report["exit_code"])
