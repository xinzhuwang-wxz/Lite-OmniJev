"""Adaptive-frame probe runner: explicit 16-frame indices on arbitrary data+adapter.

Same proven mechanism as the motion-coverage runner: process-local
baseline.load_media override feeding load_media(frame_indices=...) from a
CPU manifest keyed by media sha; frozen base + decision LoRA; official
template; single-token readout records. GPU; run by this slice under its
granted window only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import hashlib
    import os
    import sys

    sys.path.insert(0, os.getcwd())

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    from scripts import s0_development_baselines as baseline
    from train.multimodal import file_hash, load_media

    manifest = json.loads(args.manifest.read_text())
    original_loader = baseline.load_media

    def selected(path: Path, kind: str, *, num_frames: int, max_seconds: float):
        record = manifest[file_hash(path)]
        indices = record["adaptive_indices"]
        decoded = load_media(
            path, kind, num_frames=num_frames, max_seconds=max_seconds, frame_indices=indices
        )
        metadata = decoded["metadata"]
        assert metadata["total_num_frames"] == record["total_num_frames"]
        assert metadata["frames_indices"] == indices
        return decoded

    baseline.load_media = selected
    started = time.time()
    records = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "gpu_used": True, "sampling": "adaptive-8uniform-8peak"}
    try:
        processor = baseline.default_processor_factory(args.model_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        weights = args.adapter / "adapter_model.safetensors"
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != args.adapter_sha256:
            raise SystemExit(f"adapter sha mismatch {digest}")
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
        model.requires_grad_(False)
        model.eval()
        model.to("cuda")

        def forward(prepared):
            inputs = baseline.move_inputs_to_cuda(prepared.inputs, torch, torch.bfloat16)
            with torch.inference_mode():
                result = model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                logits = result.logits[0, -1].float()
                assert bool(torch.isfinite(logits).all())
                logprobs = logits.log_softmax(-1)
                return {
                    int(i): float(logprobs[int(i)].item())
                    for i in prepared.answer_token_ids.values()
                }

        samples = baseline.load_samples(args.data)
        with (args.output_dir / "records.jsonl").open("x", encoding="utf-8") as stream:
            for sample in samples:
                prepared = baseline.prepare_sample(sample, processor, args.media_root)
                record = baseline.make_record(prepared, pass_index=1, forward=forward)
                record["sampling_policy"] = "adaptive"
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
        report.update(
            status="completed",
            exit_code=0,
            samples=len(samples),
            metrics=baseline.score_records(records),
            wall_seconds=round(time.time() - started, 3),
            adapter_sha256=digest,
        )
    finally:
        baseline.load_media = original_loader
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({k: report.get(k) for k in ("status", "samples", "metrics")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
