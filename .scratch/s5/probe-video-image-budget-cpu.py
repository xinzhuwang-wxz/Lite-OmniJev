"""Public DEV-only CPU tensor comparison; no model weights or GPU inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from data.schema import Sample  # noqa: E402
from scripts import private_stage_eval as stage  # noqa: E402
from scripts import s0_development_baselines as baseline  # noqa: E402
from scripts.s0_public_baselines import sha256_file, sha256_json  # noqa: E402
from train.multimodal import configure_image_processor  # noqa: E402
from train.trainer import MEDIA_FORWARD_KEYS, prepare_media_example  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    assert sha256_file(args.data) == args.data_sha256
    with args.data.open() as handle:
        row = next(line for line in handle if line.strip())
    sample = Sample.model_validate_json(row)
    assert sample.meta.pool == "dev" and [m.type for m in sample.prefix.media] == ["video"]
    import torch
    from transformers import AutoProcessor

    assert not torch.cuda.is_available()
    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        str(args.model_dir), local_files_only=True, trust_remote_code=False
    )
    video_config_sha = sha256_json(processor.video_processor.to_dict())
    branches = {}
    for budget in (280, 1120):
        configure_image_processor(processor, budget)
        assert sha256_json(processor.video_processor.to_dict()) == video_config_sha
        prepared = baseline.prepare_sample(
            sample, stage.BudgetProcessor(processor), args.media_root
        )
        assert all(
            getattr(value, "device", torch.device("cpu")).type == "cpu"
            for value in prepared.inputs.values()
        )
        branches[str(budget)] = {
            "tensors": baseline.summarize_inputs(prepared.inputs),
            "prompt_sha256": prepared.prompt_sha256,
            "answer_token_ids_sha256": sha256_json(prepared.answer_token_ids),
            "prompt_tokens": prepared.prompt_tokens,
        }
        del prepared
    configure_image_processor(processor, 280)
    dev = prepare_media_example(
        sample, processor, args.media_root, max_length=8192, video_num_frames=16
    )
    ids = torch.tensor([dev["input_ids"]], dtype=torch.long)
    inputs = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        **{key: dev[key] for key in MEDIA_FORWARD_KEYS if key in dev},
    }
    dev_tensors = baseline.summarize_inputs(inputs)
    source = Path(stage.__file__).resolve().parents[1]
    result = {
        "schema_version": "public-video-image-budget-cpu-parity-v1",
        "status": "completed",
        "exit_code": 0,
        "scope": "one public video input; image-budget-only preprocessing comparison",
        "gpu_used": False,
        "model_weights_loaded": False,
        "private_data_read": False,
        "data_sha256": args.data_sha256,
        "public_sample_sha256": hashlib.sha256(row.encode()).hexdigest(),
        "media_sha256": sample.prefix.media[0].sha256,
        "media_type": "video",
        "video_frames": 16,
        "model_processor_config_sha256": sha256_file(args.model_dir / "processor_config.json"),
        "video_processor_config_sha256": video_config_sha,
        "source_files": {
            name: sha256_file(source / name)
            for name in [
                "scripts/private_stage_eval.py",
                "scripts/s0_development_baselines.py",
                "train/multimodal.py",
                "train/trainer.py",
                "scripts/s1_dev_eval.py",
            ]
        },
        "branches": branches,
        "image_budget_only_tensor_identity": branches["280"] == branches["1120"],
        "dev280_tensor_manifest": dev_tensors,
        "actual_dev_vs_baseline1120_preprocessor_identity": dev_tensors
        == branches["1120"]["tensors"],
        "probability_parity": "not_measured",
        "gpu_input_dtype_cast_parity": "not_measured",
    }
    args.report.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["branches", "dev280_tensor_manifest"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
