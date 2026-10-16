"""Explicit GPU-only numerical parity probe for a SHA-bound rank-expanded derivative."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import time
from pathlib import Path

path = Path(__file__).with_name("evaluate-paired-train12.py")
spec = importlib.util.spec_from_file_location("rank_parity_official_reuse", path)
assert spec and spec.loader
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
require = base.require
sha = base.file_sha256
BEFORE_SHA = "2c28645ba52ca05a9fb0514c75d31e8c588e12d7dd69bf86946cee87e8fb4711"
BEFORE_RECEIPT_SHA = "87adf4d1600e5718ff330f271da5e43916a6c71d06fcad27a42ec8b23c79eae8"
INPUT_REPORT_SHA = "0b352bd26481ab94cb8be26c383ff9b12e88862bfef0d48abe75c81f3be7ae8c"


def max_probability_difference(before, after):
    require(set(before) == set(after) == {"yes", "no", "U"}, "all legal plus U actions")
    require(
        all(math.isfinite(v) for row in [before, after] for v in row.values()),
        "finite probabilities",
    )
    return max(abs(before[k] - after[k]) for k in before)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--initializer", type=Path, required=True)
    ap.add_argument("--expansion-report-sha256", required=True)
    ap.add_argument(
        "--inputs", type=Path, default=base.ROOT / "artifacts/s4-paired-train12-input-001"
    )
    ap.add_argument(
        "--before", type=Path, default=base.ROOT / "artifacts/s4-paired-train12-before-001"
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if not a.execute:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    a.output.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "failed",
        "exit_code": 1,
        "gpu_requested": a.execute,
        "budget_seconds": 90,
        "cost": None,
        "training_authorized": False,
        "scope": "initializer numerical parity only",
        "probe_sha256": sha(Path(__file__)),
        "reused_evaluator_sha256": sha(path),
    }

    def interrupted(signum, frame):
        raise TimeoutError(f"parity timeout/signal:{signum}")

    for sig in [signal.SIGALRM, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupted)
    signal.alarm(90)
    try:
        expansion_path = a.initializer / "rank-expansion.json"
        require(sha(expansion_path) == a.expansion_report_sha256, "expanded derivative receipt SHA")
        expansion = json.loads(expansion_path.read_bytes())
        require(
            expansion["status"] == "converted_research_initializer"
            and expansion["exit_code"] == 0
            and expansion["source_weights_sha256"] == base.INIT_SHA,
            "fixed source derivative",
        )
        require(
            expansion["old_rank"] == 16
            and expansion["new_rank"] == 64
            and expansion["old_alpha"] == 32
            and expansion["new_alpha"] == 128
            and all(expansion["checks"].values()),
            "transform invariants",
        )
        require(
            sha(Path(expansion["source_path"]) / "adapter_model.safetensors") == base.INIT_SHA,
            "source changed",
        )
        for name, digest in expansion["output_sha256"].items():
            require(
                Path(name).name == name and sha(a.initializer / name) == digest,
                "derivative file SHA",
            )
        transform = Path(__file__).with_name("expand-decision-lora-rank.py")
        require(sha(transform) == expansion["transform_sha256"], "actual transform code SHA")
        require(
            sha(a.before / "receipt.json") == BEFORE_RECEIPT_SHA
            and sha(a.before / "records.jsonl") == BEFORE_SHA,
            "actual f2db before evidence",
        )
        before_receipt = json.loads((a.before / "receipt.json").read_bytes())
        require(
            before_receipt["status"] == "completed" and before_receipt["exit_code"] == 0,
            "before completed",
        )
        rows = [json.loads(line) for line in (a.before / "records.jsonl").read_text().splitlines()]
        before = {r["id"]: r for r in rows}
        require(
            len(before) == len(rows) == 12
            and all(r["adapter_sha256"] == base.INIT_SHA for r in rows),
            "before twelve source rows",
        )
        require(sha(a.inputs / "report.json") == INPUT_REPORT_SHA, "fixed12subset receipt")
        prepared = json.loads((a.inputs / "report.json").read_bytes())
        for name, digest in prepared["output_sha256"].items():
            require(sha(a.inputs / name) == digest, "subset file SHA")
        config = base.LoopConfig.model_validate_json((a.inputs / "config-ce.json").read_bytes())
        samples = base.smoke_samples(
            config, a.inputs / "train.jsonl", expected_data_versions=config.expected_data_versions
        )
        base.verify_data_is_train_pool(
            a.inputs / "train.jsonl", [a.inputs / "allocation-ledger.jsonl"]
        )
        require(set(before) == {s.id for s in samples}, "same twelve IDs")
        for name, digest in before_receipt["asset_hashes"].items():
            require(sha(base.MODEL / name) == digest, "fixed base/tokenizer/template asset")
        for name, digest in before_receipt["code_sha256"].items():
            require(sha(Path(name)) == digest, "same official preparation/readout code")
        report.update(
            status="ready_without_gpu",
            exit_code=0,
            expanded_weights_sha256=expansion["output_sha256"]["adapter_model.safetensors"],
            expanded_config_sha256=expansion["output_sha256"]["adapter_config.json"],
            expansion_report_sha256=a.expansion_report_sha256,
            source_weights_sha256=base.INIT_SHA,
            actual_before_records_sha256=BEFORE_SHA,
            parity_validated=False,
            transform_sha256=expansion["transform_sha256"],
        )
        if a.execute:
            import importlib
            import subprocess

            require(
                not subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
                ).strip(),
                "GPU busy",
            )
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            peft = importlib.import_module("peft")
            processor = transformers.AutoProcessor.from_pretrained(
                str(base.MODEL), local_files_only=True, trust_remote_code=False
            )
            base.configure_image_processor(processor, config.image_max_soft_tokens)
            tokenizer = base.OfficialTokenizerAdapter(base.MODEL)
            roots = [Path(r) for r in config.media_roots]
            examples = [
                base.prepare_mixed_example(
                    s,
                    tokenizer,
                    processor,
                    roots[0],
                    max_length=config.max_length,
                    extra_roots=roots[1:],
                    video_num_frames=16,
                )
                for s in samples
            ]
            bindings = []
            for sample, example in zip(samples, examples, strict=True):
                binding = {
                    "id": sample.id,
                    "sample_sha256": hashlib.sha256(base.data_bytes([sample])).hexdigest(),
                    "input_sha256": base.reference_prepared_hash(
                        {
                            **example,
                            "attention_mask": [1] * len(example["input_ids"]),
                            "use_cache": False,
                            "logits_to_keep": 1,
                        }
                    ),
                    "answer_token_ids": example["answer_token_ids"],
                    "target": sample.questions[0].target.value,
                }
                require(
                    all(before[sample.id][k] == v for k, v in binding.items()),
                    "complete12input/mapping parity",
                )
                bindings.append(binding)
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            model_base = transformers.Gemma4ForConditionalGeneration.from_pretrained(
                str(base.MODEL),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            model = (
                peft.PeftModel.from_pretrained(model_base, str(a.initializer), is_trainable=False)
                .to("cuda")
                .eval()
            )
            model.requires_grad_(False)
            maximum_joint = maximum_conditional = maximum_logits = 0.0
            with torch.inference_mode():
                for binding, example in zip(bindings, examples, strict=True):
                    ids = torch.tensor([example["input_ids"]], device="cuda", dtype=torch.long)
                    media = {
                        k: example[k].to("cuda") for k in base.MEDIA_FORWARD_KEYS if k in example
                    }
                    output = model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=False,
                        logits_to_keep=1,
                        return_dict=True,
                        **media,
                    )
                    logits = output.logits[0, -1].float()
                    require(bool(torch.isfinite(logits).all()), "finite full vocabulary")
                    raw = {
                        token: float(logits[index].item())
                        for token, index in example["answer_token_ids"].items()
                    }
                    readout = base.read_probabilities(raw, ["yes", "no"], temperature=1)
                    old = before[binding["id"]]
                    maximum_joint = max(
                        maximum_joint,
                        max_probability_difference(
                            old["joint_probabilities"], readout.joint_probabilities
                        ),
                    )
                    maximum_conditional = max(
                        maximum_conditional,
                        max(
                            abs(old["probabilities"][k] - readout.probabilities[k])
                            for k in ["yes", "no"]
                        ),
                    )
                    maximum_logits = max(
                        maximum_logits, max(abs(old["raw_logits"][k] - raw[k]) for k in raw)
                    )
                    del output, logits, ids, media
            report.update(
                max_joint_probability_difference=maximum_joint,
                max_conditional_probability_difference=maximum_conditional,
                max_legal_plus_u_logit_difference=maximum_logits,
                verified_complete_input_mapping=12,
                parity_tolerance=1e-3,
                parity_validated=max(maximum_joint, maximum_conditional) <= 1e-3,
            )
            require(report["parity_validated"], "GPU probability parity exceeds1e-3")
            require(
                sha(a.initializer / "adapter_model.safetensors")
                == report["expanded_weights_sha256"],
                "derivative changed",
            )
            report.update(status="completed", exit_code=0)
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        report["duration_seconds"] = time.monotonic() - started
        (a.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(report["exit_code"])


if __name__ == "__main__":
    main()
