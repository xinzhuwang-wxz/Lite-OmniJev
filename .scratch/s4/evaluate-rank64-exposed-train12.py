"""Actual first-invocation TRAIN pairs, rank64 initializer versus its step10 result."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import time
from pathlib import Path

path = Path(__file__).with_name("evaluate-paired-train12.py")
spec = importlib.util.spec_from_file_location("exposed_pair_reuse", path)
assert spec and spec.loader
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
require = base.require
sha = base.file_sha256
ROOT = base.ROOT
TRAIN_SHA = "11e14016bfe5e546fe67bb0b7b1a21dbc5b89a082253ffbb743ee3bbd4e90a05"
PREP_SHA = "2c51e16cb88ab91dfb45dca84a2325b371ad259dce522a01fa50e042090cfa87"
INIT_SHA = "62920299c7a1ac33b2a874b586a9d7f3ea3eaa4e38198bc20388642482318fc6"
AFTER_SHA = "07c2d4dc397c75b0d030d3aa2822e1dfeac27f3653b633dc42aa92dcb4803bc4"
TRAIN_REPORT = ROOT / "reports/s4-rank64-matched-pilot-training-001.json"
AFTER = ROOT / "artifacts/s4-rank64-matched-pilot-train-001/adapter-invocation-step-0010"


def exposed_pairs(samples, metadata, recipe, seed):
    require(len(samples) == len(recipe["draws"]) == 1200, "fixed global1200")
    order = list(range(1200))
    random.Random(seed).shuffle(order)
    actual = [samples[order[i]] for i in range(120)]
    require(
        all(s.id == recipe["draws"][i]["item_id"] for i, s in enumerate(actual)),
        "actual exposed draw order",
    )
    pairs = {frozenset(m["item_id"] for m in p["members"]): p for p in metadata["rows"]}
    selected = []
    for step in range(10):
        for offset in [0, 2, 4]:
            two = actual[step * 12 + offset : step * 12 + offset + 2]
            key = frozenset(s.id for s in two)
            require(key in pairs and len(key) == 2, "actual adjacent pair metadata")
            require(all(s.meta.pool == "train" for s in two), "TRAIN exposure")
            selected.append(pairs[key])
    require(
        len(selected) == len({p["pair"] for p in selected}) == 30, "thirty actual exposed pairs"
    )
    return {"metadata_only": True, "rows": selected}


def prepare(out):
    require(not out.exists(), "output exists")
    require(sha(TRAIN_REPORT) == TRAIN_SHA, "actual rank64 training receipt SHA")
    trained = json.loads(TRAIN_REPORT.read_bytes())
    require(
        trained["status"] == "paused"
        and trained["exit_code"] == 0
        and trained["steps_completed"] == 10
        and trained["invocation"]["start_step"] == 0
        and trained["invocation"]["next_draw"] == 120,
        "actual0to10invocation",
    )
    source = ROOT / "artifacts/s4-continuous-matched-pool-001"
    require(sha(source / "report.json") == PREP_SHA, "fixed pool receipt")
    report = json.loads((source / "report.json").read_bytes())
    for name, digest in report["output_sha256"].items():
        require(sha(source / name) == digest, "global pool bound file")
    require(trained["data_sha256"] == sha(source / "train.jsonl"), "actual trained data")
    require(sha(AFTER.parent / "data.jsonl") == trained["data_sha256"], "actual runtime data")
    require(
        trained["init_adapter"]["weights_sha256"] == INIT_SHA
        and trained["adapter_sha256"] == AFTER_SHA,
        "rank64 before/after",
    )
    samples = [
        base.Sample.model_validate_json(line)
        for line in (source / "train.jsonl").read_text().splitlines()
    ]
    ledger = [
        json.loads(line) for line in (source / "allocation-ledger.jsonl").read_text().splitlines()
    ]
    metadata = json.loads((source / "matched-pair-metadata.json").read_bytes())
    actual = exposed_pairs(
        samples,
        metadata,
        json.loads((source / "draw-recipe.json").read_bytes()),
        trained["config"]["seed"],
    )
    selected, selected_ledger, pairs = base.select_train12(samples, ledger, actual)
    before = base.verified_init_adapter(Path(trained["init_adapter"]["path"]), INIT_SHA)
    after = base.verified_init_adapter(AFTER, AFTER_SHA)
    out.mkdir(mode=0o700)
    (out / "train.jsonl").write_bytes(base.data_bytes(selected))
    (out / "allocation-ledger.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in selected_ledger)
    )
    config = {**trained["config"], "sample_count": 12}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    base.verify_data_is_train_pool(out / "train.jsonl", [out / "allocation-ledger.jsonl"])
    result = {
        "status": "ready_for_readonly_probe",
        "exit_code": 0,
        "gpu_used": False,
        "training_authorized": False,
        "cost": None,
        "samples": 12,
        "pairs": 6,
        "source_actual_draws": 120,
        "source_exposed_video_pairs": 30,
        "source_training_report_sha256": TRAIN_SHA,
        "source_preparation_sha256": PREP_SHA,
        "source_runtime_data_sha256": trained["data_sha256"],
        "selection": "six whole pairs by fixed hash from actual first120draws; no DEV",
        "selected_pairs": pairs,
        "before": before,
        "after": after,
        "asset_hashes": trained["asset_hashes"],
        "helper_sha256": sha(Path(__file__)),
        "output_sha256": {p.name: sha(p) for p in out.iterdir()},
    }
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def metrics(records, pairs):
    result = base.pair_metrics(records, pairs)
    result.update(
        correct=sum(r["prediction"] == r["target"] for r in records),
        samples=len(records),
        mean_joint_ce=sum(r["joint_ce"] for r in records) / len(records),
        mean_conditional_nll=sum(-math.log(r["probabilities"][r["target"]]) for r in records)
        / len(records),
        mean_gold_margin=sum(r["gold_margin"] for r in records) / len(records),
    )
    by_id = {r["id"]: r for r in records}
    result["pair_logodds_direction"] = sum(
        next(by_id[i]["logodds"] for i in ids if by_id[i]["target"] == "yes")
        > next(by_id[i]["logodds"] for i in ids if by_id[i]["target"] == "no")
        for ids in pairs
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["prepare", "probe"], required=True)
    ap.add_argument(
        "--inputs", type=Path, default=ROOT / "artifacts/s4-rank64-exposed-train12-input-001"
    )
    ap.add_argument("--expected-input-report-sha256")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    os.umask(0o077)
    if a.phase == "prepare":
        print(json.dumps(prepare(a.inputs)))
        return
    if not a.execute:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    a.output.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "failed",
        "exit_code": 1,
        "gpu_requested": a.execute,
        "budget_seconds": 120,
        "cost": None,
        "training_authorized": False,
        "scope": "actual exposed TRAIN fit only; no generalization or gate claim",
        "probe_sha256": sha(Path(__file__)),
    }

    def interrupted(signum, frame):
        raise TimeoutError(f"exposed probe timeout:{signum}")

    for sig in [signal.SIGALRM, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupted)
    signal.alarm(120)
    try:
        require(
            a.expected_input_report_sha256
            and sha(a.inputs / "report.json") == a.expected_input_report_sha256,
            "input receipt SHA",
        )
        source = json.loads((a.inputs / "report.json").read_bytes())
        require(
            source["status"] == "ready_for_readonly_probe"
            and sha(TRAIN_REPORT) == source["source_training_report_sha256"] == TRAIN_SHA,
            "actual training source",
        )
        for name, digest in source["output_sha256"].items():
            require(sha(a.inputs / name) == digest, "subset file SHA")
        identities = {
            label: base.verified_init_adapter(
                Path(source[label]["path"]), source[label]["weights_sha256"]
            )
            for label in ["before", "after"]
        }
        require(
            identities["before"] == source["before"] and identities["after"] == source["after"],
            "adapter config identities",
        )
        for name, digest in source["asset_hashes"].items():
            require(sha(base.MODEL / name) == digest, "official assets")
        config = base.LoopConfig.model_validate_json((a.inputs / "config.json").read_bytes())
        samples = base.smoke_samples(
            config, a.inputs / "train.jsonl", expected_data_versions=config.expected_data_versions
        )
        require(len(samples) == 12 and all(s.meta.pool == "train" for s in samples), "twelve TRAIN")
        by_id = {s.id: s for s in samples}
        for ids in source["selected_pairs"]:
            require(
                base.nonmedia_key(by_id[ids[0]]) == base.nonmedia_key(by_id[ids[1]]),
                "same nonmedia pair",
            )
        report.update(
            status="ready_without_gpu",
            exit_code=0,
            input_report_sha256=a.expected_input_report_sha256,
            parity_of_inputs_verified=False,
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
            bindings = [
                {
                    "id": s.id,
                    "target": s.questions[0].target.value,
                    "sample_sha256": hashlib.sha256(base.data_bytes([s])).hexdigest(),
                    "input_sha256": base.reference_prepared_hash(
                        {
                            **e,
                            "attention_mask": [1] * len(e["input_ids"]),
                            "use_cache": False,
                            "logits_to_keep": 1,
                        }
                    ),
                    "answer_token_ids": e["answer_token_ids"],
                }
                for s, e in zip(samples, examples, strict=True)
            ]
            require(
                all(set(b["answer_token_ids"]) == {"yes", "no", "U"} for b in bindings),
                "legal2plusU",
            )
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
                peft.PeftModel.from_pretrained(
                    model_base,
                    source["before"]["path"],
                    adapter_name="before64",
                    is_trainable=False,
                )
                .to("cuda")
                .eval()
            )
            model.load_adapter(source["after"]["path"], adapter_name="after64", is_trainable=False)
            model.requires_grad_(False)
            arms = {}
            with torch.inference_mode(), (a.output / "records.jsonl").open("x") as sidecar:
                for label in ["before", "after"]:
                    model.set_adapter(label + "64")
                    model.eval()
                    model.requires_grad_(False)
                    records = []
                    for binding, example in zip(bindings, examples, strict=True):
                        ids = torch.tensor([example["input_ids"]], device="cuda", dtype=torch.long)
                        media = {
                            k: example[k].to("cuda")
                            for k in base.MEDIA_FORWARD_KEYS
                            if k in example
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
                        require(bool(torch.isfinite(logits).all()), "finite logits")
                        raw = {
                            t: float(logits[i].item())
                            for t, i in example["answer_token_ids"].items()
                        }
                        readout = base.read_probabilities(raw, ["yes", "no"], temperature=1)
                        logodds = raw["yes"] - raw["no"]
                        row = {
                            **binding,
                            "arm": label,
                            "adapter_sha256": source[label]["weights_sha256"],
                            "raw_logits": raw,
                            "probabilities": readout.probabilities,
                            "prediction": readout.prediction,
                            "joint_ce": -math.log(readout.joint_probabilities[binding["target"]]),
                            "logodds": logodds,
                            "gold_margin": logodds * (1 if binding["target"] == "yes" else -1),
                        }
                        records.append(row)
                        sidecar.write(json.dumps(row, sort_keys=True) + "\n")
                        sidecar.flush()
                        del output, logits, ids, media
                    arms[label] = records
            require(
                all(
                    all(
                        left[k] == right[k]
                        for k in [
                            "id",
                            "target",
                            "sample_sha256",
                            "input_sha256",
                            "answer_token_ids",
                        ]
                    )
                    for left, right in zip(arms["before"], arms["after"], strict=True)
                ),
                "all12input/mapping equal",
            )
            require(
                not any(p.grad is not None or p.requires_grad for p in model.parameters()),
                "read-only no gradients",
            )
            report.update(
                status="completed",
                exit_code=0,
                arms={
                    label: metrics(rows, source["selected_pairs"]) for label, rows in arms.items()
                },
                parity_of_inputs_verified=12,
                full_base_loads=1,
                forward_count=24,
                gradients_present=False,
                records_sha256=sha(a.output / "records.jsonl"),
                dtype="bfloat16",
            )
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
