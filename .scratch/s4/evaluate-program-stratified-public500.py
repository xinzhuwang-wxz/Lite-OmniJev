"""One explicit complete public DEV cohort per invocation, fixed step100adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import tarfile
import time
from pathlib import Path

from data.schema import HardTarget, Sample, answer_tokens

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = Path("/root/autodl-tmp/s4-monitored-eval-code-001")
PYTHON = "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python"
ARCHIVE = Path("/root/autodl-tmp/omnijev-s4-monitored-eval-001.tar")
ARCHIVE_SHA = "b8c5b3184ea2d09e6bb1184c7065c27fd6d354022dcf0e37169e6e390f1d6516"
TRAIN_REPORT = ROOT / "reports/s4-program-stratified-training-step0100.json"
TRAIN_SHA = "421b950611a8e8db3fa1b0ae75c960790ae56c29a80a218976c171fbdf67ae97"
ADAPTER_SHA = "e46550143b28f8798484515f0abe21996fe92ad241625a4cf6da15dc093277fd"
PROTOCOLS = {
    "image": "2539b353fb2f4e499d3609ad7b9a366f467ec381fe46d2b3bac0295ea4405690",
    "audio": "d1d7a95082c6b929bd978b5d07cf7df99c035a7b33e42b916f94f22d7dfb4731",
    "text": "a5d204e1ef63ee1e9a905a11d6cf14a432388a78808cf460d627606a6bff55de",
    "screenshot": "87d6985272f3a28dc1f108eee1f38026e12951a1c18ebde13868b82423fca5d6",
}
CROSS_HELPER = Path(
    "/root/autodl-tmp/s4-motion-code-142e744/.scratch/s4/evaluate-coverage-crossmodal-full.py"
)
CROSS_HASHES = {
    "evaluate-coverage-crossmodal-full.py": (
        "7176d167d2a1ab385d650e57d39c44393d8cb8f85e7d97f856e268f05610b80a"
    ),
    "screen-crossmodal-retention.py": (
        "0c9c616f90f4fb942dbfa5f96f7abd64d238796ff8b8c7b073b7c5deb07ed379"
    ),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def public500(data):
    samples = [
        Sample.model_validate_json(line) for line in data.read_text().splitlines() if line.strip()
    ]
    require(
        len(samples) == len({s.id for s in samples}) == 500,
        "requires actual500unique original samples",
    )
    require(
        all(
            s.meta.pool == "dev"
            and len(s.questions) == 1
            and isinstance(s.questions[0].target, HardTarget)
            for s in samples
        ),
        "public DEV hard gold only",
    )
    return samples


def losses(records, samples, cross=False):
    by_id = {s.id: s for s in samples}
    require(
        len(records) == 500 and {r["id"] for r in records} == set(by_id), "complete500readout IDs"
    )
    conditional = joint = 0.0
    for row in records:
        question = by_id[row["id"]].questions[0]
        gold = str(question.target.value)
        tokens = answer_tokens(question)
        p = row["conditional_without_u"] if cross else row["probabilities"]
        raw = row["raw_logprobs"] if cross else row["raw_logits"]
        require(
            set(p) == set(tokens) and set(raw) == set(tokens) | {"U"}, "complete legal+Ureadout"
        )
        require(
            all(math.isfinite(v) for v in raw.values()) and 0 < p[gold] <= 1, "finite probabilities"
        )
        largest = max(raw.values())
        joint += (
            math.log(math.fsum(math.exp(v - largest) for v in raw.values())) + largest - raw[gold]
        )
        conditional -= math.log(p[gold])
        if cross:
            require(tokens[row["target_index"]] == gold, "official cross target order")
        else:
            require(str(row["target"]) == gold and row["pool"] == "dev", "official original target")
    return {"conditional_nll": conditional / 500, "joint_legal_plus_u_nll": joint / 500}


def code_bindings():
    require(sha(ARCHIVE) == ARCHIVE_SHA, "fixed official runtime archive")
    names = [
        "scripts/s1_dev_eval.py",
        "scripts/check_tokens.py",
        "train/template.py",
        "train/readout.py",
        "train/trainer.py",
        "train/multimodal.py",
        "scripts/s0_development_baselines.py",
        "scripts/s0_public_baselines.py",
        "eval/metrics.py",
        "data/schema.py",
    ]
    result = {}
    with tarfile.open(ARCHIVE) as archive:
        for name in names:
            members = [
                m for m in archive.getmembers() if m.name == name or m.name.endswith("/" + name)
            ]
            require(len(members) == 1, "one archived runtime source")
            stream = archive.extractfile(members[0])
            require(stream is not None, "archived file")
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            require(sha(SOURCE / name) == digest, "runtime source changed: " + name)
            result[name] = digest
    return result


def load_cross():
    for name, digest in CROSS_HASHES.items():
        require(sha(CROSS_HELPER.with_name(name)) == digest, "fixed full-cross helper")
    spec = importlib.util.spec_from_file_location("public500_existing_cross", CROSS_HELPER)
    require(spec is not None and spec.loader is not None, "cross helper loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(cohort, cross_source, execute):
    require(sha(TRAIN_REPORT) == TRAIN_SHA, "actual terminal100step report")
    trained = json.loads(TRAIN_REPORT.read_bytes())
    require(
        trained["status"] == "completed"
        and trained["exit_code"] == 0
        and trained["steps_completed"] == 100
        and trained["adapter_sha256"] == ADAPTER_SHA,
        "fixed completed100candidate",
    )
    adapter = Path(trained["adapter_dir"])
    require(sha(adapter / "adapter_model.safetensors") == ADAPTER_SHA, "actual adapter weights")
    source_hashes = code_bindings()
    cross = None
    if cohort == "crossmodal":
        cross = load_cross()
        data, media = cross.cross_source(cross_source)
        require(sha(data) == cross.CROSS[cross_source][0], "fixed fullcross dataSHA")
        samples = public500(data)
        samples = sorted(
            samples, key=lambda s: hashlib.sha256(("42:s4-crossmodal:" + s.id).encode()).hexdigest()
        )
        model = cross.BASE
        protocol = {"source": cross_source, "data_sha256": sha(data), "helper_sha256": CROSS_HASHES}
    else:
        reference = ROOT / "reports" / f"s3-dev-eval-{cohort}-stable-001.json"
        require(sha(reference) == PROTOCOLS[cohort], "actual passed S3protocol reference")
        previous = json.loads(reference.read_bytes())
        require(
            previous["status"] == "passed"
            and previous["exit_code"] == 0
            and previous["samples"] == 500,
            "full source protocol",
        )
        command = previous["command"]

        def argument(flag):
            return command[command.index(flag) + 1] if flag in command else None

        data = Path(argument("--data"))
        model = Path(argument("--model-dir"))
        media = Path(argument("--media-root")) if argument("--media-root") else None
        require(sha(data) == previous["data_sha256"], "original full cohort bytes")
        samples = public500(data)
        if cohort in ["image", "screenshot"]:
            require(
                previous["image_max_soft_tokens"] == {"requested": 1120, "effective": 1120},
                "original1120protocol",
            )
        protocol = {
            "report_path": str(reference),
            "report_sha256": PROTOCOLS[cohort],
            "historical_adapter_sha256": previous["adapter_sha256"],
            "usage": "protocol only, not current parent scores",
        }
    require(model.name == trained["model_revision"], "fixed base revision")
    require(media is None or media.is_dir(), "missing actual media root")
    for name, digest in trained["asset_hashes"].items():
        if execute or name != "model.safetensors":
            require(sha(model / name) == digest, "model/tokenizer/template asset changed")
    receipt = {
        "status": "ready_for_public500",
        "exit_code": 0,
        "cohort": cohort,
        "samples": 500,
        "passes": 1,
        "counterfactual": False,
        "data_path": str(data),
        "data_sha256": sha(data),
        "media_root": str(media) if media else None,
        "model_revision": model.name,
        "adapter_dir": str(adapter),
        "adapter_sha256": ADAPTER_SHA,
        "adapter_config_sha256": sha(adapter / "adapter_config.json"),
        "training_report_sha256": TRAIN_SHA,
        "model_asset_sha256": trained["asset_hashes"],
        "source_protocol": protocol,
        "source_commit": "74e96e2",
        "source_archive_sha256": ARCHIVE_SHA,
        "source_file_sha256": source_hashes,
        "dtype": "bfloat16",
        "image_max_soft_tokens": image_budget(cohort),
        "cost": None,
        "stage_gate": False,
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(s.id for s in samples).encode()
        ).hexdigest(),
        "cpu_media_bytes_read": False,
        "model_weights_rehashed": execute,
        "historical_tensor_identity": "not_verified; originalS1sidecars lack complete input hashes",
        "quality_inheritance": False,
    }
    return receipt, samples, data, media, model, adapter, cross


def image_budget(cohort):
    return None if cohort == "text" else 1120


def evaluator_command(cohort, model, adapter, data, media, output, budget_seconds):
    command = [
        "timeout",
        "--signal=INT",
        "--kill-after=20s",
        str(budget_seconds),
        PYTHON,
        "-m",
        "scripts.s1_dev_eval",
        "--model-dir",
        str(model),
        "--adapter",
        str(adapter),
        "--data",
        str(data),
        "--passes",
        "1",
        "--output-dir",
        str(output / "eval"),
        "--report",
        str(output / "report.json"),
    ]
    if media:
        command.extend(["--media-root", str(media)])
    if image_budget(cohort) is not None:
        command.extend(["--image-max-soft-tokens", str(image_budget(cohort))])
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=[*PROTOCOLS, "crossmodal"], required=True)
    ap.add_argument("--cross-source", choices=["original", "supplemental"], default="original")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--budget-seconds", type=int, choices=[420, 600], default=600)
    a = ap.parse_args()
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if not a.execute:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    a.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {
        "status": "failed",
        "exit_code": 1,
        "gpu_requested": a.execute,
        "budget_seconds": a.budget_seconds,
        "runner_sha256": sha(Path(__file__)),
        "cost": None,
    }
    try:
        prepared, samples, data, media, model, adapter, cross = prepare(
            a.cohort, a.cross_source, a.execute
        )
        receipt.update(prepared)
        if a.execute:
            require(
                not subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
                ).strip(),
                "GPU busy; no concurrent evaluation",
            )
            receipt.update(status="running", phase="loading")
            (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if cross:

                def interrupted(signum, frame):
                    raise TimeoutError("fullcross budget expired")

                signal.signal(signal.SIGALRM, interrupted)
                signal.alarm(a.budget_seconds)
                evalout = a.output / "eval"
                evalout.mkdir()
                try:
                    result = cross.full_forward(samples, media, adapter, evalout, {})
                finally:
                    signal.alarm(0)
                record_path = evalout / "records.jsonl"
                records = [json.loads(line) for line in record_path.read_text().splitlines()]
                metrics = result["metrics"]
                receipt["cross_forward_result"] = result
            else:
                command = evaluator_command(
                    a.cohort, model, adapter, data, media, a.output, a.budget_seconds
                )
                receipt["command"] = command
                with (a.output / "run.log").open("x") as log:
                    process = subprocess.run(
                        command,
                        cwd=SOURCE,
                        env=os.environ.copy(),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                require(process.returncode == 0, "official evaluator failed or exceeded budget")
                result = json.loads((a.output / "report.json").read_bytes())
                require(
                    result["status"] == "passed"
                    and result["exit_code"] == 0
                    and result["samples"] == 500
                    and result["passes"] == 1,
                    "complete single-pass terminal result",
                )
                require(
                    result["data_sha256"] == receipt["data_sha256"]
                    and result["adapter_sha256"] == ADAPTER_SHA,
                    "terminal data/adapter binding",
                )
                record_path = a.output / "eval/records.jsonl"
                require(sha(record_path) == result["sidecar_sha256"], "terminal sidecar SHA")
                records = [
                    json.loads(line)
                    for line in record_path.read_text().splitlines()
                    if line.strip()
                ]
                records = [
                    r for r in records if r.get("pass") == 0 and r.get("variant") == "original"
                ]
                metrics = result["metrics"]
            require(
                metrics["count"] == 500
                and all(k in metrics for k in ["accuracy", "brier", "ece15", "coverage_at_error"]),
                "complete legal metric set",
            )
            metrics.update(losses(records, samples, cross is not None))
            require(sha(adapter / "adapter_model.safetensors") == ADAPTER_SHA, "adapter changed")
            receipt.update(
                status="completed",
                exit_code=0,
                metrics=metrics,
                records_sha256=sha(record_path),
                progress_every=25,
            )
    except Exception as exc:
        receipt.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        (a.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    raise SystemExit(receipt["exit_code"])


if __name__ == "__main__":
    main()
