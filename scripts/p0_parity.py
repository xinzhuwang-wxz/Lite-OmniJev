"""Sequential CUDA HF reference and vLLM raw-token parity; never starts a server."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any

import httpx

from scripts.assets import write_json
from scripts.check_tokens import OfficialTokenizerAdapter
from train.readout import read_probabilities
from train.trainer import (
    SmokeConfig,
    code_revision,
    data_bytes,
    prepare_examples,
    runtime_requirements,
    smoke_samples,
    verified_assets,
)

THRESHOLD = 1e-3


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(
    config: SmokeConfig, model_dir: Path, manifest: Path, data: Path, adapter: Path | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assets = verified_assets(model_dir, manifest)
    samples = smoke_samples(config, data)
    if len({sample.id for sample in samples}) != 100:
        raise ValueError("exactly 100 unique sample IDs required")
    tokenizer = OfficialTokenizerAdapter(model_dir)
    examples = prepare_examples(samples, tokenizer, config.max_length)
    if any(len(example["input_ids"]) >= config.max_length for example in examples):
        raise ValueError("prompt must leave room for one generated token")
    adapter_hashes = None
    if adapter is not None:
        required = [adapter / "adapter_config.json", adapter / "adapter_model.safetensors"]
        if not all(path.is_file() for path in required):
            raise ValueError("saved PEFT safetensors adapter and config required")
        adapter_hashes = {path.name: file_hash(path) for path in required}
    provenance = {
        "model_revision": model_dir.name,
        "implementation_hashes": {
            name: file_hash(Path(__file__).resolve().parents[1] / name)
            for name in (
                "scripts/p0_parity.py",
                "train/template.py",
                "train/readout.py",
                "scripts/check_tokens.py",
                "train/trainer.py",
            )
        },
        "asset_hashes": assets,
        "manifest_sha256": file_hash(manifest),
        "data_file_sha256": file_hash(data),
        "data_sha256": hashlib.sha256(data_bytes(samples)).hexdigest(),
        "template_sha256": hashlib.sha256(tokenizer.template.encode()).hexdigest(),
        "config": config.model_dump(),
        "mode": "adapter" if adapter else "base",
        "adapter_hashes": adapter_hashes,
        "inputs_sha256": digest(examples),
        "dtype": "bfloat16",
        "thinking": False,
        "readout": "legal_joint_U_and_conditional_answers_v1",
    }
    return provenance, examples


def row(example: dict[str, Any], logprobs: list[float]) -> dict[str, Any]:
    ids = example["token_ids"]
    if len(logprobs) != len(ids) or any(
        type(value) not in (int, float) or not math.isfinite(value) or value > 0 or value <= -9999
        for value in logprobs
    ):
        raise ValueError("complete finite unclipped raw answer logprobs required")
    labels = [str(token) for token in ids[:-1]]
    result = read_probabilities(dict(zip([*labels, "U"], logprobs, strict=True)), labels)
    return {
        "id_sha256": digest(example["id"]),
        "input_sha256": digest(example["input_ids"]),
        "token_ids": ids,
        "raw_logprobs": logprobs,
        "raw_probabilities": [math.exp(value) for value in logprobs],
        "conditional": list(result.probabilities.values()),
        "p_U": result.p_insufficient,
    }


def validate_reference(
    reference: dict[str, Any], provenance: dict[str, Any], examples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if (
        reference.get("status") != "reference-ready"
        or reference.get("action") != "hf-reference"
        or reference.get("provenance") != provenance
    ):
        raise ValueError("reference status/action/provenance mismatch")
    rows = reference.get("rows", [])
    if len(rows) != 100 or len(examples) != 100:
        raise ValueError("reference must contain all 100 rows")
    for saved, example in zip(rows, examples, strict=True):
        if saved != row(example, saved["raw_logprobs"]):
            raise ValueError("reference row identity/tokens/probabilities mismatch")
    return list(rows)


def compare_rows(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    if len(left) != 100 or len(right) != 100:
        raise ValueError("parity requires all 100 rows")
    maximum = 0.0
    raw_maximum = 0.0
    for a, b in zip(left, right, strict=True):
        if any(a[key] != b[key] for key in ("id_sha256", "input_sha256", "token_ids")):
            raise ValueError("parity row identity or token mismatch")
        differences = [abs(x - y) for x, y in zip(a["conditional"], b["conditional"], strict=True)]
        maximum = max(maximum, *differences, abs(a["p_U"] - b["p_U"]))
        raw_maximum = max(
            raw_maximum,
            *(
                abs(x - y)
                for x, y in zip(a["raw_probabilities"], b["raw_probabilities"], strict=True)
            ),
        )
    passed = maximum < THRESHOLD and raw_maximum < THRESHOLD
    return {
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "sample_count": 100,
        "max_conditional_and_U_difference": maximum,
        "max_raw_probability_difference": raw_maximum,
        "threshold_strictly_less_than": THRESHOLD,
    }


def hf_reference(
    config: SmokeConfig,
    model_dir: Path,
    adapter: Path | None,
    examples: list[dict[str, Any]],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    report["versions"] = runtime_requirements()  # Reject CPU/MPS before loading weights.
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if adapter is not None:
        peft = importlib.import_module("peft")
        model = peft.PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.requires_grad_(False).eval().to("cuda")
    report["device"] = torch.cuda.get_device_name(0)
    report["cuda_version"] = torch.version.cuda
    rows = []
    try:
        with torch.inference_mode():
            for example in examples:
                ids = torch.tensor([example["input_ids"]], device="cuda", dtype=torch.long)
                output = model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
                raw = output.logits[0, -1].float().log_softmax(-1)
                rows.append(row(example, raw[example["token_ids"]].cpu().tolist()))
    finally:
        del model
        torch.cuda.empty_cache()
    return rows


def vllm_rows(
    client: httpx.Client,
    url: str,
    served_model: str,
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    models = client.get(url.rstrip("/") + "/v1/models")
    models.raise_for_status()
    if served_model not in {item["id"] for item in models.json()["data"]}:
        raise ValueError("attested model not advertised by vLLM")
    rows = []
    for example in examples:
        response = client.post(
            url.rstrip("/") + "/v1/completions",
            json={
                "model": served_model,
                "prompt": example["input_ids"],
                "max_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "logprobs": 1,
                "logprob_token_ids": example["token_ids"],
                "return_tokens_as_token_ids": True,
                "seed": 42,
            },
        )
        response.raise_for_status()
        body = response.json()
        if body["usage"]["prompt_tokens"] != len(example["input_ids"]):
            raise ValueError("upstream prompt token count mismatch")
        raw = body["choices"][0]["logprobs"]["top_logprobs"][0]
        rows.append(row(example, [raw[f"token_id:{token}"] for token in example["token_ids"]]))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("hf-reference", "compare-vllm"))
    parser.add_argument("--config", type=Path, default=Path("configs/train-smoke.json"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--reference", type=Path, help="saved HF report; required for compare-vllm")
    parser.add_argument("--server-record", type=Path, help="verified vLLM launch attestation JSON")
    parser.add_argument("--upstream-url", default="http://127.0.0.1:8001")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    start = time.monotonic()
    report: dict[str, Any] = {
        "action": args.action,
        "status": "blocked",
        "exit_code": 1,
        "code_revision": code_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cost": None,
        "scope": "P0 text parity only; base and adapter reports both required; not P0 completion",
    }
    try:
        if args.report.exists():
            raise FileExistsError("report already exists; choose a new evidence path")
        config = SmokeConfig.model_validate_json(args.config.read_text())
        provenance, examples = prepare(
            config, args.model_dir.resolve(), args.asset_manifest, args.data, args.adapter
        )
        report["provenance"] = provenance
        if args.action == "hf-reference":
            report["rows"] = hf_reference(config, args.model_dir, args.adapter, examples, report)
            report.update(status="reference-ready", exit_code=0)
        else:
            if args.reference is None or args.server_record is None:
                raise ValueError("compare requires --reference and --server-record")
            reference = json.loads(args.reference.read_text())
            left = validate_reference(reference, provenance, examples)
            server = json.loads(args.server_record.read_text())
            if (
                server.get("provenance") != provenance
                or server.get("logprobs_mode") != "raw_logprobs"
                or server.get("dtype") != "bfloat16"
                or not isinstance(server.get("served_model"), str)
                or not server["served_model"].strip()
                or not server.get("launch_command")
            ):
                raise ValueError("server launch attestation missing or mismatched")
            report["reference_sha256"] = file_hash(args.reference)
            report["server_record_sha256"] = file_hash(args.server_record)
            report["server_identity_verification"] = "operator-attested; not remote attestation"
            with httpx.Client(timeout=120) as client:
                right = vllm_rows(client, args.upstream_url, server["served_model"], examples)
            report["rows"] = right
            report.update(compare_rows(left, right))
    except (Exception, KeyboardInterrupt) as exc:
        # Exception bodies can contain prompts/server payloads; persist only the type.
        report["error_type"] = type(exc).__name__
    report["duration_seconds"] = time.monotonic() - start
    if not args.report.exists():
        write_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "action")}))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
