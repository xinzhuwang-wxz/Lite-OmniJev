"""Finalize only TRAIN controlled-pair prepared hashes on CPU; never loads a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import tarfile
import time
from pathlib import Path
from typing import Any

from scripts.check_tokens import OfficialTokenizerAdapter
from scripts.s1_text_pools import canonical_sample_sha
from train.checkpoint import file_sha256
from train.multimodal import configure_image_processor
from train.trainer import (
    BoundedPreparedExamples,
    LoopConfig,
    claim_pair_row,
    code_revision,
    data_bytes,
    load_claim_pairs,
    prepare_mixed_example,
    reference_order,
    smoke_samples,
    verified_init_adapter,
)

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
MODEL = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
SOURCE_COMMIT = "3fe0ab2a3096f125ecbdf31426dc5d50296e386e"
ARCHIVE_SHA = "9132c2d552af09aa4569c33df3b370fc61abe5c61eb34fa2e235b795a28f2378"
REPORT_SHA = "65cd76fbd84b206fcb864c40312be271cc3da5dca2ded293e4356d8a3650832b"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def write_new(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-dir", type=Path, default=ROOT / "artifacts/s4-claim-pair-pilot-001"
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=Path("/root/autodl-tmp/omnijev-s4-claim-pair-001.tar"),
    )
    args = parser.parse_args()
    os.umask(0o077)
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    start = time.monotonic()
    out = args.pilot_dir.resolve()
    receipt_path = out / "pair-finalization.json"
    sidecar = out / "pair-sidecar.json"
    pair_path = out / "config-pair01.json"
    report: dict[str, Any] = {
        "status": "failed",
        "exit_code": 1,
        "gpu_used": False,
        "cost": None,
        "script_sha256": file_sha256(Path(__file__)),
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": ARCHIVE_SHA,
        "observed_git_revision": code_revision(),
        "output_sha256": {},
    }

    def interrupted(signum: int, frame: Any) -> None:
        raise RuntimeError(f"CPU finalization interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with receipt_path.open("x", encoding="utf-8") as receipt:
        try:
            require(not sidecar.exists() and not pair_path.exists(), "pair artifacts already exist")
            require(file_sha256(args.source_archive) == ARCHIVE_SHA, "source archive SHA mismatch")
            require(
                Path(claim_pair_row.__code__.co_filename).resolve()
                == Path("train/trainer.py").resolve(),
                "trainer imported outside declared snapshot",
            )
            with tarfile.open(args.source_archive) as archive:
                for name in (
                    "train/trainer.py",
                    "train/pair_objective.py",
                    "train/template.py",
                    "train/multimodal.py",
                ):
                    archived = archive.extractfile(name)
                    require(
                        archived is not None
                        and hashlib.sha256(archived.read()).hexdigest() == file_sha256(Path(name)),
                        "runtime source differs from pinned archive: " + name,
                    )
            source_report = out / "report.json"
            require(
                file_sha256(source_report) == REPORT_SHA, "pilot preparation report SHA mismatch"
            )
            prepared = read(source_report)
            require(
                prepared["status"] == "prepared_pending_pair_hashes"
                and prepared["exit_code"] == 0
                and prepared["draws"] == 120
                and prepared["pair_count"] == 20
                and prepared["gpu_used"] is False
                and prepared["expanded_isolation"] == prepared["pilot_isolation"] == "isolated",
                "pilot is not the isolated CPU-prepared120draw/20pair source",
            )
            outputs = prepared["output_sha256"]
            require(isinstance(outputs, dict) and bool(outputs), "pilot output hashes missing")
            for name, expected in outputs.items():
                require(
                    Path(name).name == name and file_sha256(out / name) == expected,
                    "pilot output SHA mismatch: " + name,
                )
            for name in ("inventory-audit.json", "inventory-expanded-audit.json"):
                require(
                    read(out / name)["isolation"]["outcome"] == "isolated",
                    "inventory isolation summary failed",
                )
            report["preparation_report_sha256"] = REPORT_SHA
            report["input_sha256"] = dict(outputs)
            ce_raw = read(out / "config-ce0.json")
            ce = LoopConfig.model_validate(ce_raw)
            require(
                ce.pair_loss_weight == 0
                and ce.reference_kl_weight == ce.brier_weight == 0
                and ce.image_max_soft_tokens == 1120
                and ce.video_num_frames == 16
                and ce.preparation_mode == "lazy"
                and bool(ce.media_roots),
                "CE input preparation/weights differ from frozen pilot",
            )
            identity = prepared["init_adapter"]
            require(
                verified_init_adapter(Path(identity["path"]), identity["weights_sha256"])
                == identity
                and identity["weights_sha256"]
                == "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818",
                "pilot init adapter identity changed",
            )
            data = out / "train.jsonl"
            samples = smoke_samples(ce, data, expected_data_versions=ce.expected_data_versions)
            order = reference_order(ce, len(samples))
            recipe = read(out / "draw-recipe.json")
            require(len(recipe["draws"]) == 120, "draw recipe count mismatch")
            for draw, index in enumerate(order):
                row = recipe["draws"][draw]
                require(
                    row["draw"] == draw
                    and row["data_index"] == index
                    and row["item_id"] == samples[index].id
                    and row["sample_sha256"] == canonical_sample_sha(samples[index]),
                    "draw recipe canonical sample/order mismatch",
                )
            transformers = importlib.import_module("transformers")
            processor = transformers.AutoProcessor.from_pretrained(
                str(MODEL), local_files_only=True, trust_remote_code=False
            )
            configure_image_processor(processor, ce.image_max_soft_tokens)
            tokenizer = OfficialTokenizerAdapter(MODEL)
            roots = [Path(p) for p in ce.media_roots or []]
            examples = BoundedPreparedExamples(
                samples,
                lambda sample: prepare_mixed_example(
                    sample,
                    tokenizer,
                    processor,
                    roots[0],
                    max_length=ce.max_length,
                    extra_roots=roots[1:],
                    video_num_frames=ce.video_num_frames,
                ),
                ce.preparation_cache_bytes,
                report,
            )
            rows = []
            for step in range(10):
                for offset in (0, 2):
                    draw = step * 12 + offset
                    pi, ni = order[draw : draw + 2]
                    rows.append(
                        claim_pair_row(
                            draw, pi, ni, samples[pi], samples[ni], examples[pi], examples[ni]
                        )
                    )
            canonical_sha = hashlib.sha256(data_bytes(samples)).hexdigest()
            ledger = out / "allocation-ledger.jsonl"
            payload = {
                "header": {
                    "schema": "train-claim-pairs-v1",
                    "data_sha256": canonical_sha,
                    "ledger_sha256": file_sha256(ledger),
                    "seed": 42,
                    "steps": 10,
                    "accumulation": 12,
                },
                "pairs": rows,
            }
            write_new(sidecar, payload)
            pair_fields = {
                "pair_loss_weight": 0.1,
                "pair_sidecar": str(sidecar),
                "pair_sidecar_sha256": file_sha256(sidecar),
                "pair_ledger": str(ledger),
                "pair_ledger_sha256": file_sha256(ledger),
            }
            require(
                not any(key in ce_raw for key in pair_fields), "CE config already has pair fields"
            )
            pair_raw = {**ce_raw, **pair_fields}
            require(
                {key: value for key, value in pair_raw.items() if key not in pair_fields} == ce_raw,
                "non-pair CE configuration changed",
            )
            pair = LoopConfig.model_validate(pair_raw)
            require(
                len(load_claim_pairs(pair, samples, examples, data)) == 20,
                "real CPU pair validation did not cover20pairs",
            )
            write_new(pair_path, pair_raw)
            for name, expected in outputs.items():
                require(
                    file_sha256(out / name) == expected,
                    "original pilot output changed during finalization",
                )
            report.update(
                status="validated",
                exit_code=0,
                canonical_data_sha256=canonical_sha,
                prepared_pair_draws=40,
                pairs=20,
                init_adapter=identity,
                code_sha256={
                    name: file_sha256(Path(name))
                    for name in (
                        "train/trainer.py",
                        "train/pair_objective.py",
                        "train/template.py",
                        "train/multimodal.py",
                    )
                },
                output_sha256={
                    "pair-sidecar.json": file_sha256(sidecar),
                    "config-pair01.json": file_sha256(pair_path),
                },
            )
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        finally:
            report["duration_seconds"] = time.monotonic() - start
            json.dump(report, receipt, indent=2, sort_keys=True, allow_nan=False)
            receipt.write("\n")
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in ("status", "exit_code", "output_sha256", "duration_seconds", "error")
            }
        )
    )
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
