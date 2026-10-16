"""Select a balanced fixed12-family TRAIN diagnostic; never consult model/dev outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample
from scripts.train_s1 import verify_data_is_train_pool
from train.checkpoint import file_sha256
from train.trainer import LoopConfig, data_bytes, smoke_samples, verified_init_adapter

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = ROOT / "artifacts/s4-temporal-curriculum-pilot-001"
SOURCE_REPORT_SHA = "e3b48c872e2a0d38675dd60d9e3bf073f73a85ad1a1f28d4825fa5c28a4b848b"
INIT_SHA = "4b8e422f3ca29e863999d2cd713bb9fd3f0a46e5360faa259a3bb8a4a9814bed"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def select_train(samples: list[Sample]) -> list[Sample]:
    require(all(s.meta.pool == "train" for s in samples), "selection accepts TRAIN only")
    candidates = [
        s
        for s in samples
        if len(s.questions) == 1
        and s.questions[0].type == "noul"
        and isinstance(s.questions[0].target, HardTarget)
        and len(s.prefix.media) == 1
        and s.prefix.media[0].type == "video"
        and set(s.questions[0].instruction.lower().split()) & {"before", "after", "when", "while"}
    ]
    ranked = sorted(
        candidates,
        key=lambda s: hashlib.sha256(("42:s4-train-learnability:" + s.id).encode()).hexdigest(),
    )
    selected = []
    families = set()
    for gold in ("yes", "no"):
        chosen = []
        for candidate in ranked:
            if (
                candidate.questions[0].target.value == gold
                and candidate.meta.family_id not in families
            ):
                chosen.append(candidate)
                families.add(candidate.meta.family_id)
                if len(chosen) == 6:
                    break
        require(
            len(chosen) == 6, "insufficient independent TRAIN temporal families for balanced6yes6no"
        )
        selected.extend(chosen)
    return sorted(
        selected,
        key=lambda s: hashlib.sha256(("42:s4-train-learnability:" + s.id).encode()).hexdigest(),
    )


def write(path: Path, payload: Any) -> None:
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/s4-train-learnability-001")
    args = parser.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    started = time.monotonic()
    report = {
        "status": "failed",
        "exit_code": 1,
        "gpu_used": False,
        "training_started": False,
        "script_sha256": file_sha256(Path(__file__)),
        "output_sha256": {},
    }
    try:
        require(
            file_sha256(SOURCE / "report.json") == SOURCE_REPORT_SHA,
            "audited source report SHA changed",
        )
        source = json.loads((SOURCE / "report.json").read_bytes())
        require(
            source["status"] == "ready_for_preflight"
            and source["exit_code"] == 0
            and source["inventory_isolation"] == "isolated"
            and source["source_admission"] == "verified_existing_scope",
            "source is not admitted isolated TRAIN",
        )
        for name, digest in source["output_sha256"].items():
            require(
                Path(name).name == name and file_sha256(SOURCE / name) == digest,
                "source output changed:" + name,
            )
        config = LoopConfig.model_validate_json((SOURCE / "config-ce.json").read_bytes())
        samples = smoke_samples(
            config, SOURCE / "train.jsonl", expected_data_versions=config.expected_data_versions
        )
        selected = select_train(samples)
        init = source["init_adapter"]
        require(
            init["weights_sha256"] == INIT_SHA
            and verified_init_adapter(Path(init["path"]), INIT_SHA) == init,
            "CE0 init changed",
        )
        raw = json.loads((SOURCE / "config-ce.json").read_bytes())
        raw.pop("expected_data_versions", None)
        raw.update(
            sample_count=12,
            steps=10,
            accumulation=12,
            modality="video",
            learning_rate=2e-5,
            expected_data_version=selected[0].meta.data_version,
            midway_step=5,
            checkpoint_every=5,
        )
        require(len({s.meta.data_version for s in selected}) == 1, "TRAIN video versions differ")
        LoopConfig.model_validate(raw)
        train = args.output / "train.jsonl"
        with train.open("xb") as stream:
            stream.write(data_bytes(selected))
        ledger = args.output / "allocation-ledger.jsonl"
        with ledger.open("xb") as stream:
            stream.write((SOURCE / "allocation-ledger.jsonl").read_bytes())
        verify_data_is_train_pool(train, [ledger])
        write(args.output / "config-ce.json", raw)
        report.update(
            status="ready_for_preflight",
            exit_code=0,
            source_report_sha256=SOURCE_REPORT_SHA,
            source_data_sha256=source["output_sha256"]["train.jsonl"],
            source_isolation="isolated",
            isolation_basis="exact canonical TRAIN subset of audited source; no new samples/media",
            samples=12,
            unique_families=12,
            gold_counts={"yes": 6, "no": 6},
            draws=120,
            init_adapter=init,
            selected_ids=[s.id for s in selected],
            arm_wallclock_limit_seconds=240,
            before_after_budget_seconds=90,
            output_sha256={
                name: file_sha256(args.output / name)
                for name in ("train.jsonl", "allocation-ledger.jsonl", "config-ce.json")
            },
        )
    except Exception as exc:
        report.update(status="failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
    report["duration_seconds"] = time.monotonic() - started
    write(args.output / "report.json", report)
    print(json.dumps(report, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
