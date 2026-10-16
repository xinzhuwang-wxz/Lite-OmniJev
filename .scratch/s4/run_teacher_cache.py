"""I5 teacher cache: OneJev-27B full legal(+U) readout over the fixed TRAIN slice.

GPU runner scheduled by main. Reuses the proven s0 baseline readout path
(official chat template, thinking off, single-token readout, answer-token
mapping) but binds to the pinned slice sha instead of the dev build report,
which is what lets a TRAIN slice through without touching the dev binding.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SLICE_SHA = "6b224886ac62a36d8b67d53158dd896b9f3c4b39bdd74c4551a0cba8d571c94a"
MODEL_ID = "onejev-27b"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, default=Path("configs/project-model-assets.json"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=1800)
    args = parser.parse_args()
    if sha256_file(args.data) != SLICE_SHA:
        raise SystemExit("slice sha mismatch; refusing to cache an unpinned input")
    import os
    import sys

    # Run from a repo checkout (cwd) so the proven baseline modules resolve;
    # this script itself lives in the slice artifacts directory.
    sys.path.insert(0, os.getcwd())
    from scripts import s0_development_baselines as baseline
    from scripts.s0_public_baselines import model_evidence

    started = time.time()
    report: dict = {
        "schema": "lite-omnijev.s6-teacher-cache-150.v1",
        "status": "failed",
        "model_id": MODEL_ID,
        "slice_sha256": SLICE_SHA,
        "gpu_used": True,
    }
    try:
        model, directory = model_evidence(
            args.asset_manifest, args.model_root, MODEL_ID, require_weights=True
        )
        samples = baseline.load_samples(args.data)
        # A pinned TRAIN slice, not a dev-500 build: skip the 500-row dev
        # contract, keep the identity-uniqueness part of it.
        if len({sample.id for sample in samples}) != len(samples):
            raise baseline.DevelopmentBaselineError("duplicate sample ids in slice")
        if len(samples) != 150:
            raise SystemExit(f"expected 150 slice rows, found {len(samples)}")
        processor = baseline.default_processor_factory(directory)
        forward, cleanup = baseline.real_forward_factory(directory, MODEL_ID, "bfloat16")
        args.records.parent.mkdir(parents=True, exist_ok=True)
        with args.records.open("x", encoding="utf-8") as stream:
            for index, sample in enumerate(samples):
                prepared = baseline.prepare_sample(sample, processor, args.media_root)
                record = baseline.make_record(prepared, pass_index=1, forward=forward)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
        cleanup()
        report.update(
            status="completed",
            exit_code=0,
            records=args.records.name,
            records_sha256=sha256_file(args.records),
            model=model,
            wall_seconds=round(time.time() - started, 3),
        )
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({k: report.get(k) for k in ("status", "wall_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
