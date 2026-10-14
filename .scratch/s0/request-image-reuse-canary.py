"""Compare all public input summaries under request-local official image reuse."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import traceback
from pathlib import Path

from scripts.s0_prepare import read_manifest
from train.multimodal import prepare_case
from train.request_media_cache import request_image_processor

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument(
    "--manifest", type=Path, default=Path("tests/fixtures/s0-prepare-public-001/manifest.json")
)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
if args.report.exists():
    parser.error("report exists")
started = time.monotonic()
r = {
    "status": "running",
    "exit_code": 1,
    "weights_loaded": False,
    "gpu_forward": False,
    "s0_acceptance": False,
    "cost_usd": None,
    "cases": [],
    "cache_hits": 0,
    "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    "implementation_sha256": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path("train/request_media_cache.py"), Path("train/multimodal.py"))
    },
    "runtime": {
        "python": platform.python_version(),
        "transformers": importlib.metadata.version("transformers"),
        "torch": importlib.metadata.version("torch"),
    },
}


def save():
    r["elapsed_seconds"] = time.monotonic() - started
    args.report.write_text(json.dumps(r, indent=2) + "\n")


save()
try:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    for case in read_manifest(args.manifest):
        wrapped = request_image_processor(processor)
        baseline = prepare_case(case, processor, args.asset_root, profile="canonical_http")
        first = prepare_case(case, wrapped, args.asset_root, profile="canonical_http")
        second = prepare_case(case, wrapped, args.asset_root, profile="canonical_http")
        assert baseline.summary == first.summary == second.summary
        assert baseline.http_messages == first.http_messages == second.http_messages
        assert baseline.answer_token_ids == first.answer_token_ids == second.answer_token_ids
        r["cache_hits"] += wrapped.request_image_cache_stats.hits
        r["cases"].append(
            {
                "id": case.id,
                "category": case.category,
                "exact": True,
                "summary_sha256": hashlib.sha256(
                    json.dumps(baseline.summary, sort_keys=True).encode()
                ).hexdigest(),
                "cache_hits": wrapped.request_image_cache_stats.hits,
            }
        )
        if len(r["cases"]) % 10 == 0:
            save()
            print("completed", len(r["cases"]), flush=True)
    assert len(r["cases"]) == 100 and r["cache_hits"] >= 40
    r.update(status="passed_cpu_input_equivalence_only", exit_code=0)
except Exception as e:
    r.update(
        status="failed",
        exit_code=1,
        error_type=type(e).__name__,
        stack=[
            {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
            for f in traceback.extract_tb(e.__traceback__)
        ],
    )
save()
print(json.dumps({k: r[k] for k in ("status", "exit_code", "elapsed_seconds", "cache_hits")}))
raise SystemExit(r["exit_code"])
