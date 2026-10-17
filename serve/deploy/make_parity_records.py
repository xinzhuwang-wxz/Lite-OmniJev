"""Operator record builder for S0 parity/gateway harnesses (identity attestation helper).

Modes:
  vllm    — server record for `scripts.s0_parity compare-vllm`; provenance must
            byte-match the run's build_provenance output, so the same path/dtype
            arguments as the compare invocation are required here.
  gateway — gateway operator record for `scripts.s0_gateway_parity compare-gateway`;
            provenance is copied verbatim from the passed HF/vLLM report.

The operator is responsible for verifying the live process (cmdline, weights,
logprobs mode) before passing --operator-attested. This tool writes records; it
cannot attest anything by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.s0_parity import ALLOWED_DTYPES, ParityPaths, build_provenance  # noqa: E402
from train.multimodal import IMAGE_MAX_SOFT_TOKENS  # noqa: E402


def existing_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("vllm", "gateway"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-attested", action="store_true")
    parser.add_argument(
        "--launch-command", help="Exact vLLM (mode=vllm) or gateway launch command line"
    )
    parser.add_argument("--served-model", help="vLLM --served-model-name (mode=vllm)")
    parser.add_argument("--model-version", help="Gateway LITE_MODEL_VERSION (mode=gateway)")
    parser.add_argument("--hf-vllm-report", type=Path, help="compare-vllm report (mode=gateway)")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--dtype", choices=sorted(ALLOWED_DTYPES))
    parser.add_argument("--image-max-soft-tokens", type=int, choices=IMAGE_MAX_SOFT_TOKENS)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"refusing to overwrite existing record: {args.output}")
    if not args.operator_attested:
        parser.error(
            "operator attestation required: --operator-attested after verifying the live process"
        )

    if args.mode == "vllm":
        missing = [
            name
            for name in (
                "manifest",
                "asset_root",
                "model_dir",
                "asset_manifest",
                "dtype",
                "launch_command",
                "served_model",
            )
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"mode=vllm requires: {', '.join(missing)}")
        paths = ParityPaths(
            manifest=args.manifest,
            asset_root=args.asset_root,
            model_dir=args.model_dir,
            asset_manifest=args.asset_manifest,
            adapter_dir=args.adapter_dir,
        )
        record = {
            "operator_attested": True,
            "raw_logprobs_confirmed": True,
            "logprobs_mode": "raw_logprobs",
            "launch_command": args.launch_command,
            "served_model": args.served_model,
            "provenance": build_provenance(
                paths, dtype=args.dtype, image_max_soft_tokens=args.image_max_soft_tokens
            ),
        }
    else:
        missing = [
            name
            for name in ("model_version", "launch_command", "hf_vllm_report")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"mode=gateway requires: {', '.join(missing)}")
        report = existing_report(args.hf_vllm_report)
        if report.get("action") != "compare-vllm" or report.get("status") != "passed":
            parser.error("gateway record requires a passed compare-vllm report")
        record = {
            "operator_attested": True,
            "model_version": args.model_version,
            "gateway_launch_command": args.launch_command,
            "s0_hf_vllm_report_sha256": hashlib.sha256(
                args.hf_vllm_report.read_bytes()
            ).hexdigest(),
            "upstream_server_record_sha256": report["server"]["record_sha256"],
            "provenance": report["provenance"],
        }
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "mode": args.mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
