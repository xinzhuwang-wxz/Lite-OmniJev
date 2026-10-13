"""CUDA-gated Jev-Omni teacher forward readiness entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.assets import write_json
from train.teacher_inference import (
    DEFAULT_HEAD_CONVERSION_REPORT,
    DEFAULT_HEAD_SPEC,
    DEFAULT_MODEL_ASSET_MANIFEST,
    DEFAULT_MODEL_DIR,
    DEFAULT_SAFE_HEAD,
    DEFAULT_SOURCE_DIR,
    teacher_forward_readiness_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-source-dir",
        "--source-dir",
        dest="protocol_source_dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-asset-manifest", type=Path, default=DEFAULT_MODEL_ASSET_MANIFEST)
    parser.add_argument("--head", type=Path, default=DEFAULT_SAFE_HEAD)
    parser.add_argument("--head-spec", type=Path, default=DEFAULT_HEAD_SPEC)
    parser.add_argument(
        "--head-conversion-report", type=Path, default=DEFAULT_HEAD_CONVERSION_REPORT
    )
    parser.add_argument("--expected-head-sha256")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = teacher_forward_readiness_report(
        protocol_source_dir=args.protocol_source_dir,
        model_dir=args.model_dir,
        model_asset_manifest=args.model_asset_manifest,
        head_path=args.head,
        spec_path=args.head_spec,
        conversion_report=args.head_conversion_report,
        expected_head_sha256=args.expected_head_sha256,
    )
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
