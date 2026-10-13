"""CPU-only Jev-Omni teacher source/head/processor preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.assets import write_json
from train.teacher_inference import (
    DEFAULT_HEAD_CONVERSION_REPORT,
    DEFAULT_HEAD_SPEC,
    DEFAULT_SAFE_HEAD,
    DEFAULT_SOURCE_DIR,
    run_teacher_preflight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--head", type=Path, default=DEFAULT_SAFE_HEAD)
    parser.add_argument("--head-spec", type=Path, default=DEFAULT_HEAD_SPEC)
    parser.add_argument(
        "--head-conversion-report", type=Path, default=DEFAULT_HEAD_CONVERSION_REPORT
    )
    parser.add_argument("--expected-head-sha256")
    parser.add_argument("--synthetic-modality", choices=["text", "image", "audio"], default="audio")
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_teacher_preflight(
        source_dir=args.source_dir,
        head_path=args.head,
        spec_path=args.head_spec,
        conversion_report=args.head_conversion_report,
        synthetic_modality=args.synthetic_modality,
        expected_head_sha256=args.expected_head_sha256,
        ffmpeg_binary=args.ffmpeg_binary,
    )
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
