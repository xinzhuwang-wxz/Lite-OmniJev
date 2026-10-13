"""Offline preflight or real CUDA LoRA smoke; never downloads model weights."""

import argparse
import json
from pathlib import Path

from scripts.assets import write_json
from train.trainer import SmokeConfig, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/train-smoke.json"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training/p0-smoke"))
    parser.add_argument("--report", type=Path, default=Path("reports/train-preflight.json"))
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    try:
        config = SmokeConfig.model_validate_json(args.config.read_text())
        report = run(
            config,
            args.model_dir.resolve(),
            args.asset_manifest,
            args.output_dir,
            execute=args.action == "run",
            data_path=args.data,
            resume_from=args.resume_from,
        )
    except Exception as exc:
        report = {"status": "blocked", "exit_code": 1, "error": f"{type(exc).__name__}: {exc}"}
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
