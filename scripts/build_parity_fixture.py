"""Build independent program-truth calibration and confirmation parity fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.parity_fixture import build_fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data/parity-fixture-v1"),
        help="directory to receive calibration.jsonl, confirmation.jsonl and manifest.json",
    )
    args = parser.parse_args(argv)
    manifest = build_fixture(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
