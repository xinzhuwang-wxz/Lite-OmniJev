"""Build a redacted sealed-gate summary from an isolated annotation export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sealed_gate import SealedManifest, load_annotations, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = SealedManifest.model_validate_json(args.manifest.read_text(encoding="utf-8"))
    annotations = load_annotations(args.annotations)
    summary = summarize(manifest, annotations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
