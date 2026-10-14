"""Build a redacted sealed-gate summary from an isolated annotation export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from eval.sealed_gate import SealedManifest, load_annotations, load_development, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def write_redacted_failure(out: Path, category: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "passed": False,
                "redaction": "aggregate_counts_no_payload_labels_or_item_ids",
                "failures": {category: 1},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    try:
        manifest_text = args.manifest.read_text(encoding="utf-8")
    except OSError:
        write_redacted_failure(args.out, "manifest_unreadable")
        print("sealed_gate_error=manifest_unreadable")
        return 1
    try:
        manifest = SealedManifest.model_validate_json(manifest_text)
    except (ValidationError, ValueError):
        write_redacted_failure(args.out, "invalid_manifest")
        print("sealed_gate_error=invalid_manifest")
        return 1
    try:
        annotations = load_annotations(args.annotations)
    except (OSError, ValidationError, ValueError):
        write_redacted_failure(args.out, "invalid_annotations")
        print("sealed_gate_error=invalid_annotations")
        return 1
    development = None
    if args.development is not None:
        try:
            development = load_development(args.development)
        except (OSError, ValidationError, ValueError):
            write_redacted_failure(args.out, "invalid_development")
            print("sealed_gate_error=invalid_development")
            return 1
    summary = summarize(manifest, annotations, development)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
