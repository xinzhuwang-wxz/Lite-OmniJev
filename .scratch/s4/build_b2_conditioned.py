"""B2 conditioning builder: perception summaries -> conditioned decision rows.

CPU only. Input: caption probe JSONL (validated summaries) plus the source
pool rows. Output: Sample-schema rows whose prefix carries the perception
summary as text; `variant` selects whether the raw video stays attached
(summary+video) or the decision pass sees text only (summary-only serving
shape). Invalid/missing summaries are dropped and counted, never faked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUMMARY_STATE = "Perception summary (model-generated, probe evidence): "


def conditioned_row(row: dict, summary: dict, *, keep_video: bool) -> dict:
    summary_text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    prefix = {
        "state": row["prefix"]["state"] + "\n" + SUMMARY_STATE + summary_text,
        "media": row["prefix"]["media"] if keep_video else [],
    }
    meta = dict(row["meta"])
    meta["data_version"] = (
        row["meta"]["data_version"] + "+b2-summary-video-v1"
        if keep_video
        else row["meta"]["data_version"] + "+b2-summary-only-v1"
    )
    return {**row, "prefix": prefix, "meta": meta}


def build(captions_path: Path, source_path: Path, output: Path, *, keep_video: bool) -> dict:
    summaries = {}
    invalid = 0
    for line in captions_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("summary_valid") and record.get("summary") is not None:
            summaries[record["id"]] = record["summary"]
        else:
            invalid += 1
    emitted = 0
    dropped_missing = 0
    with output.open("x", encoding="utf-8") as stream:
        for line in source_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["meta"]["modality"] != ["video"]:
                continue
            summary = summaries.get(row["id"])
            if summary is None:
                dropped_missing += 1
                continue
            stream.write(json.dumps(conditioned_row(row, summary, keep_video=keep_video), ensure_ascii=False) + "\n")
            emitted += 1
    return {
        "emitted": emitted,
        "captions_valid": len(summaries),
        "captions_invalid": invalid,
        "dropped_missing_summary": dropped_missing,
        "variant": "summary+video" if keep_video else "summary-only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-video", action="store_true", help="summary-only serving shape")
    args = parser.parse_args(argv)
    stats = build(args.captions, args.source, args.output, keep_video=not args.drop_video)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
