"""Convert MMAU test-mini and MVBench public rows into the Sample contract.

Imports the frozen loaders from the S0 teacher benchmark path (no copying):
row selection, manifest-hash assertions, option order, and gold parsing are
the exact code the teacher runs used, so any drift fails loudly. Output rows
load through scripts.s1_dev_eval.load_samples with a media root that carries
pre-staged media files named by sha256 (see --stage-media).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval.teacher_benchmarks import (  # noqa: E402
    MMAU_EXPECTED_ROWS,
    MVBench_EXPECTED_ROWS,
    file_sha256,
    load_mmau_examples,
    load_mvbench_examples,
)

# Metadata is extra-forbidden in the contract, so license provenance notes
# travel in the converter report, not in Sample.meta.
MMAU_LICENSE = {"license": "CC-BY-NC-4.0", "usage": "research"}
MVBENCH_LICENSE = {"license": "MIT+restricted-media", "usage": "research"}
LICENSE_NOTES = {
    "mmau-test-mini": "CC-BY-NC-4.0 covers dataset and embedded audio; repo Apache-2.0 is not the media license",
    "mvbench": "task metadata MIT; media academic-research-only, underlying source agreements apply",
}


def _config(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_examples(args: argparse.Namespace, config: dict):
    """Same selection + hash assertion as scripts/s0_teacher_eval._load_examples."""
    from scripts import s0_teacher_eval

    expected_mmau = config["mmau"]["file"]
    parquet = args.mmau_parquet
    if parquet is None:
        parquet = s0_teacher_eval._default_mmau_parquet(config)
    if parquet.stat().st_size != expected_mmau.get("bytes") or file_sha256(parquet) != expected_mmau.get(
        "sha256"
    ):
        raise ValueError("mmau_source_hash_mismatch")
    mmau = load_mmau_examples(parquet, expected_mmau)
    if len(mmau) != MMAU_EXPECTED_ROWS:
        raise ValueError("mmau_row_count_mismatch")
    mvbench = load_mvbench_examples(
        frame_manifest=args.mvbench_frame_manifest,
        metadata_dir=args.mvbench_metadata_dir,
        config_path=args.config,
    )
    if len(mvbench) != MVBENCH_EXPECTED_ROWS:
        raise ValueError("mvbench_row_count_mismatch")
    return mmau, mvbench


def _sample_rows(examples, *, benchmark: str, media_kind: str, license_info: dict):
    rows = []
    for example in examples:
        letters = [chr(65 + i) for i in range(len(example.options))]
        media = None
        if media_kind == "audio":
            media = {
                "type": "audio",
                "uri": f"media/{example.media_bytes_sha256}.wav",
                "sha256": example.media_bytes_sha256,
            }
        else:
            first = example.video_frames[0]
            media = {
                "type": "video",
                "uri": f"media/{example.media_bytes_sha256 or first.png_sha256}.frames.json",
                "sha256": example.media_bytes_sha256 or first.png_sha256,
            }
        rows.append(
            {
                "id": f"{benchmark}:{example.row_id}",
                "prefix": {
                    "state": (
                        "Answer the multiple-choice question using the attached audio."
                        if media_kind == "audio"
                        else "Answer the multiple-choice question using the attached video."
                    ),
                    "media": [media],
                },
                "questions": [
                    {
                        "qid": "choice",
                        "type": "choice",
                        "instruction": example.question,
                        "options": list(example.options),
                        "score_levels": None,
                        "score_descriptions": None,
                        "target": {
                            "kind": "hard",
                            "value": letters[example.label_index],
                            "source": f"public_bench:{benchmark}",
                        },
                        "pair_of": None,
                    }
                ],
                "meta": {
                    "modality": [media_kind],
                    "source_dataset": benchmark,
                    "source_revision": example.source_sha256[:64],
                    "source_sha256": example.source_sha256,
                    **license_info,
                    "pool": "dev",
                    "family_id": f"{benchmark}:{example.group}",
                    "data_version": f"{benchmark}-public-eval-v1",
                },
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/video-eval-assets.json")
    parser.add_argument("--mmau-parquet", type=Path)
    parser.add_argument("--mvbench-frame-manifest", type=Path, required=True)
    parser.add_argument("--mvbench-metadata-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage-media",
        action="store_true",
        help="write audio bytes / frame manifests into <output-dir>/media so "
        "s1_dev_eval can resolve them by sha256",
    )
    args = parser.parse_args(argv)
    config = _config(args.config)
    mmau, mvbench = _load_examples(args, config)
    mmau_rows = _sample_rows(mmau, benchmark="mmau-test-mini", media_kind="audio", license_info=MMAU_LICENSE)
    mvbench_rows = _sample_rows(mvbench, benchmark="mvbench", media_kind="video", license_info=MVBENCH_LICENSE)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"mmau_rows": len(mmau_rows), "mvbench_rows": len(mvbench_rows)}
    for name, rows in (("mmau-test-mini.jsonl", mmau_rows), ("mvbench.jsonl", mvbench_rows)):
        path = args.output_dir / name
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        report[name] = {"rows": len(rows), "sha256": file_sha256(path)}
    if args.stage_media:
        media_dir = args.output_dir / "media"
        media_dir.mkdir(exist_ok=True)
        audio_written = frames_written = 0
        for example in mmau:
            target = media_dir / f"{example.media_bytes_sha256}.wav"
            if not target.exists():
                target.write_bytes(example.media_bytes)
                audio_written += 1
        for example in mvbench:
            digest = example.media_bytes_sha256 or example.video_frames[0].png_sha256
            target = media_dir / f"{digest}.frames.json"
            if not target.exists():
                payload = {
                    "benchmark": "mvbench",
                    "row_id": example.row_id,
                    "frames": [
                        {"index": f.index, "png": f.png, "png_sha256": f.png_sha256}
                        for f in example.video_frames
                    ],
                }
                target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
                frames_written += 1
        report["media"] = {"audio_written": audio_written, "frame_manifests_written": frames_written}
    report["license_notes"] = LICENSE_NOTES
    report["prompt_parity_note"] = (
        "Teacher path builds its own prompt via build_benchmark_processor_inputs; "
        "this converter emits the standard decision-contract prefix state plus the "
        "raw public question text and options in original order. s1_dev_eval renders "
        "the official chat template with letters A..; no teacher-specific framing, "
        "few-shot, or task tags are added. Differences from the teacher run are "
        "therefore: prefix wording, letter mapping, and the absence of the "
        "teacher's processor-level prompt assembly."
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
