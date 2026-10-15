"""Assemble S3 same-adapter five-modality metrics and the 8/16-frame control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts.assets import write_json
from scripts.s2_gate import GateInputError, image_budget, load_dev_report, load_report

# Matched S0 Jev-Omni dev500 baselines: reports/s0-baseline-summary.md.
AUDIO_TEACHER_ACCURACY = 0.38
VIDEO_TEACHER_ACCURACY = 0.698


def accuracy(path: Path, label: str) -> tuple[dict[str, Any], float]:
    report = load_dev_report(path, label)
    value = report.get("metrics", {}).get("product_accuracy")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise GateInputError(f"{label} needs a finite accuracy in [0, 1]: {path}")
    return report, float(value)


def assemble_metrics(
    current: dict[str, Path], previous: dict[str, Path], video8: Path
) -> dict[str, float]:
    if set(current) != {"audio", "video", "image", "screenshot", "text"} or set(previous) != {
        "image",
        "screenshot",
        "text",
    }:
        raise GateInputError("five current modalities and three S2 references are required")
    measured = {name: accuracy(path, name) for name, path in current.items()}
    reference = {name: accuracy(path, f"S2 {name}") for name, path in previous.items()}
    identities = {
        (report.get("model_revision"), report.get("adapter_sha256"))
        for report, _ in measured.values()
    }
    if len(identities) != 1 or any(not all(identity) for identity in identities):
        raise GateInputError("all five S3 reports must identify the same model and adapter")
    old_identities = {
        (report.get("model_revision"), report.get("adapter_sha256"))
        for report, _ in reference.values()
    }
    if len(old_identities) != 1 or any(not all(identity) for identity in old_identities):
        raise GateInputError("S2 reference reports must identify one model and adapter")
    if next(iter(old_identities))[0] != next(iter(identities))[0]:
        raise GateInputError("S2 and S3 must use the same base model revision")
    for name, (report, _) in reference.items():
        if name in {"image", "screenshot"} and image_budget(report) != image_budget(
            measured[name][0]
        ):
            raise GateInputError(f"S2/S3 {name} image budgets must match")
        if report.get("data_sha256") != measured[name][0].get("data_sha256") or not report.get(
            "data_sha256"
        ):
            raise GateInputError(f"S2/S3 {name} must use the same frozen dev data")
    control = load_report(video8, "8-frame video control")
    video = measured["video"][0]
    if (
        video.get("video_num_frames") != 16
        or control.get("video_num_frames") != 8
        or control.get("samples") != 500
        or control.get("passes") != 1
        or not video.get("data_sha256")
        or control.get("data_sha256") != video.get("data_sha256")
        or (control.get("model_revision"), control.get("adapter_sha256")) not in identities
    ):
        raise GateInputError("8/16-frame control must use the same 500 samples and model/adapter")
    value = control.get("metrics", {}).get("product_accuracy")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise GateInputError("8-frame control accuracy is missing or invalid")
    return {
        "audio_teacher_ratio": measured["audio"][1] / AUDIO_TEACHER_ACCURACY,
        "video_teacher_ratio": measured["video"][1] / VIDEO_TEACHER_ACCURACY,
        "old_modalities_max_regression": max(
            reference[name][1] - measured[name][1] for name in reference
        ),
        "frame_ab_complete": 1,
        **{f"{name}_accuracy": result[1] for name, result in measured.items()},
        "video_8frame_accuracy": float(value),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio", "video", "image", "screenshot", "text"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("image", "screenshot", "text"):
        parser.add_argument(f"--s2-{name}", type=Path, required=True)
    parser.add_argument("--video8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        metrics = assemble_metrics(
            {
                name: getattr(args, name)
                for name in ("audio", "video", "image", "screenshot", "text")
            },
            {name: getattr(args, f"s2_{name}") for name in ("image", "screenshot", "text")},
            args.video8,
        )
    except GateInputError as exc:
        print(f"invalid S3 gate input: {exc}", file=sys.stderr)
        return 1
    write_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
