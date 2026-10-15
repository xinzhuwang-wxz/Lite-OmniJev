"""CPU preparation of five frozen S3 screens; never starts GPU inference.

--root is the runtime artifact checkout; --evidence-root contains frozen reports.
Writes SHA-bound manifest, canonical baseline matches and existing evaluator argv.
Crossmodal sources remain explicit gaps; five-cohort preparation cannot qualify S4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample, answer_tokens
from train.readout import read_probabilities

S3_SHA = "1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755"
COHORTS = ("image", "screenshot", "text", "audio", "video")


def require(value: bool, category: str) -> None:
    if not value:
        raise ValueError(category)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, str]:
    require(path.is_absolute(), "absolute_binding")
    return {"path": str(path), "sha256": sha(path)}


def load(spec: dict[str, str]) -> bytes:
    path = Path(spec["path"])
    require(path.is_absolute() and sha(path) == spec["sha256"], "artifact_sha")
    return path.read_bytes()


def arg(command: list[str], flag: str) -> str:
    require(command.count(flag) == 1, "command_flag")
    return command[command.index(flag) + 1]


def write(path: Path, value: Any) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def five_sources(evidence_root: Path) -> list[dict[str, Any]]:
    probe_path = evidence_root / "reports/s3-quality-probe-summary-001.json"
    video_path = evidence_root / "reports/s3-video150-reference-001.json"
    probe, video = json.loads(probe_path.read_bytes()), json.loads(video_path.read_bytes())
    result = []
    for cohort in COHORTS:
        report_path = evidence_root / f"reports/s3-stable-{cohort}-screen-001.json"
        report = json.loads(report_path.read_bytes())
        require(report["status"] == "passed" and report["exit_code"] == 0, "baseline_status")
        require(report["adapter_sha256"] == S3_SHA, "baseline_not_s3_stable")
        command = report["command"]
        require("scripts.s1_dev_eval" in command, "baseline_evaluator")
        frozen = (
            {"data": video["data_path"], "sha256": video["data_sha256"]}
            if cohort == "video"
            else probe["sources"][cohort]
        )
        require(
            arg(command, "--data") == frozen["data"] and report["data_sha256"] == frozen["sha256"],
            "frozen_selection",
        )
        require(report["samples"] == (150 if cohort == "video" else 50), "screen_count")
        require(report["video_num_frames"] == 16 and report["passes"] >= 1, "screen_budget")
        require("--counterfactual" not in command, "unexpected_baseline_variants")
        image_budget = (
            int(arg(command, "--image-max-soft-tokens"))
            if "--image-max-soft-tokens" in command
            else None
        )
        require(cohort not in {"image", "screenshot"} or image_budget == 1120, "image_budget")
        result.append(
            {
                "cohort": cohort,
                "data": {"path": frozen["data"], "sha256": frozen["sha256"]},
                "baseline_report": binding(report_path),
                "baseline_sidecar": {
                    "path": str(Path(arg(command, "--output-dir")) / "records.jsonl"),
                    "sha256": report["sidecar_sha256"],
                },
                "selection_report": binding(video_path if cohort == "video" else probe_path),
                "model_dir": arg(command, "--model-dir"),
                "media_root": arg(command, "--media-root") if "--media-root" in command else None,
                "image_max_soft_tokens": image_budget,
                "video_frames": 16,
                "passes": 1,
                "maximum_correct_drop": 3 if cohort == "video" else 1,
            }
        )
    return result


def canonical_baseline(source: dict[str, Any]) -> dict[str, Any]:
    report = json.loads(load(source["baseline_report"]))
    require(report["adapter_sha256"] == S3_SHA, "baseline_adapter")
    require(
        report["sidecar_sha256"] == source["baseline_sidecar"]["sha256"], "baseline_sidecar_binding"
    )
    samples = [
        Sample.model_validate_json(line)
        for line in load(source["data"]).splitlines()
        if line.strip()
    ]
    require(len(samples) == (150 if source["cohort"] == "video" else 50), "data_count")
    require(len({s.id for s in samples}) == len(samples), "duplicate_data_id")
    expected = "image" if source["cohort"] == "screenshot" else source["cohort"]
    require(
        all(
            s.meta.pool == "dev" and s.meta.modality == [expected] and len(s.questions) == 1
            for s in samples
        ),
        "public_dev_only",
    )
    ids = {s.id for s in samples}
    matches = {}
    for line in load(source["baseline_sidecar"]).splitlines():
        row = json.loads(line)
        if row["pass"] == 0 and row["variant"] == "original" and row["id"] in ids:
            require(row["id"] not in matches, "duplicate_baseline_id")
            matches[row["id"]] = row
    require(set(matches) == ids, "baseline_missing_ids")
    matched, corrected = [], 0
    for sample in samples:
        row, question = matches[sample.id], sample.questions[0]
        if not isinstance(question.target, HardTarget):
            raise ValueError("hard_target_required")
        require(row["target"] == question.target.value, "baseline_target_mismatch")
        tokens = list(answer_tokens(question))
        require(set(row["raw_logits"]) == set(tokens) | {"U"}, "answer_token_mapping")
        require(all(math.isfinite(v) for v in row["raw_logits"].values()), "nonfinite_logits")
        # Canonical answer_tokens preserves yes/no order despite sorted JSON object keys.
        readout = read_probabilities(row["raw_logits"], tokens, temperature=1.0)
        corrected += int(row["prediction"] != readout.prediction)
        matched.append(
            {
                "id": sample.id,
                "target": question.target.value,
                "prediction": readout.prediction,
                "answer_order": tokens,
            }
        )
    correct = sum(row["prediction"] == row["target"] for row in matched)
    return {
        "cohort": source["cohort"],
        "count": len(samples),
        "baseline_correct": correct,
        "minimum_candidate_correct": correct - source["maximum_correct_drop"],
        "canonical_prediction_corrections": corrected,
        "records": matched,
    }


def prepare(root: Path, evidence_root: Path, output: Path, python: str) -> dict[str, Any]:
    require(all(p.is_absolute() for p in (root, evidence_root, output)), "absolute_paths")
    require(not output.exists(), "output_exists")
    train_path = root / "reports/s4-joint-pilot-train-001.json"
    train = json.loads(train_path.read_bytes())
    require(
        train["status"] == "completed"
        and train["exit_code"] == 0
        and train["steps_completed"] == 30,
        "candidate_not_completed_30",
    )
    require(
        train["config"]["steps"] == 30 and train["config"]["accumulation"] == 10, "candidate_config"
    )
    require(train["init_adapter"]["weights_sha256"] == S3_SHA, "candidate_initial_adapter")
    adapter = root / "artifacts/s4-joint-pilot-train-001/adapter"
    require(
        sha(adapter / "adapter_model.safetensors") == train["adapter_sha256"],
        "candidate_adapter_sha",
    )
    sources = five_sources(evidence_root)
    baseline = [canonical_baseline(source) for source in sources]
    output.mkdir(mode=0o700, parents=True)
    commands = []
    for source in sources:
        cohort = source["cohort"]
        require(Path(source["model_dir"]).name == train["model_revision"], "base_revision_mismatch")
        argv = [
            python,
            "-u",
            "-m",
            "scripts.s1_dev_eval",
            "--model-dir",
            source["model_dir"],
            "--adapter",
            str(adapter),
            "--data",
            source["data"]["path"],
            "--passes",
            "1",
            "--video-frames",
            "16",
            "--output-dir",
            str(output / (cohort + "-eval")),
            "--report",
            str(output / (cohort + "-report.json")),
        ]
        if source["media_root"]:
            argv += ["--media-root", source["media_root"]]
        if source["image_max_soft_tokens"] is not None:
            argv += ["--image-max-soft-tokens", str(source["image_max_soft_tokens"])]
        commands.append(
            {
                "source": cohort,
                "argv": argv,
                "foreground_command": shlex.join(argv),
                "timeout_seconds": 300,
            }
        )
    gaps = [
        {
            "source": "crossmodal_original50",
            "status": "blocked",
            "required": (
                "Explicit SHA-bound selected50 JSONL, S3-stable baseline report+sidecar, "
                "canonical answer order, media roots and same processor budget; raw "
                "development500 is not the selected50 binding."
            ),
        },
        {
            "source": "crossmodal_supplemental50",
            "status": "blocked",
            "required": (
                "Explicit SHA-bound selected50 JSONL, S3-stable matched baseline "
                "report+sidecar and evaluator-format/processor equivalence; do not infer "
                "paths or seeds."
            ),
        },
    ]
    manifest = {
        "schema_version": "s4-joint-pilot-five-screen-v1",
        "candidate_training_report": binding(train_path),
        "candidate_adapter": binding(adapter / "adapter_model.safetensors"),
        "candidate_adapter_config": binding(adapter / "adapter_config.json"),
        "sources": sources,
        "crossmodal_gaps": gaps,
    }
    write(output / "manifest.json", manifest)
    write(output / "baseline-matches.json", baseline)
    write(output / "commands.json", commands)
    result = {
        "status": "five_cohort_prepared_crossmodal_blocked",
        "gpu_started": False,
        "pilot_screen_pass": None,
        "scope": (
            "Five old cohort CPU preparation only; sources stay separate; no "
            "quality/stage admission"
        ),
        "manifest_sha256": sha(output / "manifest.json"),
        "baseline_matches_sha256": sha(output / "baseline-matches.json"),
        "commands_sha256": sha(output / "commands.json"),
        "sources": [{k: v for k, v in row.items() if k != "records"} for row in baseline],
        "crossmodal_gaps": gaps,
        "next": (
            "Run each emitted existing evaluator command under an observed300s "
            "foreground supervisor; verify candidate report+sidecar SHA and same IDs "
            "before comparing each source. Require both crossmodal sources before pilot "
            "decision."
        ),
    }
    write(output / "report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    report = prepare(args.root, args.evidence_root, args.output_dir, args.python)
    print(json.dumps({"status": report["status"], "gpu_started": False, "pilot_screen_pass": None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
