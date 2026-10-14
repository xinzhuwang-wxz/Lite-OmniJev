"""Bounded real teacher smoke on public Choice/Score/Noul media, never quality acceptance."""

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from scripts.s0_prepare import read_manifest
from train.teacher_inference import load_teacher_forward_bundle, validate_teacher_options

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
if args.report.exists():
    parser.error("report exists")
started = time.monotonic()
report = {
    "status": "running",
    "exit_code": 1,
    "model_forward_run": False,
    "scope": "Public teacher smoke only; no benchmark or stage qualification",
    "cost": None,
    "cases": [],
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "code_version": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
}


def save():
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


save()
try:
    import importlib.metadata

    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report["seed"] = 0
    report["runtime"] = {
        name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
    }
    cases = read_manifest(Path("tests/fixtures/s0-prepare-public-001/manifest.json"))
    selected = []
    for index in (0, 60, 80):
        case = cases[index]
        if case.question.type == "choice":
            options = case.question.options
        elif case.question.type == "score":
            options = case.question.score_descriptions or [
                str(i) for i in range(case.question.score_levels)
            ]
        else:
            options = ["yes", "no"]
        selected.append((case, validate_teacher_options(options)))
    revision = "5addda86ddee081a68fb067477ea100c221b8917"
    forward = load_teacher_forward_bundle(
        protocol_source_dir=args.asset_root / "quarantine/teacher" / revision,
        model_dir=args.asset_root / "models/akhilaaa3__Jev-Omni" / revision,
        model_asset_manifest=Path("configs/project-model-assets.json"),
        head_path=args.asset_root / "teacher/jev-omni" / revision / "head.safetensors",
        spec_path=Path("configs/teacher-head-jev-omni-spec.json"),
        conversion_report=args.asset_root.parent / "reports/teacher-head-conversion.json",
    )
    report["model_backbone_loaded"] = True
    save()
    for case, options in selected:
        media = (
            args.asset_root / "s0-prepare-public-001" / case.prefix.media[0].uri
            if case.prefix.media
            else None
        )
        outputs = []
        for _ in range(2):
            result = forward.predict(
                state=case.prefix.state,
                question=case.question.instruction,
                options=options,
                media=media,
                modality=case.category,
                video_frames=16,
            )
            report["model_forward_run"] = True
            values = list(result.probabilities.values())
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            assert abs(sum(values) - 1) < 1e-6
            outputs.append({"probabilities": values, "prediction_index": result.prediction_index})
        report["cases"].append(
            {
                "id": case.id,
                "modality": case.category,
                "source_primitive": case.question.type,
                "option_count": len(options),
                "outputs": outputs,
                "exact_repeatability": outputs[0] == outputs[1],
            }
        )
        save()
        print(case.category, "teacher complete", flush=True)
    report["status"] = (
        "passed" if all(c["exact_repeatability"] for c in report["cases"]) else "failed"
    )
    report["exit_code"] = 0 if report["status"] == "passed" else 1
except Exception as error:
    import traceback

    report.update(
        status="failed",
        exit_code=1,
        error_type=type(error).__name__,
        stack=[
            {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)
        ],
    )
save()
print(
    json.dumps(
        {k: report[k] for k in ("status", "exit_code", "model_forward_run", "elapsed_seconds")}
    )
)
raise SystemExit(report["exit_code"])
