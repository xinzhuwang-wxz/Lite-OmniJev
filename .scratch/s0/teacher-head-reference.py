"""GPU reference check for the converted Jev-Omni teacher decision head.

This script never loads the raw ``head.pt`` pickle and never executes the
upstream ``jev_omni.py`` module. It anchors the converted safetensors tensors to
the isolated conversion report, then compares the live GPU head output against
an independent implementation of the audited upstream formula.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.assets import write_json
from scripts.s0_prepare import read_manifest
from train.teacher_head import tensor_bytes_sha256
from train.teacher_inference import (
    PINNED_SOURCE_REVISION,
    TeacherForwardBlocked,
    expected_head_sha256_from_report,
    load_teacher_forward_bundle,
    validate_teacher_options,
)

SOURCE_FORMULA_SHA256 = "11d761b0b6cefc8aac29757f43af4b2b02af8f9b6c6dad19834c0b14538180f0"
SELECTED_MANIFEST_INDICES = (0, 60, 80)
BOUNDARY_COUNTS = (2, 3, 4, 20, 256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "peft", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _gpu_environment() -> dict[str, Any]:
    torch = __import__("torch")
    info: dict[str, Any] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        info.update(
            {
                "current_device": int(torch.cuda.current_device()),
                "device_name": torch.cuda.get_device_name(),
                "capability": list(torch.cuda.get_device_capability()),
            }
        )
    nvidia_smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    info["nvidia_smi_exit_code"] = nvidia_smi.returncode
    if nvidia_smi.returncode == 0:
        info["nvidia_smi"] = [
            line.strip() for line in nvidia_smi.stdout.splitlines() if line.strip()
        ]
    return info


def _path_sha256(path: Path | None) -> str | None:
    return _sha256(path.resolve(strict=True)) if path is not None else None


def _options_for_case(case: Any) -> tuple[str, ...]:
    if case.question.type == "choice":
        return validate_teacher_options(case.question.options)
    if case.question.type == "score":
        return validate_teacher_options(
            case.question.score_descriptions or [str(i) for i in range(case.question.score_levels)]
        )
    return validate_teacher_options(["yes", "no"])


def _probability_values(result: Any, options: Sequence[str]) -> list[float]:
    return [float(result.probabilities[option]) for option in options]


def _tensor_sha256(tensor: Any) -> str:
    return tensor_bytes_sha256(tensor.detach().float().cpu().contiguous())


def _assert_conversion_tensor_hashes(
    *,
    head_path: Path,
    conversion_report_path: Path,
    device: str,
) -> dict[str, Any]:
    safetensors = __import__("safetensors.torch", fromlist=["load_file"])
    tensors = safetensors.load_file(str(head_path.resolve(strict=True)), device=device)
    conversion = json.loads(conversion_report_path.resolve(strict=True).read_text())
    expected_tensors = conversion.get("tensors")
    if not isinstance(expected_tensors, dict) or set(tensors) != set(expected_tensors):
        raise ValueError("converted head tensor keys do not match the conversion report")
    checked: dict[str, dict[str, Any]] = {}
    for key in sorted(tensors):
        tensor = tensors[key]
        expected = expected_tensors[key]
        actual_sha256 = tensor_bytes_sha256(tensor.detach().cpu().contiguous())
        if actual_sha256 != expected.get("sha256"):
            raise ValueError(f"converted tensor hash mismatch: {key}")
        checked[key] = {
            "shape": [int(dim) for dim in tensor.shape],
            "dtype": str(tensor.dtype),
            "sha256": actual_sha256,
        }
    return {
        "source_repo": conversion.get("source_repo"),
        "source_revision": conversion.get("source_revision"),
        "source_path": conversion.get("source_path"),
        "source_size": conversion.get("source_size"),
        "source_sha256": conversion.get("source_sha256"),
        "output_sha256": conversion.get("output_sha256"),
        "jev_omni.py_sha256": conversion.get("jev_omni.py_sha256"),
        "decision_config.json_sha256": conversion.get("decision_config.json_sha256"),
        "tensors": checked,
        "raw_pickle_loaded_by_this_script": False,
        "upstream_module_executed_by_this_script": False,
        "load_file_device": device,
    }


def _formula_logits(tensors: Mapping[str, Any], features: Any, counts: Any) -> Any:
    torch = __import__("torch")
    functional = torch.nn.functional
    mu = tensors["mu"].to(device=features.device)
    sd = tensors["sd"].to(device=features.device)
    weight = tensors["linear.weight"].to(device=features.device)
    bias = tensors["linear.bias"].to(device=features.device)
    logits = functional.linear((features.float() - mu) / sd, weight, bias)
    mask = torch.arange(256, device=logits.device)[None] >= counts[:, None]
    return logits.masked_fill(mask, -1e30)


def _autocast_reference_logits(tensors: Mapping[str, Any], features: Any, counts: Any) -> Any:
    torch = __import__("torch")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        return _formula_logits(tensors, features, counts)


def _autocast_reference_probabilities(
    tensors: Mapping[str, Any], features: Any, counts: Any, option_count: int
) -> list[float]:
    torch = __import__("torch")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = _formula_logits(tensors, features, counts)
        probabilities = logits[0, :option_count].softmax(-1)
    return [float(item) for item in probabilities.float().cpu().tolist()]


def _tensor_metrics(actual: Any, expected: Any) -> dict[str, Any]:
    torch = __import__("torch")
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if bool(finite.any()):
        finite_delta = (actual[finite].float() - expected[finite].float()).abs()
        max_finite_abs = float(finite_delta.max().item())
    else:
        max_finite_abs = 0.0
    return {
        "shape": [int(dim) for dim in actual.shape],
        "dtype": str(actual.dtype),
        "exact_equal": bool(torch.equal(actual, expected)),
        "max_finite_abs_delta": max_finite_abs,
        "actual_sha256": _tensor_sha256(actual),
        "reference_sha256": _tensor_sha256(expected),
    }


def _probability_metrics(actual: Sequence[float], expected: Sequence[float]) -> dict[str, Any]:
    deltas = [abs(float(left) - float(right)) for left, right in zip(actual, expected, strict=True)]
    return {
        "actual": [float(item) for item in actual],
        "reference": [float(item) for item in expected],
        "exact_equal": list(actual) == list(expected),
        "max_abs_delta": max(deltas) if deltas else 0.0,
        "sum_actual": float(sum(actual)),
        "sum_reference": float(sum(expected)),
        "all_finite": all(math.isfinite(item) for item in [*actual, *expected]),
    }


def _run_boundary_checks(
    head: Any, tensors: Mapping[str, Any], *, device: str
) -> list[dict[str, Any]]:
    torch = __import__("torch")
    hidden = int(tensors["mu"].shape[1])
    rows = []
    features = torch.linspace(-1.0, 1.0, steps=hidden, device=device, dtype=torch.float32)[None]
    head_tensors = {key: value.to(device=device) for key, value in tensors.items()}
    for count in BOUNDARY_COUNTS:
        counts = torch.tensor([count], device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = head(features, counts)
            reference = _formula_logits(head_tensors, features, counts)
        metrics = _tensor_metrics(actual, reference)
        if not metrics["exact_equal"]:
            raise ValueError(f"boundary head mismatch at count {count}")
        masked_tail = actual[0, count:] if count < 256 else actual[0, 0:0]
        visible = actual[0, :count]
        visible_all_finite = bool(torch.isfinite(visible).all())
        masked_all_sentinel = bool((masked_tail == -1e30).all())
        if not visible_all_finite or not masked_all_sentinel:
            raise ValueError(f"boundary mask invariant failed at count {count}")
        rows.append(
            {
                "count": count,
                "shape": [int(dim) for dim in actual.shape],
                "visible_all_finite": visible_all_finite,
                "masked_all_sentinel": masked_all_sentinel,
                "actual_head_vs_reference": metrics,
            }
        )
    return rows


def _load_reference_tensors(head_path: Path, *, device: str) -> dict[str, Any]:
    safetensors = __import__("safetensors.torch", fromlist=["load_file"])
    return dict(safetensors.load_file(str(head_path.resolve(strict=True)), device=device))


def _install_head_hooks(forward: Any, capture: dict[str, Any]) -> list[Any]:
    def pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
        if len(args) != 2:
            raise RuntimeError("teacher head pre-hook expected features and counts")
        capture["features"] = args[0].detach()
        capture["counts"] = args[1].detach()

    def post_hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
        capture["logits"] = output.detach()

    return [
        forward.head.register_forward_pre_hook(pre_hook),
        forward.head.register_forward_hook(post_hook),
    ]


def _run_forward_cases(args: argparse.Namespace, report: dict[str, Any]) -> None:
    torch = __import__("torch")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    revision = PINNED_SOURCE_REVISION
    head_path = args.head or args.asset_root / "teacher/jev-omni" / revision / "head.safetensors"
    conversion_report = args.head_conversion_report or args.asset_root.parent / (
        "reports/teacher-head-conversion.json"
    )
    protocol_source_dir = (
        args.protocol_source_dir or args.asset_root / "quarantine/teacher" / revision
    )
    model_dir = args.model_dir or args.asset_root / "models/akhilaaa3__Jev-Omni" / revision

    report["provenance"] = {
        "manifest": {
            "path": str(args.model_asset_manifest),
            "sha256": _path_sha256(args.model_asset_manifest),
        },
        "conversion_report": {
            "path": str(conversion_report),
            "sha256": _path_sha256(conversion_report),
        },
        "head_spec": {"path": str(args.head_spec), "sha256": _path_sha256(args.head_spec)},
        "model_config": {
            "path": str(model_dir / "config.json"),
            "sha256": _path_sha256(model_dir / "config.json"),
        },
        "decision_config": {
            "path": str(protocol_source_dir / "decision_config.json"),
            "sha256": _path_sha256(protocol_source_dir / "decision_config.json"),
        },
    }
    expected_head_sha256 = expected_head_sha256_from_report(conversion_report)
    report["head_hash_chain"] = _assert_conversion_tensor_hashes(
        head_path=head_path,
        conversion_report_path=conversion_report,
        device="cpu",
    )
    if report["head_hash_chain"]["jev_omni.py_sha256"] != SOURCE_FORMULA_SHA256:
        raise ValueError("conversion report does not anchor the audited upstream source hash")
    report["source_formula_reference"] = {
        "jev_omni.py_sha256": SOURCE_FORMULA_SHA256,
        "formula": (
            "F.linear((features.float()-mu)/sd, linear.weight, linear.bias) "
            "then mask arange(256)>=counts with -1e30"
        ),
        "source_audit_scope": (
            "formula supplied by main session read-only source audit; this script does not "
            "execute upstream custom code"
        ),
    }
    reference_tensors = _load_reference_tensors(head_path, device="cuda")

    cases = read_manifest(args.manifest)
    selected = [
        (cases[index], _options_for_case(cases[index])) for index in SELECTED_MANIFEST_INDICES
    ]
    forward = load_teacher_forward_bundle(
        protocol_source_dir=protocol_source_dir,
        model_dir=model_dir,
        model_asset_manifest=args.model_asset_manifest,
        head_path=head_path,
        spec_path=args.head_spec,
        conversion_report=conversion_report,
        expected_head_sha256=expected_head_sha256,
    )
    report["model_backbone_loaded"] = True
    report["head_loaded_with"] = "safetensors.torch.load_file"
    report["boundary_checks"] = _run_boundary_checks(forward.head, reference_tensors, device="cuda")
    report["cases"] = []
    capture: dict[str, Any] = {}
    hooks = _install_head_hooks(forward, capture)
    try:
        for case, options in selected:
            media = (
                args.asset_root / "s0-prepare-public-001" / case.prefix.media[0].uri
                if case.prefix.media
                else None
            )
            case_report: dict[str, Any] = {
                "id": case.id,
                "modality": case.category,
                "source_primitive": case.question.type,
                "option_count": len(options),
                "iterations": [],
            }
            first_probabilities: list[float] | None = None
            for iteration in range(2):
                capture.clear()
                result = forward.predict(
                    state=case.prefix.state,
                    question=case.question.instruction,
                    options=options,
                    media=media,
                    modality=case.category,
                    video_frames=16,
                )
                report["model_forward_run"] = True
                features = capture.get("features")
                counts = capture.get("counts")
                actual_logits = capture.get("logits")
                if features is None or counts is None or actual_logits is None:
                    raise RuntimeError("teacher head hooks did not capture features/counts/logits")
                reference_logits = _autocast_reference_logits(reference_tensors, features, counts)
                reference_probabilities = _autocast_reference_probabilities(
                    reference_tensors, features, counts, len(options)
                )
                actual_probabilities = _probability_values(result, options)
                probability = _probability_metrics(actual_probabilities, reference_probabilities)
                logits = _tensor_metrics(actual_logits, reference_logits)
                case_report["iterations"].append(
                    {
                        "iteration": iteration,
                        "prediction_index": int(result.prediction_index),
                        "probabilities": actual_probabilities,
                        "reference_probabilities": reference_probabilities,
                        "probability_metrics": probability,
                        "logit_metrics": logits,
                        "feature_sha256": _tensor_sha256(features),
                        "counts": [int(item) for item in counts.detach().cpu().tolist()],
                    }
                )
                if not logits["exact_equal"] or not probability["exact_equal"]:
                    raise ValueError(f"head reference mismatch for case {case.id}")
                if first_probabilities is None:
                    first_probabilities = actual_probabilities
                elif first_probabilities != actual_probabilities:
                    raise ValueError(f"teacher repeatability mismatch for case {case.id}")
            case_report["exact_repeatability"] = True
            report["cases"].append(case_report)
            write_json(args.report, report)
    finally:
        for hook in hooks:
            hook.remove()
        close = getattr(forward, "close", None)
        if close is not None:
            close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("tests/fixtures/s0-prepare-public-001/manifest.json")
    )
    parser.add_argument("--protocol-source-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--model-asset-manifest",
        type=Path,
        default=Path("configs/project-model-assets.json"),
    )
    parser.add_argument("--head", type=Path)
    parser.add_argument(
        "--head-spec",
        type=Path,
        default=Path("configs/teacher-head-jev-omni-spec.json"),
    )
    parser.add_argument("--head-conversion-report", type=Path)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.is_symlink():
        parser.error("report exists")

    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "exit_code": 1,
        "started_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "Converted Jev-Omni teacher head GPU equivalence against a controlled "
            "formula reference; not a benchmark or quality gate"
        ),
        "method_limits": {
            "raw_pickle_loaded": False,
            "direct_raw_pickle_gpu_reference": False,
            "upstream_module_executed": False,
            "claim": (
                "The original-value side is the isolated conversion report's tensor bytehash "
                "chain plus the audited formula, not a direct GPU load of head.pt."
            ),
        },
        "seed": 0,
        "offline": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "tf32_disabled": True,
        "selected_manifest_indices": list(SELECTED_MANIFEST_INDICES),
        "model_forward_run": False,
        "model_backbone_loaded": False,
        "student_head_used": False,
        "trust_remote_code": False,
        "local_files_only": True,
        "runtime": _runtime_versions(),
        "gpu_environment": {},
        "cost_usd": None,
        "code_version": _git_head(),
        "script_sha256": _sha256(Path(__file__)),
        "manifest_sha256": _sha256(args.manifest),
    }
    write_json(args.report, report)
    try:
        try:
            report["gpu_environment"] = _gpu_environment()
        except Exception:
            report["gpu_environment"] = {"status": "unavailable"}
        _run_forward_cases(args, report)
        report["status"] = "passed"
        report["exit_code"] = 0
    except Exception as error:
        if isinstance(error, TeacherForwardBlocked):
            report.update(error.report)
        report.update(
            {
                "status": "failed",
                "exit_code": 1,
                "error_type": type(error).__name__,
                "stack": [
                    {
                        "file": Path(frame.filename).name,
                        "line": frame.lineno,
                        "function": frame.name,
                    }
                    for frame in traceback.extract_tb(error.__traceback__)
                ],
            }
        )
    report["finished_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    write_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "elapsed_seconds")}))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
