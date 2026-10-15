"""Offline native vLLM processor/HF comparison; never creates an engine or model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import resource
import subprocess
import time
import traceback
import wave
from pathlib import Path
from typing import Any

from scripts.assets import write_json
from scripts.s0_prepare import read_manifest
from train.multimodal import file_hash, native_processor_kwargs, prepare_case


def canonical_mm_data(parts: list[dict[str, Any]], video_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Decode the exact canonical HTTP bytes, as native media IO would do."""
    np = importlib.import_module("numpy")
    data: dict[str, Any] = {}
    for part in parts:
        if part["type"] == "text":
            continue
        if part["type"] == "input_audio":
            raw = base64.b64decode(part["input_audio"]["data"], validate=True)
            with wave.open(io.BytesIO(raw), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                    raise ValueError("invalid_canonical_audio")
                pcm = wav.readframes(wav.getnframes())
            data["audio"] = (np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0, 16000)
            continue
        image_module = importlib.import_module("PIL.Image")
        if part["type"] == "image_url":
            raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1], validate=True)
            with image_module.open(io.BytesIO(raw)) as image:
                data.setdefault("image", []).append(image.convert("RGB"))
        elif part["type"] == "video_url":
            chunks = part["video_url"]["url"].removeprefix("data:video/jpeg;base64,").split(",")
            frames = []
            for chunk in chunks:
                with image_module.open(io.BytesIO(base64.b64decode(chunk, validate=True))) as frame:
                    frames.append(np.asarray(frame.convert("RGB")))
            metadata = {
                key: value for key, value in video_kwargs.items() if key != "do_sample_frames"
            }
            data["video"] = [(np.stack(frames), metadata)]
        else:
            raise ValueError("unsupported_canonical_part")
    return data


def tensor_summary(value: Any) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise ValueError("unsupported_nested_tensor")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def compare_native(prepared: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Compare exact expanded IDs and the documented vLLM field adaptations."""
    np = importlib.import_module("numpy")
    hf_ids = prepared.inputs["input_ids"][0].tolist()
    native_ids = result["prompt_token_ids"]
    native = result["mm_kwargs"].get_data(device="cpu", pin_memory=False)
    checks = {"expanded_prompt_ids": hf_ids == native_ids}
    fields: dict[str, Any] = {}
    translations = {
        "pixel_values": "pixel_values",
        "image_position_ids": "pixel_position_ids",
        "pixel_values_videos": "pixel_values_videos",
        "video_position_ids": "pixel_position_ids_videos",
        "num_frames_per_video": "video_frame_counts",
    }
    for hf_key, native_key in translations.items():
        if hf_key not in prepared.inputs:
            continue
        if native_key not in native:
            checks[hf_key] = False
            continue
        hf_tensor = prepared.inputs[hf_key]
        native_tensor = native[native_key]
        adaptation = None
        if (
            hf_key in {"pixel_values_videos", "video_position_ids"}
            and len(hf_tensor.shape) == 4
            and hf_tensor.shape[0] == 1
            and tuple(hf_tensor.shape[1:]) == tuple(native_tensor.shape)
        ):
            hf_tensor = hf_tensor[0]
            adaptation = "single HF video batch to native flat frames; values unchanged"
        expected, actual = (
            tensor_summary(hf_tensor),
            tensor_summary(native[native_key]),
        )
        fields[hf_key] = {"native_key": native_key, "hf": expected, "native": actual}
        if adaptation:
            fields[hf_key]["adaptation"] = adaptation
        checks[hf_key] = expected == actual
    if "input_features" in prepared.inputs:
        mask = prepared.inputs["input_features_mask"][0].bool()
        for hf_key, native_key in (
            ("input_features", "input_features_padded"),
            ("input_features_mask", "input_features_mask"),
        ):
            expected = tensor_summary(prepared.inputs[hf_key][0][mask].unsqueeze(0))
            actual = tensor_summary(native[native_key])
            fields[hf_key] = {
                "native_key": native_key,
                "normalization": "HF valid frames only, matching native per-item unpadding",
                "hf": expected,
                "native": actual,
            }
            checks[hf_key] = expected == actual
    if prepared.http_media_io_kwargs and "video" in prepared.http_media_io_kwargs:
        meta = prepared.http_media_io_kwargs["video"]
        # TF 5.17 omits its count tensor; the frozen media still defines the count.
        expected_count = tensor_summary(np.asarray([len(meta["frames_indices"])], dtype=np.int64))
        actual_count = (
            tensor_summary(native["video_frame_counts"]) if "video_frame_counts" in native else None
        )
        fields["canonical_video_frame_count"] = {
            "source": "frozen canonical frame indices; no HF tensor fabricated",
            "expected": expected_count,
            "native": actual_count,
        }
        checks["canonical_video_frame_count"] = expected_count == actual_count
        expected = np.asarray([[index / meta["fps"] for index in meta["frames_indices"]]])
        actual = native.get("video_timestamps")
        fields["video_timestamps"] = {
            "hf": tensor_summary(expected),
            "native": tensor_summary(actual),
        }
        checks["video_timestamps"] = bool(np.array_equal(expected, actual))
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fields": fields,
        "hf_prompt_ids": tensor_summary(np.asarray(hf_ids, dtype=np.int64)),
        "native_prompt_ids": tensor_summary(np.asarray(native_ids, dtype=np.int64)),
        "native_placeholder_ranges": {
            modality: [{"offset": item.offset, "length": item.length} for item in ranges]
            for modality, ranges in result["mm_placeholders"].items()
        },
        "native_fields": sorted(native),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--hf-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--same-process-inspection", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("evidence output already exists")
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "running",
        "exit_code": 1,
        "phase": "before_native_import",
        "scope": "Native processor only; no engine/backbone/forward/CUDA or probability acceptance",
        "gpu_used": False,
        "weights_loaded": False,
        "forward_run": False,
        "complete_100_cases": False,
        "cache_mode": "disabled",
        "cases": [],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "cost": {"new_paid_resources": 0},
    }

    def checkpoint(phase: str) -> None:
        report.update(
            phase=phase,
            elapsed_seconds=round(time.monotonic() - started, 3),
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        write_json(args.output, report)

    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("HF_HUB_OFFLINE") != "1":
            raise ValueError("explicit_offline_no_cuda_environment_required")
        cases = read_manifest(args.manifest)
        reference = json.loads(args.hf_reference.read_text())
        if (
            reference.get("input_profile") != "canonical-http-v1"
            or reference.get("status") != "passed"
        ):
            raise ValueError("passing_frozen_canonical_reference_required")
        frozen = {row["id"]: row for row in reference["cases"]}
        report["manifest_sha256"] = file_hash(args.manifest)
        report["hf_reference_sha256"] = file_hash(args.hf_reference)
        if reference["manifest_sha256"] != report["manifest_sha256"]:
            raise ValueError("reference_manifest_mismatch")
        report["source_sha256"] = {
            "scripts/s0_vllm_processor.py": file_hash(Path(__file__)),
            "train/multimodal.py": file_hash(Path(__file__).parent.parent / "train/multimodal.py"),
        }
        version = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        report["code_version"] = (
            os.environ.get("S0_CODE_VERSION") or version or "unversioned_snapshot"
        )
        report["environment"]["packages"] = {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "transformers", "numpy", "pillow")
        }
        checkpoint("importing_native_config")
        config_module = importlib.import_module("vllm.config")
        report["same_process_official_class_inspection"] = args.same_process_inspection
        if args.same_process_inspection:
            checkpoint("importing_official_native_class_in_process")
            model_class = importlib.import_module(
                "vllm.model_executor.models.gemma4_mm"
            ).Gemma4ForConditionalGeneration
            importlib.import_module("vllm.model_executor.models").ModelRegistry.register_model(
                "Gemma4ForConditionalGeneration", model_class
            )
        checkpoint("constructing_native_model_config")
        cfg = config_module.ModelConfig(
            model=str(args.model_dir),
            tokenizer=str(args.model_dir),
            dtype="float32",
            seed=0,
            enforce_eager=True,
            trust_remote_code=False,
            mm_processor_cache_gb=0,
            limit_mm_per_prompt={"image": 1, "audio": 1, "video": 1},
        )
        report["native_platform"] = type(
            importlib.import_module("vllm.platforms").current_platform
        ).__name__
        checkpoint("creating_native_processor")
        native = importlib.import_module("vllm.multimodal").MULTIMODAL_REGISTRY.create_processor(
            cfg
        )
        checkpoint("loading_native_official_hf_processor")
        hf = native.info.get_hf_processor()
        selected = cases if args.full else [cases[index] for index in (0, 20, 40, 60, 80, 90)]
        for case in selected:
            report["active_case_id"] = case.id
            checkpoint("canonical_hf_preparation")
            prepared = prepare_case(case, hf, args.asset_root, profile="canonical_http")
            frozen_match = all(
                prepared.summary[key] == frozen[case.id][key]
                for key in (
                    "tensors",
                    "prompt_sha256",
                    "canonical_media",
                    "http_messages_sha256",
                    "http_media_io_kwargs_sha256",
                    "http_mm_processor_kwargs_sha256",
                    "answer_token_mapping_sha256",
                )
            )
            checkpoint("native_processing")
            assert prepared.http_messages is not None
            mm_data = canonical_mm_data(
                prepared.http_messages[0]["content"],
                (prepared.http_media_io_kwargs or {}).get("video", {}),
            )
            result = native(
                prepared.prompt,
                native.info.parse_mm_data(mm_data),
                hf_processor_mm_kwargs=native_processor_kwargs(
                    prepared.http_media_io_kwargs, prepared.http_image_max_soft_tokens
                ),
            )
            compared = compare_native(prepared, result)
            compared.update(
                id=case.id,
                category=case.category,
                video_frames=case.video_frames,
                frozen_hf_reference_match=frozen_match,
            )
            compared["passed"] = compared["passed"] and frozen_match
            report["cases"].append(compared)
            checkpoint("case_compared")
        report.update(
            status="passed" if all(row["passed"] for row in report["cases"]) else "failed",
            exit_code=0 if all(row["passed"] for row in report["cases"]) else 1,
            complete_100_cases=args.full,
        )
        report.pop("active_case_id", None)
    except Exception as exc:
        report.update(
            status="blocked",
            error_type=type(exc).__name__,
            failure_stage=report["phase"],
            stack=[
                {"file": Path(frame.filename).name, "function": frame.name, "line": frame.lineno}
                for frame in traceback.extract_tb(exc.__traceback__)
            ],
        )
    checkpoint("finished")
    print(json.dumps({key: report[key] for key in ("status", "exit_code", "complete_100_cases")}))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
