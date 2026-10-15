"""Bounded public-dev native routing probes; separate CPU/HF/vLLM/compare phases.

No server startup, merged weights, synthetic cases, or stage admission. HF must
finish and release its CUDA allocation before an operator starts vLLM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.schema import Sample  # noqa: E402
from scripts import s0_parity as parity  # noqa: E402
from train.multimodal import PreparationCase, prepare_case  # noqa: E402

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
MODEL = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
ASSETS = Path("/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json")
ADAPTERS = {
    "general": (
        ROOT / "artifacts/s4-joint-pilot-train-001/adapter",
        "6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac",
    ),
    "video": (
        ROOT / "artifacts/s4-video-pilot-train-001/adapter",
        "14496e283908180303e6d48d0983dbd4c574b4f35f7bb350acf59d9c2a176c9c",
    ),
}
COHORTS = ("image", "screenshot", "audio", "video")


def sha(path: Path) -> str:
    return parity.sha256_file(path)


def bound(item: dict[str, str]) -> Path:
    path = Path(item["path"])
    if not path.is_absolute() or path.is_symlink() or sha(path) != item["sha256"]:
        raise ValueError("input path/SHA binding failed")
    return path


def selection(
    manifest_path: Path, per_cohort: int
) -> list[tuple[str, Path, list[PreparationCase]]]:
    manifest = json.loads(manifest_path.read_text())
    selected = []
    for cohort in COHORTS:
        source = next(item for item in manifest["sources"] if item["cohort"] == cohort)
        path = bound(source["data"])
        cases = []
        # Fixed prefix of already-public dev samples; no calibration/holdout files.
        with path.open() as stream:
            for _ in range(per_cohort):
                sample = Sample.model_validate_json(next(stream))
                if sample.meta.pool != "dev" or len(sample.questions) != 1:
                    raise ValueError("exactly one question and pool=dev required")
                cases.append(
                    PreparationCase(
                        id=sample.id,
                        category=cohort,
                        video_frames=16,
                        prefix=sample.prefix,
                        question=sample.questions[0],
                        meta=sample.meta,
                    )
                )
        selected.append((cohort, Path(source["media_root"]), cases))
    return selected


def inputs(processor: Any, selected: list) -> list[dict[str, Any]]:
    result = []
    for cohort, root, cases in selected:
        for case in cases:
            prepared = prepare_case(
                case, processor, root, profile="canonical_http", image_max_soft_tokens=1120
            )
            result.append(
                {
                    "cohort": cohort,
                    "id_sha256": parity.row_id_hash(case.id),
                    "input_sha256": parity.prepared_input_hash(prepared),
                    "media_sha256": parity.prepared_media_hash(prepared),
                    "prompt_token_ids": prepared.inputs["input_ids"][0].tolist(),
                    "answer_token_ids": prepared.answer_token_ids,
                    "summary": prepared.summary,
                }
            )
    return result


def identity(manifest: Path) -> dict:
    adapters = {}
    for name, (path, expected) in ADAPTERS.items():
        evidence = parity.adapter_evidence(path)
        if evidence is None or evidence.safetensors_sha256 != expected:
            raise ValueError(f"{name} adapter SHA mismatch")
        adapters[name] = evidence.__dict__
    return {
        "manifest_sha256": sha(manifest),
        "adapters": adapters,
        "model_dir": str(MODEL),
        "asset_manifest_sha256": sha(ASSETS),
        "dtype": "bfloat16",
        "image_max_soft_tokens": 1120,
        "video_frames": 16,
        "thinking": False,
        "script_sha256": sha(Path(__file__)),
        "reused_code_sha256": parity.code_hashes(),
    }


def same_inputs(actual: list, prepared: dict) -> None:
    if actual != prepared["inputs"]:
        raise ValueError("canonical input binding changed from prepare phase")


def compare_small(hf: list, serving: list) -> dict:
    left, right = parity.rows_from_json(hf), parity.rows_from_json(serving)
    if (
        len(left) not in {8, 100}
        or len(right) != len(left)
        or len({r.id_sha256 for r in left}) != len(left)
    ):
        raise ValueError("exactly 8 or 100 unique probe rows required")
    maxima = dict.fromkeys(("raw_probabilities", "joint", "conditional", "p_U"), 0.0)
    for expected, actual in zip(left, right, strict=True):
        for key in ("id_sha256", "input_sha256", "media_sha256", "token_ids"):
            if getattr(expected, key) != getattr(actual, key):
                raise ValueError("actual inference row identity mismatch")
        for key in ("raw_probabilities", "joint", "conditional"):
            maxima[key] = max(
                maxima[key],
                *(
                    abs(a - b)
                    for a, b in zip(getattr(expected, key), getattr(actual, key), strict=True)
                ),
            )
        maxima["p_U"] = max(maxima["p_U"], abs(expected.p_U - actual.p_U))
    passed = all(value < 1e-3 for value in maxima.values())
    return {
        "passed": passed,
        "samples": len(left),
        "max_differences": maxima,
        "threshold_exclusive": 1e-3,
        "release_or_stage_acceptance": False,
    }


def bind_rows(rows: list, prepared_inputs: list) -> None:
    index = {item["id_sha256"]: item for item in prepared_inputs}
    for row in rows:
        item = index[row["id_sha256"]]
        for key in ("input_sha256", "media_sha256"):
            if row[key] != item[key]:
                raise ValueError("inference inputs changed since preparation")
        if row["token_ids"] != list(item["answer_token_ids"].values()):
            raise ValueError("inference answer mapping changed")


async def vllm(selected: list, processor: Any, args: argparse.Namespace) -> tuple[list, list]:
    rows, summaries = [], []
    for cohort, root, cases in selected:
        route = "video" if cohort == "video" else "general"
        readouts, trace = await parity.run_vllm_pass(
            cases=cases,
            processor=processor,
            asset_root=root,
            served_model=getattr(args, route + "_model"),
            upstream_url=args.url,
            image_max_soft_tokens=1120,
        )
        rows.extend(parity.rows_to_json(readouts))
        summaries.extend({"cohort": cohort, "route": route, **item} for item in trace)
    return rows, summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "hf", "vllm", "compare"))
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "artifacts/s4-pilot-five-screen-001/manifest.json"
    )
    parser.add_argument("--per-cohort", type=int, choices=(2, 25), default=2)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--hf-report", type=Path)
    parser.add_argument("--vllm-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--general-model", default="s4-general")
    parser.add_argument("--video-model", default="s4-video")
    args = parser.parse_args()
    os.umask(0o077)
    # Reserve report first: never overwrite; a failure always leaves evidence.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sink = args.output.open("x")
    start = time.monotonic()
    report: dict[str, Any] = {
        "phase": args.phase,
        "status": "failed",
        "stage_gate": False,
        "cost": None,
        "seed": 42,
        "rows": [],
        "crossmodal_gap": "PreparationCase forbids mixed media; no synthetic substitute",
    }
    try:
        report["identity"] = identity(args.manifest)
        if args.phase == "compare":
            hf = json.loads(args.hf_report.read_text())
            serving = json.loads(args.vllm_report.read_text())
            if hf["status"] != "completed" or serving["status"] != "completed":
                raise ValueError("both actual inference phases must complete")
            for key in ("identity", "inputs"):
                if hf[key] != serving[key]:
                    raise ValueError(f"HF/vLLM {key} binding mismatch")
            if {k: v for k, v in hf["identity"].items() if k != "script_sha256"} != {
                k: v for k, v in report["identity"].items() if k != "script_sha256"
            }:
                raise ValueError("current source/adapter identity differs from inference reports")
            report["comparison"] = compare_small(hf["rows"], serving["rows"])
            report["input_report_hashes"] = {
                "hf": sha(args.hf_report),
                "vllm": sha(args.vllm_report),
            }
        else:
            selected = selection(args.manifest, args.per_cohort)
            processor = parity.load_official_processor(MODEL)
            report["inputs"] = inputs(processor, selected)
            if args.phase != "prepare":
                prepared = json.loads(args.prepared.read_text())
                if prepared["status"] != "completed" or prepared["identity"] != report["identity"]:
                    raise ValueError("prepare identity/status mismatch")
                same_inputs(report["inputs"], prepared)
                report["prepared_report_sha256"] = sha(args.prepared)
            if args.phase == "hf":
                for cohort, root, cases in selected:
                    route = "video" if cohort == "video" else "general"
                    rows = parity.run_hf_pass(
                        cases=cases,
                        processor=processor,
                        paths=parity.ParityPaths(
                            args.manifest, root, MODEL, ASSETS, ADAPTERS[route][0]
                        ),
                        dtype_name="bfloat16",
                        image_max_soft_tokens=1120,
                    )
                    report["rows"].extend(parity.rows_to_json(rows))
                    # Persist partial progress if a later cohort fails.
                    sink.seek(0)
                    sink.truncate()
                    json.dump(report, sink)
                    sink.flush()
            elif args.phase == "vllm":
                report["rows"], report["server_summaries"] = asyncio.run(
                    vllm(selected, processor, args)
                )
                report["server_identity_boundary"] = (
                    "model aliases operator-attested; client verifies actual prompt token IDs"
                )
        if args.phase in {"hf", "vllm"}:
            bind_rows(report["rows"], report["inputs"])
        report["status"] = "completed"
        return_code = 2 if args.phase == "compare" and not report["comparison"]["passed"] else 0
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        return_code = 1
    finally:
        report["duration_seconds"] = time.monotonic() - start
        report["exit_code"] = return_code if report["status"] == "completed" else 1
        report["environment"] = {"python": sys.version, "packages": parity.package_versions()}
        sink.seek(0)
        sink.truncate()
        json.dump(report, sink, indent=2)
        sink.write("\n")
        sink.close()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
