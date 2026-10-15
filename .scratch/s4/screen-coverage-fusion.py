"""Single alpha.5 first-pilot/coverage effective-delta fusion and fixed dev screen.

Default is CPU fusion plus bound preparation only; --execute explicitly enables
GPU. Frozen evaluator/order/thresholds are imported from the coverage screen.
This is public-dev screening, never stage acceptance; no ratio sweep.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path

SHARED = Path(__file__).with_name("screen-crossmodal-retention.py")
_spec = importlib.util.spec_from_file_location("coverage_screen", SHARED)
assert _spec is not None and _spec.loader is not None
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
ROOT, ORDER, MINIMUM, CROSS = _shared.ROOT, _shared.ORDER, _shared.MINIMUM, _shared.CROSS
sha, read, require = _shared.sha, _shared.read, _shared.require
cross_forward, ordinary = _shared.cross_forward, _shared.ordinary
FIRST = "6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac"
SECOND = "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--first", type=Path, default=ROOT / "artifacts/s4-joint-pilot-train-001/adapter"
    )
    parser.add_argument(
        "--second", type=Path, default=ROOT / "artifacts/s4-crossmodal-retention-train-001/adapter"
    )
    parser.add_argument(
        "--adapter", type=Path, default=ROOT / "artifacts/s4-coverage-fusion050-001"
    )
    parser.add_argument("--expected-first-config-sha256", required=True)
    parser.add_argument("--expected-second-config-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--through", choices=ORDER, default="video")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/s4-coverage-fusion050-screen-001"
    )
    args = parser.parse_args()
    os.umask(0o077)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    start = time.monotonic()
    receipt = args.output / "receipt.json"
    if args.resume:
        result = read(receipt)
    else:
        args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        result = {
            "status": "preparing",
            "exit_code": 1,
            "results": {},
            "cost": None,
            "stage_gate": False,
        }

    def save():
        result["duration_seconds_this_invocation"] = time.monotonic() - start
        temp = receipt.with_suffix(".tmp")
        with temp.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        temp.replace(receipt)

    try:
        adapter = args.adapter
        for directory, weights_sha, config_sha in (
            (args.first, FIRST, args.expected_first_config_sha256),
            (args.second, SECOND, args.expected_second_config_sha256),
        ):
            require(
                sha(directory / "adapter_model.safetensors") == weights_sha,
                "source weights SHA mismatch",
            )
            require(
                sha(directory / "adapter_config.json") == config_sha, "source config SHA mismatch"
            )
        if not adapter.exists():
            require(not args.resume, "resume fusion missing")
            from scripts.lora_fusion import fuse

            fuse(args.first, args.second, FIRST, SECOND, adapter, alpha=0.5, seed=0)
        fusion = read(adapter / "fusion_report.json")
        require(
            fusion["status"] == "ready_for_gpu_evaluation"
            and fusion["exit_code"] == 0
            and fusion["promotion_allowed"] is True
            and fusion["alpha"] == 0.5
            and fusion["seed"] == 0
            and math.isfinite(fusion["max_relative_frobenius_error"])
            and fusion["max_relative_frobenius_error"] <= 1e-3,
            "fusion is not the approved bounded alpha.5 candidate",
        )
        require(fusion["expected_weights_sha256"] == [FIRST, SECOND], "fusion sources mismatch")
        require(
            fusion["source_paths"] == [str(args.first.resolve()), str(args.second.resolve())],
            "fusion source paths mismatch",
        )
        require(
            fusion["sources"]
            == [
                {"weights_sha256": FIRST, "config_sha256": args.expected_first_config_sha256},
                {"weights_sha256": SECOND, "config_sha256": args.expected_second_config_sha256},
            ],
            "fusion source identity mismatch",
        )
        require(
            fusion["code_sha256"] == sha(Path("scripts/lora_fusion.py")), "fusion algorithm changed"
        )
        require(
            sha(adapter / "adapter_model.safetensors") == fusion["output"]["weights_sha256"],
            "fusion weights mismatch",
        )
        require(
            sha(adapter / "adapter_config.json")
            == fusion["output"]["config_sha256"]
            == args.expected_first_config_sha256,
            "fusion config mismatch",
        )
        prior = ROOT / "artifacts/s4-pilot-five-screen-001"
        manifest, commands = read(prior / "manifest.json"), read(prior / "commands.json")
        identity = {
            "fusion_report_sha256": sha(adapter / "fusion_report.json"),
            "fusion_algorithm_sha256": sha(Path("scripts/lora_fusion.py")),
            "shared_evaluator_sha256": sha(SHARED),
            "sources": fusion["sources"],
            "alpha": 0.5,
            "adapter_sha256": fusion["output"]["weights_sha256"],
            "adapter_config_sha256": sha(adapter / "adapter_config.json"),
            "prior_manifest_sha256": sha(prior / "manifest.json"),
            "prior_commands_sha256": sha(prior / "commands.json"),
            "runner_sha256": sha(Path(__file__)),
            "crossmodal_pins": CROSS,
        }
        # JSON round-trip keeps tuple-based pins identical after receipt reload.
        identity = json.loads(json.dumps(identity))
        if args.resume:
            require(result["identity"] == identity, "candidate/source/runner identity changed")
            require(
                result["status"] in ("prepared", "partial_pass", "completed"),
                "failed screens cannot resume",
            )
            require(
                list(result["results"]) == list(ORDER[: len(result["results"])]),
                "non-prefix screen history",
            )
            for item in result["results"].values():
                require(
                    item["passed"]
                    and sha(Path(item["report"])) == item["report_sha256"]
                    and sha(Path(item["sidecar"])) == item["sidecar_sha256"],
                    "prior stage evidence changed",
                )
        result.update(
            identity=identity,
            through=args.through,
            scope="frozen public-dev screening only; no stage acceptance",
            thresholds=MINIMUM,
            video_subgroups={"temporal": 60, "motion": 7},
        )
        if not args.execute:
            result.update(status="prepared", exit_code=0)
            save()
            return 0
        result.update(status="running", exit_code=1)
        save()
        for name in ORDER[: ORDER.index(args.through) + 1]:
            if name in result["results"]:
                continue
            result["active_stage"] = name
            stage_started = time.monotonic()
            stage = args.output / name
            stage.mkdir(exist_ok=False, mode=0o700)
            if name in CROSS:
                report = cross_forward(name, adapter, stage)
                path = stage / "report.json"
                with path.open("x") as stream:
                    json.dump(report, stream, indent=2)
                item = {
                    "count": 50,
                    "correct": round(report["metrics"]["accuracy"] * 50),
                    "selected_ids_sha256": report["selected_ids_sha256"],
                    "data_sha256": report["data_sha256"],
                    "report": str(path),
                    "report_sha256": sha(path),
                    "sidecar": str(stage / "records.jsonl"),
                    "sidecar_sha256": sha(stage / "records.jsonl"),
                }
            else:
                source = next(source for source in manifest["sources"] if source["cohort"] == name)
                command = next(command for command in commands if command["source"] == name)
                item = ordinary(name, adapter, stage, source, command)
            require(
                sha(adapter / "adapter_model.safetensors") == identity["adapter_sha256"]
                and sha(adapter / "adapter_config.json") == identity["adapter_config_sha256"],
                "candidate adapter changed during screen",
            )
            item.update(required_correct=MINIMUM[name], passed=item["correct"] >= MINIMUM[name])
            if name == "video":
                item["passed"] &= (
                    item["groups"].get("temporal", {}).get("correct", 0) >= 60
                    and item["groups"].get("motion", {}).get("correct", 0) >= 7
                )
            item["duration_seconds"] = time.monotonic() - stage_started
            result["results"][name] = item
            result.update(
                status="partial_pass" if item["passed"] else "screen_failed",
                exit_code=0 if item["passed"] else 2,
            )
            save()
            if not item["passed"]:
                return 2
        result.update(
            status="completed" if len(result["results"]) == len(ORDER) else "partial_pass",
            exit_code=0,
        )
        save()
        return 0
    except Exception as exc:
        result.update(status="execution_failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        save()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
