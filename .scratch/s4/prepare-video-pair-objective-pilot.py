"""Build the actual fixed-course cross-video sidecar on CPU; never launch training."""

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

from train.trainer import LoopConfig, smoke_samples
from train.video_pairs import build_video_pairs, load_video_pairs

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
SOURCE = Path("/root/autodl-tmp/s4-video-pair-code-7b75f9f")
ARCHIVE_SHA = "49b11d244dc483a59895695468697e05ee4d45ff8b491cad377724137b15d32d"
DATA_SHA = "17cb9a0066639dafd1feb21e3819cc38a367c86e78ce2a2907a5047d6cb437aa"
LEDGER_SHA = "1d18f75d395a6935b1fdececd7530d543144b4fa0e2a0974c31541b3d9c5fbb3"
METADATA_SHA = "879fec9a0dabe9f12fd65eddaa8f963d93f3c2c58d5d8555eada8fa2c1f44afe"
CONFIG_SHA = "e01916029545aea72be7e03dc2cbb9a5b35530c26c4a71711ec1bd4518c6f871"
QUESTIONS_SHA = "11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4"
EXPOSURE_SHA = "24b2bcb85e7067c3759cba3f007782c46b174695fd77399da15bc468cfb3a35e"
POLICY_SHA = "681429177c887d11991a4b148a5801ab4e33d2ba4b3fb7b1f84c694c44a20f04"
CE_REPORT_SHA = "11e14016bfe5e546fe67bb0b7b1a21dbc5b89a082253ffbb743ee3bbd4e90a05"
INIT_SHA = "62920299c7a1ac33b2a874b586a9d7f3ea3eaa4e38198bc20388642482318fc6"
INIT_CONFIG_SHA = "8d79bb86dc522bc6da2b3de170dc704e51b60e2d2518582b07e8ffbfcdd6bc0a"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound(path, expected):
    require(sha(path) == expected, f"SHA mismatch: {path}")
    return {"path": str(path), "sha256": expected}


def read(path):
    return json.loads(path.read_text())


def derive_config(original, sidecar, ledger):
    changes = {
        "video_pair_loss_weight": 0.1,
        "video_pair_sidecar": str(sidecar),
        "video_pair_sidecar_sha256": sha(sidecar),
        "video_pair_ledger": str(ledger),
        "video_pair_ledger_sha256": sha(ledger),
    }
    require(
        not set(changes).intersection(original), "CE control already contains video pair fields"
    )
    result = original | changes
    LoopConfig.model_validate(result)
    require(
        {k for k in result if result[k] != original.get(k)} == set(changes),
        "only five video pair config fields may change",
    )
    require(
        result["checkpoint_every"] == result["midway_step"] == 10,
        "retain actual CE checkpoint/adapter cadence",
    )
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "explicitly hide CUDA for CPU-only prep")
    require(Path.cwd().resolve() == SOURCE, "run from immutable source projection")
    require(
        Path(__import__("train.video_pairs", fromlist=["__file__"]).__file__)
        .resolve()
        .parent.parent
        == SOURCE,
        "wrong imported video pair core",
    )
    out = ROOT / "artifacts/s4-video-pair-objective-pilot-001"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "failed",
        "exit_code": 1,
        "gpu_used": False,
        "training_authorized": False,
        "stage_gate": False,
        "cost": None,
        "source_commit": "7b75f9f",
        "invocation_steps_initial": 10,
        "scope": "Existing TRAIN1200 only; metadata/program/gold/ledger CPU validation",
    }
    try:
        prepared = ROOT / "artifacts/s4-continuous-matched-pool-001"
        original_config = ROOT / "artifacts/s4-rank64-matched-pilot-001/config-ce.json"
        official = Path(
            "/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official/files/questions/train.json"
        )
        initializer = ROOT / "artifacts/s4-rank64-initializer-001"
        exposure = ROOT / "artifacts/s4-pilot-exposure-021/report.json"
        policy = ROOT / "reports/explore/video-pair-objective-feasibility-001.md"
        ce_report = ROOT / "reports/s4-rank64-matched-pilot-training-001.json"
        report["bindings"] = {
            name: bound(path, expected)
            for name, path, expected in [
                ("data", prepared / "train.jsonl", DATA_SHA),
                ("ledger", prepared / "allocation-ledger.jsonl", LEDGER_SHA),
                ("matched_metadata", prepared / "matched-pair-metadata.json", METADATA_SHA),
                ("control_config", original_config, CONFIG_SHA),
                ("official_train_questions", official, QUESTIONS_SHA),
                ("latest_exposure", exposure, EXPOSURE_SHA),
                ("proposal", policy, POLICY_SHA),
                ("actual_ce_control", ce_report, CE_REPORT_SHA),
                ("initializer", initializer / "adapter_model.safetensors", INIT_SHA),
                ("initializer_config", initializer / "adapter_config.json", INIT_CONFIG_SHA),
                (
                    "source_archive",
                    Path("/root/autodl-tmp/omnijev-video-pair-code-7b75f9f.tar"),
                    ARCHIVE_SHA,
                ),
            ]
        }
        er = read(exposure)
        require(
            er["exit_code"] == 0 and not er["unknown_update_bearing_inputs"],
            "exposure unknown inputs",
        )
        control = read(ce_report)
        require(
            control["status"] == "paused"
            and control["exit_code"] == 0
            and control["steps_completed"] == 10
            and control["init_adapter"]["weights_sha256"] == INIT_SHA,
            "actual same-initializer CE first10 control",
        )
        config = LoopConfig.model_validate(read(original_config))
        samples = smoke_samples(
            config, prepared / "train.jsonl", expected_data_versions=config.expected_data_versions
        )
        videos = [sample for sample in samples if sample.meta.modality == ["video"]]
        require(
            len(videos) == len({s.meta.family_id for s in videos}) == 600, "fixed600 video families"
        )
        parser_path = Path(
            "/root/autodl-tmp/s4-video-pair-prep-helpers/inspect-temporal-order-contrast.py"
        )
        bound(parser_path, "c4346e8d61613c710e9dbca78d6796776984402bfc85f30add342d857ddfa5c8")
        spec = importlib.util.spec_from_file_location("official_selected_scenes", parser_path)
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        scenes = parser.selected_scenes(
            official, {int(s.meta.family_id.split(":")[1]) for s in videos}
        )
        questions = {}
        for sample in videos:
            scene = scenes[int(sample.meta.family_id.split(":")[1])]
            matches = [
                q
                for q in scene["questions"]
                if q["question_id"] == int(sample.id.rsplit("-", 1)[1])
            ]
            require(len(matches) == 1, "one original official question per selected sample")
            questions[sample.id] = matches[0]
        sidecar = out / "video-pairs.json"
        payload = build_video_pairs(
            samples, (prepared / "matched-pair-metadata.json").read_text(), questions, LEDGER_SHA
        )
        sidecar.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
        derived = derive_config(
            read(original_config), sidecar, prepared / "allocation-ledger.jsonl"
        )
        config_path = out / "config-video-pair.json"
        config_path.write_text(json.dumps(derived, indent=2) + "\n")
        strict_config = LoopConfig.model_validate_json(config_path.read_text())
        pairs = load_video_pairs(strict_config, samples, prepared / "train.jsonl")
        require(len(pairs) == 300, "all300 pairs audited")
        report.update(
            status="cpu_prepared_only",
            exit_code=0,
            samples=1200,
            audited_pairs=300,
            official_train_question_records=600,
            unique_video_families=600,
            replay_draws=600,
            actual_draw_pair_offsets=[0, 2, 4],
            new_data_added=False,
            output_sha256={p.name: sha(p) for p in [sidecar, config_path]},
            data_path=str(prepared / "train.jsonl"),
            ledger_path=str(prepared / "allocation-ledger.jsonl"),
            source_sha256={
                name: sha(SOURCE / name)
                for name in [
                    "train/trainer.py",
                    "train/video_pairs.py",
                    "train/video_pair_objective.py",
                ]
            },
            wrapper_sha256=sha(Path(__file__)),
            model_media_preflight="not run; metadata and official TRAIN provenance only",
            checkpoint_cadence="retained checkpoint10 for matched CE; proposal step5 not applied",
            next_step="Root decision and real-model preflight required before training",
        )
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}")
    finally:
        report["duration_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
