"""Aggregate saved public DEV video readouts and admitted TRAIN capacity; no GPU."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import Sample, answer_tokens
from eval.metrics import evaluate
from scripts.gold_inventory import digest
from scripts.s1_dev_eval import metrics_from_records
from scripts.s1_text_pools import canonical_sample_sha
from train.readout import read_probabilities

DATA_SHA = "79ed30be3a687a6c767c4d7c55c24dc7ac08d5a1698a6f49e3bc7d47e74e63d2"
ADAPTERS = [
    "1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755",
    "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818",
]


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def sha(path: Path) -> str:
    return digest(path.read_bytes())


def read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def kind(sample: Sample) -> str:
    words = set(sample.questions[0].instruction.lower().split())
    return (
        "temporal"
        if words & {"before", "after", "when", "while"}
        else "motion"
        if words & {"moving", "stationary"}
        else "existence"
    )


def field(command: list[str], flag: str) -> str:
    require(command.count(flag) == 1, "command_flag")
    return command[command.index(flag) + 1]


def bind_record(record: dict, sample: Sample) -> dict:
    question = sample.questions[0]
    require(
        record["id"] == sample.id and record["pool"] == sample.meta.pool == "dev", "row_identity"
    )
    require(record["family_hash"] == sample.meta.family_id, "row_family")
    require(record["target"] == question.target.value, "row_target")
    require(list(answer_tokens(question)) == ["yes", "no"], "answer_mapping")
    require(
        set(record["raw_logits"]) == {"yes", "no", "U"}
        and set(record["probabilities"]) == {"yes", "no"},
        "answer_mapping",
    )
    actual = read_probabilities(record["raw_logits"], ["yes", "no"], temperature=1)
    require(
        all(
            math.isclose(actual.probabilities[k], record["probabilities"][k], abs_tol=1e-12)
            for k in ["yes", "no"]
        ),
        "probability_binding",
    )
    require(
        math.isclose(actual.p_insufficient, record["p_insufficient"], abs_tol=1e-12), "u_binding"
    )
    return {
        **record,
        "canonical_prediction": actual.prediction,
        "canonical_sample_sha256": canonical_sample_sha(sample),
    }


def load(root: Path, name: str, index: int) -> tuple[dict, dict, dict]:
    path = root / "reports" / (name + ".json")
    report = read(path)
    require(
        report["status"] == "passed" and report["exit_code"] == 0 and report["samples"] == 500,
        "report_status",
    )
    require(
        report["passes"] == (2 if index == 0 else 1) and report["video_num_frames"] == 16,
        "recorded_config",
    )
    cmd = report["command"]
    data = Path(field(cmd, "--data"))
    out = Path(field(cmd, "--output-dir"))
    if not out.is_absolute():
        out = root / out
    sidecar = out / "records.jsonl"
    adapter = Path(field(cmd, "--adapter"))
    require(sha(data) == report["data_sha256"] == DATA_SHA, "data_sha")
    require(sha(sidecar) == report["sidecar_sha256"], "sidecar_sha")
    require(
        sha(adapter / "adapter_model.safetensors") == report["adapter_sha256"] == ADAPTERS[index],
        "adapter_sha",
    )
    samples = {s.id: s for s in (Sample.model_validate(r) for r in rows(data))}
    require(
        len(samples) == 500
        and all(s.meta.pool == "dev" and s.meta.modality == ["video"] for s in samples.values()),
        "public_dev_scope",
    )
    selected = [r for r in rows(sidecar) if r["pass"] == 0 and r["variant"] == "original"]
    require(
        len(selected) == 500
        and len({r["id"] for r in selected}) == 500
        and {r["id"] for r in selected} == samples.keys(),
        "record_coverage",
    )
    bound = {r["id"]: bind_record(r, samples[r["id"]]) for r in selected}
    return (
        report,
        samples,
        {
            "records": bound,
            "report_binding": {"path": str(path), "sha256": sha(path)},
            "data_binding": {"path": str(data), "sha256": sha(data)},
            "sidecar_binding": {"path": str(sidecar), "sha256": sha(sidecar)},
            "adapter_sha256": report["adapter_sha256"],
            "script_sha256": report["script_sha256"],
            "model_revision": report["model_revision"],
            "versions": report["versions"],
            "recorded_config": {
                "model_dir": field(cmd, "--model-dir"),
                "media_root": field(cmd, "--media-root"),
                "video_num_frames": 16,
                "image_max_soft_tokens": report["image_max_soft_tokens"],
            },
        },
    )


def summarize(first: dict, second: dict) -> dict:
    require(first.keys() == second.keys(), "matched_ids")
    for ident in first:
        require(
            first[ident]["canonical_sample_sha256"] == second[ident]["canonical_sample_sha256"]
            and first[ident]["target"] == second[ident]["target"],
            "matched_inputs",
        )
    result = {}
    for name, records in [("s3", first), ("coverage", second)]:
        values = list(records.values())
        confusion = Counter(r["target"] + "->" + r["prediction"] for r in values)
        result[name] = {
            "metrics": metrics_from_records(values),
            "correct": sum(r["prediction"] == r["target"] for r in values),
            "confusion": dict(confusion),
            "canonical_tie_break_prediction_differences": sum(
                r["prediction"] != r["canonical_prediction"] for r in values
            ),
            "mean_p_yes": sum(r["probabilities"]["yes"] for r in values) / len(values),
        }
    for key, prediction in [
        ("saved_product", "prediction"),
        ("canonical_yes_no", "canonical_prediction"),
    ]:
        result[key + "_paired"] = dict(
            Counter(
                ("correct" if first[i][prediction] == first[i]["target"] else "wrong")
                + "->"
                + ("correct" if second[i][prediction] == second[i]["target"] else "wrong")
                for i in first
            )
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    a = parser.parse_args()
    started = time.monotonic()
    br, bs, b = load(a.root, "s3-dev-eval-video-stable-001", 0)
    ar, ass, after = load(a.root, "s4-coverage-video-full-001", 1)
    require(
        br["model_revision"] == ar["model_revision"]
        and b["recorded_config"] == after["recorded_config"],
        "recorded_input_config_match",
    )
    baseline, candidate = b.pop("records"), after.pop("records")
    require(
        all(bs[i].model_dump(mode="json") == ass[i].model_dump(mode="json") for i in bs),
        "matched_full_samples",
    )
    result = summarize(baseline, candidate)
    old_values = list(baseline.values())
    result["historical_s3_report_metrics"] = br["metrics"]
    result["historical_s3_sorted_no_yes_recalculation"] = evaluate(
        [[r["probabilities"][k] for k in ["no", "yes"]] for r in old_values],
        [["no", "yes"].index(r["target"]) for r in old_values],
    )
    result["canonical_probability_ties"] = {
        name: sum(r["probabilities"]["yes"] == r["probabilities"]["no"] for r in values.values())
        for name, values in [("s3", baseline), ("coverage", candidate)]
    }
    result["heuristic_groups"] = {
        k: summarize(
            {i: r for i, r in baseline.items() if kind(bs[i]) == k},
            {i: r for i, r in candidate.items() if kind(bs[i]) == k},
        )
        for k in ["existence", "motion", "temporal"]
    }
    # Failure tests mutate only in-memory identity metadata; no data files changed.
    rejected = []
    for test in ["target", "raw_logits", "family_hash"]:
        bad = copy.deepcopy(next(iter(baseline.values())))
        if test == "raw_logits":
            bad[test].pop("U")
        elif test == "target":
            bad[test] = "U"
        else:
            bad[test] = "changed"
        try:
            bind_record(bad, bs[bad["id"]])
        except ValueError:
            rejected.append(test)
        else:
            raise AssertionError("identity_failure_not_rejected")
    crpath = a.root / "artifacts/s4-crossmodal-retention-pilot-002/report.json"
    cr = read(crpath)
    second = a.root / "artifacts/s4-video-pilot-001"
    sr = read(second / "report.json")
    require(
        sha(second / "train.jsonl")
        == cr["prior_draw_bindings"]["second"]["prepared_data"]["sha256"]
        == sr["output_sha256"]["train.jsonl"],
        "coverage_source_lineage",
    )
    trainroot = a.root / "artifacts/s3-pools-video-001/run-a"
    tr = read(trainroot / "report.json")
    trainpath = trainroot / "train.jsonl"
    require(
        sha(trainpath) == sr["input_sha256"]["video_source"] == tr["outputs"]["train_jsonl_sha256"],
        "train_source_sha",
    )
    require(
        sha(trainroot / "report.json") == sr["input_sha256"]["video"]["report"], "train_receipt_sha"
    )
    train = [Sample.model_validate(r) for r in rows(trainpath)]
    require(
        len(train) == 8299
        and all(s.meta.pool == "train" and s.meta.modality == ["video"] for s in train),
        "train_only",
    )
    capacity = {
        k: {
            "samples": sum(kind(s) == k for s in train),
            "families": len({s.meta.family_id for s in train if kind(s) == k}),
            "targets": dict(Counter(s.questions[0].target.value for s in train if kind(s) == k)),
        }
        for k in ["existence", "motion", "temporal"]
    }
    result.update(
        {
            "scope": "saved public DEV500 and TRAIN-only capacity; no GPU or new samples",
            "input_identity_boundary": (
                "Same SHA-bound data and recorded config, not prepared tensor/token "
                "equality; neither sidecar persists input_sha/token IDs. Video soft-token "
                "count is not directly recorded; image_max_soft_tokens effective280 is not a "
                "per-video-frame count."
            ),
            "answer_mapping_boundary": (
                "Legal yes/no/U labels verified; numeric tokenizer IDs not persisted."
            ),
            "heuristic": (
                "screen rule: temporal before/after/when/while first, motion "
                "moving/stationary next, otherwise existence; not official task labels"
            ),
            "bindings": {
                "s3": b,
                "coverage": after,
                "coverage_preparation": {"path": str(crpath), "sha256": sha(crpath)},
                "video_preparation": {
                    "path": str(second / "report.json"),
                    "sha256": sha(second / "report.json"),
                },
                "train_source": {"path": str(trainpath), "sha256": sha(trainpath)},
                "train_report": {
                    "path": str(trainroot / "report.json"),
                    "sha256": sha(trainroot / "report.json"),
                },
            },
            "train_capacity": capacity,
            "train_unique_families": len({s.meta.family_id for s in train}),
            "identity_rejection_checks": rejected,
            "gpu_used": False,
            "cost": None,
            "duration_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "script_sha256": globals().get("SOURCE_SHA") or sha(Path(__file__)),
            "exit_code": 0,
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
