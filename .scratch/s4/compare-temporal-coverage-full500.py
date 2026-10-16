"""Bound public DEV500 comparison; optimistic ranking fits are never adopted."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path

from data.schema import Sample


def helper(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired = helper("compare-temporal-curriculum")
ranking = helper("diagnose-video-ranking")
DATA_SHA = "79ed30be3a687a6c767c4d7c55c24dc7ac08d5a1698a6f49e3bc7d47e74e63d2"
ADAPTERS = [
    "0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818",
    "250ef3e3c8a5359e6d7b6aaaeafa6d12b253603669a4b9f8a7a70c52c4b72707",
]


def same_forward(old, new):
    def extract(source):
        tree = ast.parse(source)
        return ast.dump(
            next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "forward_readout"
            ),
            include_attributes=False,
        )

    return extract(old) == extract(new)


def analyze(samples, before, after):
    result = paired.compare(samples, before, after)
    result.pop("changed_predictions")
    result["ranking"] = {}
    result["confusion"] = {}
    for name in ["all", "temporal", "motion", "existence"]:
        indices = [i for i, s in enumerate(samples) if name == "all" or paired.group(s) == name]
        result["ranking"][name] = {}
        result["confusion"][name] = {}
        for arm, rows in [("baseline", before), ("candidate", after)]:
            chosen = [dict(rows[i], canonical_prediction=rows[i]["prediction"]) for i in indices]
            result["ranking"][name][arm] = ranking.summarize(chosen)
            result["confusion"][name][arm] = {
                gold: {
                    pred: sum(r["target"] == gold and r["prediction"] == pred for r in chosen)
                    for pred in ["yes", "no"]
                }
                for gold in ["yes", "no"]
            }
        result["summaries"][name]["both_wrong"] = sum(
            before[i]["prediction"] != before[i]["target"]
            and after[i]["prediction"] != after[i]["target"]
            for i in indices
        )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    d = a.inputs
    data = d / "dev.jsonl"
    paired.require(paired.sha(data) == DATA_SHA, "public source SHA changed")
    samples = [Sample.model_validate_json(x) for x in data.read_text().splitlines() if x.strip()]
    reports = [paired.read(d / (arm + "-report.json")) for arm in ["baseline", "candidate"]]
    rows = []
    for arm, adapter, report in zip(["baseline", "candidate"], ADAPTERS, reports, strict=True):
        path = d / (arm + ".jsonl")
        paired.require(
            report["status"] == "passed" and report["exit_code"] == 0, "evaluation incomplete"
        )
        paired.require(
            report["adapter_sha256"] == adapter and report["data_sha256"] == DATA_SHA,
            "model/source identity",
        )
        paired.require(paired.sha(path) == report["sidecar_sha256"], "records SHA changed")
        rows.append([json.loads(x) for x in path.read_text().splitlines() if x.strip()])
    paired.require(len(samples) == 500, "requires full500")
    for key in [
        "model_revision",
        "samples",
        "passes",
        "video_num_frames",
        "versions",
        "image_max_soft_tokens",
        "counterfactual",
    ]:
        paired.require(reports[0][key] == reports[1][key], "protocol differs: " + key)
    old = d / "baseline-evaluator.py"
    new = Path(__file__).resolve().parents[2] / "scripts/s1_dev_eval.py"
    paired.require(
        paired.sha(old) == reports[0]["script_sha256"]
        and paired.sha(new) == reports[1]["script_sha256"],
        "evaluator SHA mismatch",
    )
    paired.require(same_forward(old.read_text(), new.read_text()), "forward implementation differs")
    result = analyze(samples, *rows)
    result["fixed_prefixes"] = {
        str(n): paired.compare(samples[:n], rows[0][:n], rows[1][:n])["summaries"]["all"]
        for n in [25, 100, 250, 500]
    }
    result["prefix_boundary"] = (
        "Original source order, fixed nodes; no prefix selected for benefit."
    )
    result.update(
        status="completed",
        exit_code=0,
        cost=None,
        input_comparability={
            "same_source_sha500_order_family_target": True,
            "same_recorded_protocol": True,
            "forward_readout_ast_identical": True,
            "tensor_exact_verified": False,
            "boundary": "Historical sidecars lack prepared tensor/input hashes. Source hash and "
            "forward implementation match; decoded tensors were not retained or rerun.",
        },
        bindings={
            "data_sha256": DATA_SHA,
            "adapters": ADAPTERS,
            "report_sha256": [
                paired.sha(d / (arm + "-report.json")) for arm in ["baseline", "candidate"]
            ],
            "sidecar_sha256": [r["sidecar_sha256"] for r in reports],
            "script_sha256": [r["script_sha256"] for r in reports],
        },
        oracle_boundary="DEV gold optimistic diagnostic only; subgroup fits cannot be added; "
        "no threshold exported/adopted, not generalization or calibration evidence",
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
