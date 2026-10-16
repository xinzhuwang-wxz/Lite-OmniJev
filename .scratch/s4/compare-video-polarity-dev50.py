"""CPU same-input original DEV50 comparison; no model execution or threshold fitting."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample
from scripts.s0_development_baselines import answer_tokens_for_record, score_records
from train.readout import read_probabilities


def module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def validate_record(r: dict[str, Any]) -> None:
    require(
        answer_tokens_for_record(r) == ("yes", "no")
        and set(r["answer_token_ids"]) == {"yes", "no", "U"}
        and len(set(r["answer_token_ids"].values())) == 3,
        "canonical_yes_no_U",
    )
    require(type(r["target_index"]) is int and r["target_index"] in [0, 1], "binary_gold")
    require(all(math.isfinite(v) and v <= 0 for v in r["raw_logprobs"].values()), "raw_logprobs")
    readout = read_probabilities(r["raw_logprobs"], ["yes", "no"], temperature=1)
    require(
        r["prediction"] == readout.prediction
        and math.isclose(r["p_U"], readout.p_insufficient, abs_tol=1e-12, rel_tol=0)
        and all(
            math.isclose(r["conditional_without_u"][k], v, abs_tol=1e-12, rel_tol=0)
            for k, v in readout.probabilities.items()
        )
        and all(
            math.isclose(r["joint_k_plus_u"][k], v, abs_tol=1e-12, rel_tol=0)
            for k, v in readout.joint_probabilities.items()
        ),
        "unchanged_readout",
    )


def paired_identity(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> None:
    require(
        len(first) == len(second) == 50
        and len({r["id"] for r in first}) == len({r["id"] for r in second}) == 50,
        "unique_same50",
    )
    for a, b in zip(first, second, strict=True):
        validate_record(a)
        validate_record(b)
        for k in [
            "id",
            "family_id",
            "question_id",
            "question_type",
            "target_index",
            "target_source",
            "modalities",
            "prompt_sha256",
            "payload_sha256",
            "input_sha256",
            "input_summary_sha256",
            "answer_token_ids",
            "frame_indices",
            "sampling_policy",
            "pass_index",
        ]:
            require(a[k] == b[k], "sameinput:" + k)
        require(
            a["sampling_policy"] == "uniform" and a["pass_index"] == 1,
            "single_uniform_original_pass",
        )


def training_fairness(
    control: list[Sample],
    polarity: list[Sample],
    control_ledger: list[dict[str, Any]],
    polarity_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    require(len(control) == len(polarity) == 120, "matched_train120")
    import random

    order = list(range(120))
    random.Random(42).shuffle(order)
    c, p = [[rows[i] for i in order] for rows in [control, polarity]]
    ledgers = [{r["item_id"]: r for r in rows} for rows in [control_ledger, polarity_ledger]]
    families = set()
    wrapper = "For this video, is the correct yes/no answer to the following question 'no'?\n\n"
    for step in range(10):
        require(
            c[step * 12 + 6 : step * 12 + 12] == p[step * 12 + 6 : step * 12 + 12], "same60_replay"
        )
        for offset in [0, 2, 4]:
            a, b = c[step * 12 + offset : step * 12 + offset + 2]
            x, y = p[step * 12 + offset : step * 12 + offset + 2]
            require(
                a.questions == b.questions == x.questions
                and a.prefix == b.prefix == x.prefix == y.prefix,
                "same_original_question_and_video",
            )
            require(
                len({r.meta.family_id for r in [a, b, x, y]}) == 1
                and a.meta.family_id not in families,
                "same30_unique_family",
            )
            families.add(a.meta.family_id)
            for r in [b, x, y]:
                require(
                    r.meta.model_dump(exclude={"data_version"})
                    == a.meta.model_dump(exclude={"data_version"}),
                    "same_train_component_metadata",
                )
            gold = a.questions[0].target
            require(
                isinstance(gold, HardTarget) and gold.value in ["yes", "no"],
                "train_binary_gold",
            )
            assert isinstance(gold, HardTarget)
            expected = x.questions[0].model_dump(mode="json")
            expected["instruction"] = wrapper + x.questions[0].instruction
            expected["target"] = {
                "kind": "hard",
                "value": "yes" if gold.value == "no" else "no",
                "source": "gold-derived-noul-no-complement-v1",
            }
            require(
                y.questions[0].model_dump(mode="json") == expected,
                "only_inverse_question_gold_change",
            )
            parents = [
                ledgers[i][r.id]["parent_sample_sha256"]
                for i, rows in enumerate([[a, b], [x, y]])
                for r in rows
            ]
            require(len(set(parents)) == 1, "same_parent_gold_binding")
    require(
        len(families) == 30
        and len({s.prefix.media[0].sha256 for s in control if s.meta.modality == ["video"]}) == 30,
        "matched30_families",
    )
    return {
        "same30_physical_families": True,
        "same60_replay_full_fields": True,
        "same_actual_seed42_pair_offsets": True,
        "only_treatment": "second question is no-answer wrapper plus complement; arm IDs differ",
        "historical_unseen_claim": False,
    }


def summarize(records: list[dict[str, Any]], auc: Any) -> dict[str, Any]:
    result = score_records(records)
    labels = [r["target_index"] == 0 for r in records]
    scores = [r["conditional_without_u"]["yes"] for r in records]
    correct = sum(r["prediction"] == ["yes", "no"][r["target_index"]] for r in records)
    nll = {}
    for key, tokens in [
        ("conditional_nll", ["yes", "no"]),
        ("joint_k_plus_u_nll", ["yes", "no", "U"]),
    ]:
        values = []
        for r in records:
            peak = max(r["raw_logprobs"][k] for k in tokens)
            values.append(
                peak
                + math.log(math.fsum(math.exp(r["raw_logprobs"][k] - peak) for k in tokens))
                - r["raw_logprobs"][["yes", "no"][r["target_index"]]]
            )
        nll[key] = math.fsum(values) / len(values)
    return {
        **result,
        **nll,
        "correct": correct,
        "count": len(records),
        "temporal_heuristic_p_yes_auc": auc(scores, labels),
        "prediction_counts": dict(Counter(r["prediction"] for r in records)),
        "gold_counts": {"yes": sum(labels), "no": len(labels) - sum(labels)},
        "mean_p_yes": math.fsum(scores) / len(scores),
        "mean_p_U": math.fsum(r["p_U"] for r in records) / len(records),
        "confusion": {
            gold + "->" + pred: sum(
                ["yes", "no"][r["target_index"]] == gold and r["prediction"] == pred
                for r in records
            )
            for gold in ["yes", "no"]
            for pred in ["yes", "no"]
        },
    }


def paired_changes(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, int]:
    corrected = regressed = changed = 0
    for a, b in zip(first, second, strict=True):
        gold = ["yes", "no"][a["target_index"]]
        corrected += a["prediction"] != gold and b["prediction"] == gold
        regressed += a["prediction"] == gold and b["prediction"] != gold
        changed += a["prediction"] != b["prediction"]
    return {"corrected": corrected, "regressed": regressed, "changed_predictions": changed}


def compare(root: Path) -> dict[str, Any]:
    h = module("reconcile-pilot-exposure")
    ranking = module("diagnose-video-ranking")
    prep_dir = root / "artifacts/s4-video-polarity-pilot-001"
    prep = h.read_json(prep_dir / "report.json")[0]
    require(
        h.sha(prep_dir / "report.json")
        == "1d14c7780892fe10eda4b2606dece555fbbd73179574f60b42ba46784411ab80",
        "frozen_prototype",
    )
    bindings = {str(prep_dir / "report.json"): h.sha(prep_dir / "report.json")}
    rows = {}
    trains = {}
    prepared = {}
    ledgers = {}
    identities = {}
    for arm in ["control", "polarity"]:
        name = "s4-video-polarity-" + arm
        td = root / "artifacts" / (name + "-train-001")
        final, fb = h.read_json(root / "reports" / (name + "-training-001.json"))
        runtime, rb = h.read_json(td / "run.json")
        require(
            final["status"] == runtime["status"] == "completed"
            and final["steps_completed"] == runtime["steps_completed"] == 10
            and final["exit_code"] == runtime["exit_code"] == 0,
            "both_terminal10",
        )
        extra, _ = h.polarity_bindings(root, prep_dir / arm, prep, final, runtime, arm)
        bindings.update(extra)
        for b in [fb, rb]:
            bindings[b["path"]] = b["sha256"]
        require(
            h.sha(td / "data.jsonl") == h.POLARITY_DATA[arm] == final["data_sha256"],
            "actual_train_data",
        )
        h.equivalent(td / "data.jsonl", prep_dir / arm / "train.jsonl")
        bindings[str(td / "data.jsonl")] = h.sha(td / "data.jsonl")
        prepared[arm] = [
            Sample.model_validate_json(line)
            for line in (prep_dir / arm / "train.jsonl").read_bytes().splitlines()
        ]
        ledgers[arm] = [
            json.loads(line)
            for line in (prep_dir / arm / "allocation-ledger.jsonl").read_bytes().splitlines()
        ]
        trains[arm] = final
        summary_path = root / "reports" / (name + "-dev50-001.json")
        summary, sb = h.read_json(summary_path)
        launch = h.read_json(root / "reports" / (name + "-launch-001.json"))[0]
        require(
            h.sha(summary_path) == launch["dev50_report_sha256"]
            and summary["status"] == "completed"
            and summary["exit_code"] == 0
            and summary["same_input_rows"] == 50
            and summary["adapter_sha256"] == final["adapter_sha256"],
            "dev_summary_chain",
        )
        bindings[sb["path"]] = sb["sha256"]
        dev = root / "artifacts" / (name + "-dev50-001") / "uniform"
        receipt, eb = h.read_json(dev / "receipt.json")
        require(
            eb["sha256"] == summary["candidate_receipt_sha256"]
            and receipt["status"] == "completed"
            and receipt["exit_code"] == 0
            and receipt["identity"]["adapter_sha256"] == final["adapter_sha256"]
            and receipt["identity"]["passes"] == 1,
            "dev_receipt_chain",
        )
        bindings[eb["path"]] = eb["sha256"]
        records = dev / "records.jsonl"
        require(h.sha(records) == receipt["records_sha256"], "dev_sidecar_sha")
        bindings[str(records)] = h.sha(records)
        rows[arm] = [json.loads(line) for line in records.read_bytes().splitlines()]
        identities[arm] = receipt["identity"]
        require(score_records(rows[arm]) == receipt["metrics"], "unchanged_saved_metrics")
    require(
        trains["control"]["init_adapter"] == trains["polarity"]["init_adapter"],
        "same_actual_parent",
    )
    configs = []
    for arm in ["control", "polarity"]:
        config = dict(trains[arm]["config"])
        config["expected_data_versions"] = [
            v.replace("polarity-" + arm + "-v1", "polarity-ARM-v1")
            for v in config["expected_data_versions"]
        ]
        configs.append(config)
    require(configs[0] == configs[1], "same_training_hyperparameters")
    require(
        {
            k: v
            for k, v in identities["control"].items()
            if k not in ["adapter_sha256", "adapter_config_sha256"]
        }
        == {
            k: v
            for k, v in identities["polarity"].items()
            if k not in ["adapter_sha256", "adapter_config_sha256"]
        },
        "same_dev_identity",
    )
    fairness = training_fairness(
        prepared["control"], prepared["polarity"], ledgers["control"], ledgers["polarity"]
    )
    paired_identity(rows["control"], rows["polarity"])
    models = {arm: summarize(records, ranking.auc) for arm, records in rows.items()}
    require(all(h.sha(Path(path)) == value for path, value in bindings.items()), "mutable_inputs")
    return {
        "status": "compared",
        "exit_code": 0,
        "gpu_used": False,
        "stage_gate": False,
        "scope": "fixed heuristic temporal DEV50; descriptive comparison, no threshold fit",
        "same_input_rows": 50,
        "training_fairness": fairness,
        "models": models,
        "paired": {
            "control_to_polarity": paired_changes(rows["control"], rows["polarity"]),
            "polarity_to_control": paired_changes(rows["polarity"], rows["control"]),
        },
        "bindings": bindings,
        "helper_sha256": h.sha(Path(__file__)),
        "ranking_helper_sha256": h.sha(Path(ranking.__file__)),
        "exposure_helper_sha256": h.sha(Path(h.__file__)),
        "cost": None,
    }


def main() -> int:
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    start = time.monotonic()
    require(
        a.root.is_absolute() and a.output.is_absolute() and not a.output.exists(),
        "new_absolute_output",
    )
    try:
        r = compare(a.root)
    except Exception as error:
        r = {
            "status": "rejected",
            "exit_code": 1,
            "gpu_used": False,
            "category": str(error) if isinstance(error, ValueError) else type(error).__name__,
            "cost": None,
        }
    r["duration_seconds"] = time.monotonic() - start
    with a.output.open("x") as f:
        json.dump(r, f, indent=2)
        f.write("\n")
    print(json.dumps({k: r[k] for k in ["status", "exit_code", "gpu_used"]}))
    return int(r["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
