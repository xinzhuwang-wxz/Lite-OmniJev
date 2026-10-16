"""CPU paired identity and shared-math behavior; no GPU or real questions."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

from train.readout import read_probabilities


def module():
    spec = importlib.util.spec_from_file_location(
        "comparison", Path(__file__).parents[1] / ".scratch/s4/compare-video-polarity-dev50.py"
    )
    code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code)
    return code


def records():
    rows = []
    for i in range(50):
        raw = {
            "yes": math.log(0.8 if i % 2 == 0 else 0.1),
            "no": math.log(0.1 if i % 2 == 0 else 0.8),
            "U": math.log(0.1),
        }
        value = read_probabilities(raw, ["yes", "no"])
        rows.append(
            {
                "id": f"id{i}",
                "family_id": f"f{i}",
                "question_id": "q",
                "question_type": "noul",
                "target_index": i % 2,
                "target_source": "gold",
                "modalities": ["video"],
                "prompt_sha256": "a" * 64,
                "payload_sha256": "b" * 64,
                "input_sha256": "c" * 64,
                "input_summary_sha256": "d" * 64,
                "answer_token_ids": {"yes": 1, "no": 2, "U": 3},
                "frame_indices": list(range(16)),
                "sampling_policy": "uniform",
                "pass_index": 1,
                "raw_logprobs": raw,
                "prediction": value.prediction,
                "p_U": value.p_insufficient,
                "conditional_without_u": value.probabilities,
                "joint_k_plus_u": value.joint_probabilities,
            }
        )
    return rows


@pytest.mark.parametrize("bad", [None, "input", "target", "mapping", "readout"])
def test_strict_sameinput_and_actual_readout(bad):
    code = module()
    a = records()
    b = copy.deepcopy(a)
    if bad == "input":
        b[0]["input_sha256"] = "0" * 64
    elif bad == "target":
        b[0]["target_index"] = 1
    elif bad == "mapping":
        b[0]["answer_token_ids"]["yes"] = 99
    elif bad == "readout":
        b[0]["conditional_without_u"]["yes"] = 0.6
    if bad:
        with pytest.raises(ValueError):
            code.paired_identity(a, b)
    else:
        code.paired_identity(a, b)


def test_shared_metrics_auc_and_bidirectional_change():
    code = module()
    rows = records()
    ranking = code.module("diagnose-video-ranking")
    result = code.summarize(rows, ranking.auc)
    assert result["correct"] == 50 and result["temporal_heuristic_p_yes_auc"] == 1
    assert result["prediction_counts"] == {"yes": 25, "no": 25}
    assert result["joint_k_plus_u_nll"] > result["conditional_nll"]
    other = copy.deepcopy(rows)
    other[0]["prediction"] = "no"
    assert code.paired_changes(rows, other) == {
        "corrected": 0,
        "regressed": 1,
        "changed_predictions": 1,
    }
    assert code.paired_changes(other, rows) == {
        "corrected": 1,
        "regressed": 0,
        "changed_predictions": 1,
    }


def test_matched_training_fairness_rejects_replay_or_inverse_mutation():
    from test_video_polarity import module as builder_module
    from test_video_polarity import video

    builder = builder_module()
    code = module()
    h = builder.helper()
    desired = {arm: [] for arm in ["control", "polarity"]}
    ledgers = {arm: [] for arm in desired}
    for step in range(10):
        for n in range(step * 3, step * 3 + 3):
            source = video(n, "yes" if n % 2 else "no")
            source.prefix.media[0].sha256 = f"{n:064x}"
            for arm in desired:
                for s in builder.variants(source, arm):
                    desired[arm].append(s)
                    ledgers[arm].append(
                        {
                            "item_id": s.id,
                            "parent_sample_sha256": builder.canonical_sample_sha(source),
                        }
                    )
        replay = [video(100 + step * 6 + i) for i in range(6)]
        for arm in desired:
            desired[arm].extend(copy.deepcopy(replay))
    arranged = {arm: h.inverse_order(rows) for arm, rows in desired.items()}
    # Replay synthetic rows carry text modality to distinguish them from the video treatment.
    for arm in arranged:
        for s in arranged[arm]:
            if s.id.startswith("v1"):
                s.meta.modality = ["text"]
    code.training_fairness(
        arranged["control"], arranged["polarity"], ledgers["control"], ledgers["polarity"]
    )
    changed = copy.deepcopy(arranged["polarity"])
    changed[0].prefix.state = "changed"
    with pytest.raises(ValueError):
        code.training_fairness(
            arranged["control"], changed, ledgers["control"], ledgers["polarity"]
        )
