"""Single-arm launch contract; CPU tests never execute subprocesses or read keys."""

import copy
import importlib.util
from pathlib import Path

import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "launch", Path(__file__).parents[1] / ".scratch/s4/launch-video-polarity-pilot.py"
    )
    code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code)
    return code


def inputs(code, arm):
    report = {
        "status": "cpu_prepared_pending_root_training_decision",
        "exit_code": 0,
        "training_authorized": False,
        "families": 30,
        "arms": {
            arm: {
                "inventory_isolation": "isolated",
                "output_sha256": {"train.jsonl": code.DATA[arm]},
                "ledger_check": {"verified_train": 120},
            }
        },
    }
    config = {
        "sample_count": 120,
        "steps": 10,
        "accumulation": 12,
        "seed": 42,
        "midway_step": 5,
        "checkpoint_every": 5,
        "dtype": "bfloat16",
        "video_num_frames": 16,
        "image_max_soft_tokens": 1120,
        "rank": 16,
        "alpha": 32,
    }
    return report, config


@pytest.mark.parametrize("arm", ["control", "polarity"])
def test_same_parent_and_fixed_contract(arm):
    code = module()
    r, c = inputs(code, arm)
    before = copy.deepcopy(r)
    code.arm_contract(arm, r, c)
    assert r == before
    paths = code.paths(arm)
    assert f"s4-video-polarity-{arm}-train-001" in str(paths["train"])
    assert code.PARENT_SHA == "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"


@pytest.mark.parametrize(
    "bad", ["arm", "data", "isolation", "steps", "draws", "objective", "midway", "authorized"]
)
def test_reject_wrong_arm_or_changed_budget(bad):
    code = module()
    r, c = inputs(code, "control")
    arm = "control"
    if bad == "arm":
        arm = "both"
    elif bad == "data":
        r["arms"][arm]["output_sha256"]["train.jsonl"] = code.DATA["polarity"]
    elif bad == "isolation":
        r["arms"][arm]["inventory_isolation"] = "pending"
    elif bad == "steps":
        c["steps"] = 11
    elif bad == "draws":
        c["sample_count"] = 121
    elif bad == "objective":
        c["brier_weight"] = 0.5
    elif bad == "midway":
        c["midway_step"] = 4
    else:
        r["training_authorized"] = True
    with pytest.raises(ValueError):
        code.arm_contract(arm, r, c)


def test_execute_refuses_existing_output_before_subprocess(tmp_path, monkeypatch):
    code = module()
    paths = {
        k: tmp_path / k
        for k in ["train", "training", "preflight", "launch", "dev", "dev_report", "prep"]
    }
    paths["training"].write_text("existing")
    monkeypatch.setattr(code, "paths", lambda arm: paths)
    monkeypatch.setattr(code.subprocess, "run", lambda *a, **k: pytest.fail("must not execute"))
    with pytest.raises(ValueError, match="refuse_overwrite"):
        code.execute("control", {})
    assert paths["training"].read_text() == "existing"
