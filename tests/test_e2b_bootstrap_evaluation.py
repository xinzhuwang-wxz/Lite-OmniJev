"""CPU guard/selection tests for independent paired E2B evaluation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[1] / ".scratch/s6/evaluate-e2b-bootstrap.py"
    spec = importlib.util.spec_from_file_location("e2b_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trained(helper):
    return {
        "status": "paused",
        "exit_code": 0,
        "steps_completed": 10,
        "adapter_sha256": helper.ADAPTER_SHA,
        "model_revision": helper.MODEL.name,
        "data_sha256": "795f69fe9474523009bd953a0436bc505e85efdd4fb0f5e7c41e4d86c3bb028f",
    }


def test_fixed_independent_e2b_training_binding(helper):
    helper.validate_training(trained(helper))


@pytest.mark.parametrize(
    "changes",
    [
        {"model_revision": "e4b"},
        {"init_adapter": {}},
        {"adapter_sha256": "wrong"},
        {"steps_completed": 20},
        {"exit_code": 1},
        {"data_sha256": "wrong"},
    ],
)
def test_reject_other_model_inherited_adapter_or_changed_run(helper, changes):
    with pytest.raises(ValueError):
        helper.validate_training({**trained(helper), **changes})


def test_video_selection_uses_existing_fixed_motion_rule(helper, monkeypatch):
    monkeypatch.setattr(helper.baseline, "validate_samples", lambda samples, cohort: None)
    rows = [
        SimpleNamespace(id=f"video-{i}", questions=[SimpleNamespace(instruction="before motion")])
        for i in range(500)
    ]
    selected = helper.select_samples(rows, "video")
    expected = sorted(
        rows,
        key=lambda s: helper.hashlib.sha256(("42:s4-motion-coverage:" + s.id).encode()).hexdigest(),
    )[:50]
    assert selected == expected
    assert helper.select_samples(list(reversed(rows)), "video") == expected
    assert helper.select_samples(rows, "text") == rows[:50]
