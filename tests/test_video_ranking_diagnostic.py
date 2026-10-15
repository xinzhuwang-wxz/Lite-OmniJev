"""Public math behavior for tie-correct AUC and endpoint-complete oracle fits."""

import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / ".scratch/s4/diagnose-video-ranking.py"
spec = importlib.util.spec_from_file_location("video_ranking", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    ("scores", "labels", "expected"),
    [
        ([0.5, 0.5, 0.5, 0.5], [True, False, True, False], 0.5),
        ([0.1, 0.2, 0.8, 0.9], [False, False, True, True], 1),
        ([0.1, 0.2, 0.8, 0.9], [True, True, False, False], 0),
        ([0.1, 0.5, 0.5, 0.9], [False, False, True, True], 0.875),
    ],
)
def test_auc(scores, labels, expected):
    assert module.auc(scores, labels) == expected


@pytest.mark.parametrize(
    ("scores", "labels", "best"),
    [
        ([0.1, 0.9], [False, False], 2),
        ([0.1, 0.9], [True, True], 2),
        ([0.5, 0.5], [True, False], 1),
        ([0.1, 0.9], [False, True], 2),
        ([0.1, 0.9], [True, False], 1),
    ],
)
def test_oracle_endpoints_and_unsplit_ties(scores, labels, best):
    assert module.oracle(scores, labels)["best_correct"] == best
    assert module.oracle(scores, labels)["threshold_exported_or_adopted"] is False


def test_single_class_auc_undefined_and_invalid_inputs_rejected():
    assert module.auc([0.2, 0.3], [True, True]) is None
    with pytest.raises(ValueError):
        module.auc([float("nan")], [True])
    with pytest.raises(ValueError):
        module.oracle([0.2], [True, False])
