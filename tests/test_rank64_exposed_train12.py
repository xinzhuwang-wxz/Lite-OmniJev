"""TRAIN diagnostics must select only pairs actually consumed by this invocation."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / ".scratch/s4/evaluate-rank64-exposed-train12.py"
spec = importlib.util.spec_from_file_location("rank64_exposed", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_select_only_thirty_pairs_in_actual_first120draws():
    samples = [SimpleNamespace(id=str(i), meta=SimpleNamespace(pool="train")) for i in range(1200)]
    order = list(range(1200))
    random.Random(42).shuffle(order)
    actual = [samples[i] for i in order]
    recipe = {"draws": [{"item_id": s.id} for s in actual]}
    metadata = {
        "rows": [
            {
                "pair": step * 3 + offset // 2,
                "members": [{"item_id": actual[step * 12 + offset + j].id} for j in [0, 1]],
            }
            for step in range(100)
            for offset in [0, 2, 4]
        ]
    }
    exposed = module.exposed_pairs(samples, metadata, recipe, 42)
    assert len(exposed["rows"]) == 30
    assert {p["pair"] for p in exposed["rows"]} == set(range(30))
    recipe["draws"][0]["item_id"] = "wrong"
    with pytest.raises(ValueError, match="order"):
        module.exposed_pairs(samples, metadata, recipe, 42)


def test_logodds_direction_is_distinct_from_both_correct():
    pairs = []
    rows = []
    for i in range(6):
        ids = [f"{i}:yes", f"{i}:no"]
        pairs.append(ids)
        for identity, gold, odds in zip(ids, ["yes", "no"], [1.0, 0.5], strict=True):
            rows.append(
                {
                    "id": identity,
                    "target": gold,
                    "prediction": "yes",
                    "probabilities": {
                        "yes": 0.7 if gold == "yes" else 0.6,
                        "no": 0.3 if gold == "yes" else 0.4,
                    },
                    "joint_ce": 1.0,
                    "gold_margin": odds if gold == "yes" else -odds,
                    "logodds": odds,
                }
            )
    result = module.metrics(rows, pairs)
    assert result["pair_logodds_direction"] == 6
    assert result["pair_both_correct"] == 0
    assert result["correct"] == 6


def test_only_verified_initializer_metadata_is_removed():
    before = {"path": "frozen", "weights_sha256": "a" * 64, "config_sha256": "b" * 64}
    trained = {
        "config": {"sample_count": 1200, "rank": 64, "alpha": 128, "init_adapter": before},
        "init_adapter": before,
    }
    value = {**trained["config"], "sample_count": 12}
    normalized = module.normalized_loop_fields(value, trained, before)
    assert normalized == {"sample_count": 12, "rank": 64, "alpha": 128}
    assert value["init_adapter"] == before
    with pytest.raises(ValueError, match="subset"):
        module.normalized_loop_fields({**value, "rank": 16}, trained, before)
    with pytest.raises(ValueError, match="initializer"):
        module.normalized_loop_fields(value, trained, {**before, "path": "wrong"})
