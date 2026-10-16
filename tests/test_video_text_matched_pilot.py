"""Matched-pair course must preserve the model-visible nonmedia question contract."""

import importlib.util
from pathlib import Path

import pytest
from test_gold_inventory import sample

from data.schema import Sample

path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-video-text-matched-pilot.py"
spec = importlib.util.spec_from_file_location("matched_course", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pair():
    values = []
    for i, gold in enumerate(["yes", "no"]):
        value = sample(str(i), pool="train", modality="video")
        value["meta"]["family_id"] = f"family:{i}"
        value["questions"][0]["target"]["value"] = gold
        value["prefix"]["media"][0]["sha256"] = str(i) * 64
        values.append(Sample.model_validate(value))
    return values


def test_pair_ignores_only_media_and_gold_not_criteria_or_state():
    left, right = pair()
    module.validate_pair(left, right, ["events"], ["events"])
    right.prefix.state += " changed"
    with pytest.raises(ValueError, match="nonmedia"):
        module.validate_pair(left, right, ["events"], ["events"])


def test_reject_program_media_or_gold_mismatch():
    left, right = pair()
    with pytest.raises(ValueError, match="program"):
        module.validate_pair(left, right, ["before"], ["after"])
    right.prefix.media[0].sha256 = left.prefix.media[0].sha256
    with pytest.raises(ValueError, match="media"):
        module.validate_pair(left, right, [], [])


def test_inverse_shuffle_has_adjacent_opposite_gold_pairs():
    groups = {}
    for i in range(30):
        left, right = pair()
        for j, r in enumerate([left, right]):
            r.id = f"{i}:{j}"
            r.meta.family_id = f"family:{i}:{j}"
        groups[(str(i),)] = [left, right]
    pairs = module.choose_pairs(groups)
    assert len(pairs) == 30
    draws = [r for p in pairs for r in p]
    # Public inverse helper requires120; use60mock replay identities to prove real120order.
    desired = draws + [f"replay:{i}" for i in range(60)]
    arranged = module.h.inverse_order(desired)
    import random

    order = list(range(120))
    random.Random(42).shuffle(order)
    assert [arranged[i] for i in order] == desired
