"""Global fixed-pool order must survive deterministic shuffle without changing pair adjacency."""

import importlib.util
import random
from pathlib import Path

import pytest
from test_video_text_matched_pilot import pair

path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-continuous-matched-pool.py"
spec = importlib.util.spec_from_file_location("continuous_matched", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_global_inverse_shuffle_and_first_invocation_prefix():
    desired = list(range(1200))
    stored = module.inverse_order(desired)
    order = list(range(1200))
    random.Random(42).shuffle(order)
    actual = [stored[i] for i in order]
    assert actual == desired
    assert actual[:120] == desired[:120]
    assert actual[120:240] == desired[120:240]
    with pytest.raises(ValueError, match="1200"):
        module.inverse_order(desired[:120])


def test_select_three_hundred_complete_opposite_gold_pairs():
    groups = {}
    for i in range(300):
        members = pair()
        for j, r in enumerate(members):
            r.id = f"{i}:{j}"
            r.meta.family_id = f"family:{i}:{j}"
        groups[(str(i),)] = members
    selected = module.choose_pairs(groups)
    assert len(selected) == 300
    assert len({r.meta.family_id for p in selected for r in p}) == 600
    assert all({r.questions[0].target.value for r in p} == {"yes", "no"} for p in selected)
