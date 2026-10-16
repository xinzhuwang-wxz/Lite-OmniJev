"""Public CPU behavior of the fixed continuous TRAIN draw pool."""

import importlib.util
import random
from collections import Counter
from pathlib import Path

import pytest
from test_temporal_curriculum_preparation import row

path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-continuous-main-pool.py"
spec = importlib.util.spec_from_file_location("continuous_pool", path)
assert spec and spec.loader
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def source():
    return [
        row(f"{task}-{target}-{n}", task, target)
        for task in ("temporal", "motion", "existence")
        for target in ("yes", "no")
        for n in range(310)
    ]


def test_fixed_pool_actual_optimizer_blocks_and_inverse_shuffle():
    selected = m.select_video(source(), set())
    old = {c: [row(f"{c}-{n}", modality="text") for n in range(100)] for c in m.OLD}
    cross = [row(f"cross-{n}", modality="text") for n in range(20)]
    draws = m.layout(selected, cross, old)
    arranged, order = m.inverse_order(draws)
    indices = list(range(1200))
    random.Random(42).shuffle(indices)
    assert indices == order
    actual = [arranged[i] for i in indices]
    assert len(actual) == 1200
    assert len({s.meta.family_id for i, s in enumerate(actual) if i % 12 < 6}) == 600
    assert Counter(m.kind(s) for i, s in enumerate(actual) if i % 12 < 6) == {
        "temporal": 500,
        "motion": 50,
        "existence": 50,
    }
    for step in range(100):
        block = actual[12 * step : 12 * step + 12]
        assert Counter(s.questions[0].target.value for s in block[:6]) == {"yes": 3, "no": 3}
        assert block[6:8] == cross[2 * (step % 10) : 2 * (step % 10) + 2]
        assert block[8:] == [old[c][step] for c in m.OLD]


def test_hash_selection_independent_of_input_order_and_excludes_lineage():
    original = source()
    excluded = {s.meta.family_id for s in original[:20]}
    selected = m.select_video(original, excluded)
    assert selected == m.select_video(list(reversed(original)), excluded)
    assert not excluded & {s.meta.family_id for bucket in selected.values() for s in bucket}
    assert {k: len(v) for k, v in selected.items()} == {
        (task, t): 250 if task == "temporal" else 25
        for task in ("temporal", "motion", "existence")
        for t in ("yes", "no")
    }


def test_insufficient_label_family_capacity_and_nontrain_rejected():
    with pytest.raises(ValueError, match="video_capacity"):
        m.select_video(source()[:10], set())
    values = source()
    values[0] = values[0].model_copy(
        update={"meta": values[0].meta.model_copy(update={"pool": "dev"})}
    )
    with pytest.raises(ValueError, match="video_train_gold"):
        m.select_video(values, set())
    with pytest.raises(ValueError, match="old_capacity"):
        m.unique_families([row("one", modality="text")], 100, "text")


def test_alias_retains_gold_family_media_source_and_exact_canonical_binding():
    s = row("original")
    h = m.helper("prepare-claim-pair-pilot.py")
    ledger = {
        "item_id": s.id,
        "pool": "train",
        "derived_sample_sha256": h.canonical_sample_sha(s),
        "license": "CC0-1.0",
        "family_id": s.meta.family_id,
    }
    a, record = m.alias_draw(s, 0, ledger)
    b, _ = m.alias_draw(s, 1, ledger)
    assert a.id != b.id != s.id
    assert a.meta == s.meta and a.questions == s.questions and a.prefix == s.prefix
    assert record["source_item_id"] == s.id
    assert record["source_sample_sha256"] == h.canonical_sample_sha(s)
    assert record["derived_sample_sha256"] == h.canonical_sample_sha(a)
    with pytest.raises(ValueError, match="source_ledger"):
        m.alias_draw(s, 2, {**ledger, "derived_sample_sha256": "0" * 64})
