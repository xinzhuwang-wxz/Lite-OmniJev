"""Gold-derived polarity is deterministic TRAIN-only supervision, never a U label."""

import copy
import importlib.util
from pathlib import Path

import pytest
from test_gold_inventory import sample

from data.schema import Sample, answer_tokens, validate_pool_isolation


def module():
    spec = importlib.util.spec_from_file_location(
        "polarity", Path(__file__).parents[1] / ".scratch/s4/prepare-video-polarity-pilot.py"
    )
    code = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(code)
    return code


def video(index, label="yes"):
    r = sample(f"v{index}", pool="train", modality="video")
    r["meta"]["family_id"] = f"family:{index}"
    r["questions"][0]["type"] = "noul"
    r["questions"][0]["target"] = {"kind": "hard", "value": label, "source": "official_gold"}
    return Sample.model_validate(r)


@pytest.mark.parametrize("gold", ["yes", "no"])
def test_original_control_and_inverse_preserve_media_gold_provenance(gold):
    code = module()
    original = video(1, gold)
    before = copy.deepcopy(original)
    c0, c1 = code.variants(original, "control")
    p0, p1 = code.variants(original, "polarity")
    assert original == before
    assert c0.questions == c1.questions == p0.questions == original.questions
    assert p1.questions[0].instruction == code.WRAPPER + original.questions[0].instruction
    assert p1.questions[0].target.value == ("yes" if gold == "no" else "no")
    assert answer_tokens(p1.questions[0]) == ("yes", "no")  # U remains separate, not supervised.
    assert len({s.id for s in [c0, c1, p0, p1]}) == 4
    for s in [c0, c1, p0, p1]:
        assert s.prefix == original.prefix and s.meta.family_id == original.meta.family_id
        assert s.meta.source_sha256 == original.meta.source_sha256 and s.meta.pool == "train"
    validate_pool_isolation([c0, c1, p0, p1])
    sealed = p1.model_copy(deep=True)
    sealed.meta.pool = "holdout"
    with pytest.raises(ValueError, match="crosses pools"):
        validate_pool_isolation([p0, sealed])


@pytest.mark.parametrize("bad", ["insufficient", "dev", "choice", "media"])
def test_reject_noncanonical_gold_and_nontrain(bad):
    code = module()
    r = video(1).model_dump(mode="json")
    if bad == "insufficient":
        r["questions"][0]["target"] = {"kind": "insufficient"}
    elif bad == "dev":
        r["meta"]["pool"] = "dev"
    elif bad == "choice":
        r["questions"][0].update(
            type="choice",
            options=["a", "b"],
            target={"kind": "hard", "value": "A", "source": "gold"},
        )
    else:
        r["prefix"]["media"] = []
    with pytest.raises(ValueError):
        code.variants(Sample.model_validate(r), "polarity")


def test_hash_selection_balanced_unique_family_order_independent():
    code = module()
    rows = [video(i, "yes" if i % 2 else "no") for i in range(80)]
    selected = code.select_families(rows)
    assert selected == code.select_families(list(reversed(rows)))
    assert len(selected) == len({s.meta.family_id for s in selected}) == 30
    assert sum(code.video_gold(s) == "yes" for s in selected) == 15
    assert all(len(selected[i : i + 3]) == 3 for i in range(0, 30, 3))
    with pytest.raises(ValueError, match="capacity"):
        code.select_families(rows[:20])


def test_source_ledger_rejects_changed_gold_binding():
    code = module()
    h = code.helper()
    source = video(1)
    ledger = [
        {
            "item_id": source.id,
            "pool": "train",
            "derived_sample_sha256": code.canonical_sample_sha(source),
        }
    ]
    assert h.exact_ledger([source], ledger) == ledger
    changed = source.model_copy(deep=True)
    changed.questions[0].target.value = "no"
    with pytest.raises(ValueError, match="exact_source_ledger"):
        h.exact_ledger([changed], ledger)


def test_doubled_draws_match_actual_seed42_order():
    code = module()
    h = code.helper()
    sources = [video(i, "yes" if i % 2 else "no") for i in range(30)]
    control = [s for original in sources for s in code.variants(original, "control")]
    polarity = [s for original in sources for s in code.variants(original, "polarity")]
    # inverse_order's contract is exactly120; append identical replay identities.
    replay = [video(100 + i) for i in range(60)]
    import random

    order = list(range(120))
    random.Random(42).shuffle(order)
    for desired in [control + replay, polarity + replay]:
        arranged = h.inverse_order(desired)
        assert [arranged[i] for i in order] == desired
    assert {s.prefix.media[0].sha256 for s in control} == {
        s.prefix.media[0].sha256 for s in polarity
    }
