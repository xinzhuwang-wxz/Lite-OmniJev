"""Synthetic TRAIN-only quota and actual draw-layout checks."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest
from test_gold_inventory import sample

from data.schema import Sample

path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-temporal-curriculum-pilot.py"
spec = importlib.util.spec_from_file_location("temporal_prepare", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(ident, task="temporal", target="yes", modality="video"):
    value = sample(ident, pool="train", modality=modality)
    value["meta"]["family_id"] = "family:" + ident
    value["questions"][0]["instruction"] = {
        "temporal": "before event",
        "motion": "moving object",
        "existence": "visible object",
    }[task]
    value["questions"][0]["target"]["value"] = target
    return Sample.model_validate(value)


def source():
    return [
        row(f"{task}-{target}-{n}", task, target)
        for task in ["temporal", "motion", "existence"]
        for target in ["yes", "no"]
        for n in range(30)
    ]


def test_fixed_final_quotas_unique_families_and_determinism():
    video = module.select_video(source())
    assert {k: len(v) for k, v in video.items()} == {
        ("temporal", "yes"): 25,
        ("temporal", "no"): 25,
        ("motion", "yes"): 2,
        ("motion", "no"): 3,
        ("existence", "yes"): 3,
        ("existence", "no"): 2,
    }
    assert module.select_video(list(reversed(source()))) == video
    assert len({s.meta.family_id for bucket in video.values() for s in bucket}) == 60
    with pytest.raises(ValueError, match="video_bucket_capacity"):
        module.select_video(source()[:10])


def test_every_step_five_temporal_one_alternating_minor_two_cross_four_old():
    video = module.select_video(source())
    pairs = [
        (row(f"cross-y-{n}", target="yes"), row(f"cross-n-{n}", target="no")) for n in range(10)
    ]
    old = {
        c: [row(f"{c}-{n}", modality="text") for n in range(10)]
        for c in ["audio", "image", "screenshot", "text"]
    }
    draws = module.desired_draws(video, pairs, old)
    targets = Counter()
    for step in range(10):
        group = draws[step * 12 : (step + 1) * 12]
        assert all(module.kind(s) == "temporal" for s in group[:5])
        assert Counter(s.questions[0].target.value for s in group[:5]) == (
            {"yes": 3, "no": 2} if step % 2 == 0 else {"yes": 2, "no": 3}
        )
        assert module.kind(group[5]) == ("motion" if step % 2 == 0 else "existence")
        assert group[6:8] == list(pairs[step])
        assert group[8:] == [old[c][step] for c in ["audio", "image", "screenshot", "text"]]
        targets.update(s.questions[0].target.value for s in group[:6])
    assert targets == {"yes": 30, "no": 30}
    assert len({s.id for s in draws}) == 120


def test_source_pool_and_family_reuse_fail_closed():
    values = source()
    payload = values[0].model_dump(mode="json")
    payload["meta"]["pool"] = "dev"
    with pytest.raises(ValueError, match="video_train_gold"):
        module.select_video([Sample.model_validate(payload), *values[1:]])
    same_family = [
        s.model_copy(update={"meta": s.meta.model_copy(update={"family_id": "same"})})
        for s in values
    ]
    with pytest.raises(ValueError, match="video_bucket_capacity"):
        module.select_video(same_family)


def test_explicit_new_namespace_excludes_prior_sixty_but_preserves_quotas():
    larger = [
        row(f"{task}-{target}-{n}", task, target)
        for task in ["temporal", "motion", "existence"]
        for target in ["yes", "no"]
        for n in range(80)
    ]
    prior = module.select_video(larger)
    excluded = frozenset(s.meta.family_id for bucket in prior.values() for s in bucket)
    namespace = "s4-temporal-coverage-segment001:42:"
    new = module.select_video(larger, hash_namespace=namespace, excluded_families=excluded)
    assert {key: len(bucket) for key, bucket in new.items()} == {
        key: len(bucket) for key, bucket in prior.items()
    }
    families = {s.meta.family_id for bucket in new.values() for s in bucket}
    assert len(families) == 60 and not families & excluded
    assert (
        module.select_video(
            list(reversed(larger)), hash_namespace=namespace, excluded_families=excluded
        )
        == new
    )
    # Defaults retain their original namespace and exact old selected source rows.
    assert module.select_video(larger) == prior
    with pytest.raises(ValueError, match="video_bucket_capacity"):
        module.select_video(larger, excluded_families=frozenset(s.meta.family_id for s in larger))
