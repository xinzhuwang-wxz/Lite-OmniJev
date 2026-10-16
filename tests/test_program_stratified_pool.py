"""Official-program priority, strict capacity and actual balanced prefix allocation."""

import collections
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-program-stratified-pool.py"
spec = importlib.util.spec_from_file_location("stratified_course", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def buckets():
    return {
        g: {
            gold: [
                SimpleNamespace(
                    id=f"{g}:{gold}:{i}",
                    meta=SimpleNamespace(family_id=f"{g}:{gold}:{i}"),
                    questions=[SimpleNamespace(target=SimpleNamespace(value=gold))],
                )
                for i in range(75)
            ]
            for gold in ["yes", "no"]
        }
        for g in module.GROUPS
    }


def test_official_program_priority_keeps_collision_relations():
    assert (
        module.program_group(["events", "filter_collision", "filter_after", "get_frame", "exist"])
        == "before_after"
    )
    assert module.program_group(["events", "get_frame", "exist"]) == "event_motion"
    assert module.program_group(["get_frame", "exist"]) == "endpoint_motion"
    assert module.program_group(["objects", "exist"]) == "exist"
    with pytest.raises(ValueError, match="existential"):
        module.program_group(["objects", "count"])


def test_full_strata_and_prefix_balance_without_repetition():
    chosen, layout = module.select_layout(buckets())
    assert len(chosen) == len({s.meta.family_id for s in chosen}) == 600
    for step, rows in enumerate(layout):
        counts = collections.Counter(r.id.split(":")[0] for r in rows)
        assert counts == dict(
            zip(module.GROUPS, (2, 2, 1, 1) if step % 2 == 0 else (1, 1, 2, 2), strict=True)
        )
        assert collections.Counter(r.questions[0].target.value for r in rows) == {"yes": 3, "no": 3}
    for steps, count in [(10, 15), (20, 30), (100, 150)]:
        rows = [r for group in layout[:steps] for r in group]
        assert collections.Counter(r.id.split(":")[0] for r in rows) == {
            g: count for g in module.GROUPS
        }
    data = buckets()
    data["exist"]["no"].pop()
    with pytest.raises(ValueError, match="insufficient"):
        module.select_layout(data)
