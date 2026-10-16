"""Time coverage and conservative camera-transition proxy, no pixel-visibility claims."""

import importlib.util
from pathlib import Path

import pytest

p = Path(__file__).resolve().parents[1] / ".scratch/s4/diagnose-temporal-observability.py"
spec = importlib.util.spec_from_file_location("observability", p)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
INDICES = [round(i * 127 / 15) for i in range(16)]


def annotation():
    return {
        "object_property": [
            {"object_id": 0, "color": "red", "material": "metal", "shape": "sphere"},
            {"object_id": 1, "color": "blue", "material": "rubber", "shape": "cube"},
        ],
        "motion_trajectory": [
            {
                "frame_id": f,
                "objects": [
                    {"object_id": 0, "inside_camera_view": 10 <= f < 80},
                    {"object_id": 1, "inside_camera_view": 30 <= f < 100},
                ],
            }
            for f in range(128)
        ],
        "collision": [],
    }


def program(relation):
    return [
        "events",
        "events",
        "objects",
        "red",
        "filter_color",
        "unique",
        "filter_in",
        "unique",
        relation,
        "objects",
        "blue",
        "filter_color",
        "filter_in",
        "exist",
    ]


def test_exact_endpoint_and_bracket_distance_are_different_claims():
    assert module.frame_coverage(0, INDICES)["exact_sample"]
    covered = module.frame_coverage(10, INDICES)
    assert covered["nearest_index"] == 8 and covered["nearest_distance"] == 2
    assert covered["bracket_width"] == 9 and not covered["exact_sample"]
    with pytest.raises(ValueError):
        module.frame_coverage(128, INDICES)
    with pytest.raises(ValueError):
        module.frame_coverage(10, [0, 127])


def test_camera_transition_is_explicit_and_initial_visibility_is_not_an_entry():
    a = annotation()
    assert module.transitions(a, 0, "filter_in") == [10]
    assert module.transitions(a, 0, "filter_out") == [80]
    for frame in a["motion_trajectory"]:
        frame["objects"][0]["inside_camera_view"] = True
    assert module.transitions(a, 0, "filter_in") == []
    assert not module.diagnose(program("filter_after"), a, INDICES, "yes")["anchor_resolved"]


@pytest.mark.parametrize("relation,gold", [("filter_before", "no"), ("filter_after", "yes")])
def test_subset_order_proxy_checks_official_gold_without_relabeling(relation, gold):
    result = module.diagnose(program(relation), annotation(), INDICES, gold)
    order = result["critical_order_proxy"]
    assert order["gold_agrees_with_transition_proxy"]
    assert order["gap_frames"] == 20 and not order["same_nearest_sample"]
    mismatched = module.diagnose(
        program(relation), annotation(), INDICES, "no" if gold == "yes" else "yes"
    )
    assert not mismatched["critical_order_proxy"]["gold_agrees_with_transition_proxy"]


def test_repeated_anchor_is_unresolved_not_arbitrarily_selected():
    a = annotation()
    for frame in a["motion_trajectory"]:
        if 40 <= frame["frame_id"] < 50:
            frame["objects"][0]["inside_camera_view"] = False
    result = module.diagnose(program("filter_after"), a, INDICES, "yes")
    assert not result["anchor_resolved"] and "repeated" in result["unresolved_reason"]
