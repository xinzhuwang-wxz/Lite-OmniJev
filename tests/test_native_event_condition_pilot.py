"""Native original gold remains authoritative over sampling proxies."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from test_temporal_order_contrast import annotation

path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-native-event-condition-pilot.py"
spec = importlib.util.spec_from_file_location("native_prepare", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def program():
    return [
        "events",
        "events",
        "objects",
        "red",
        "filter_color",
        "unique",
        "filter_in",
        "unique",
        "filter_after",
        "objects",
        "blue",
        "filter_color",
        "filter_out",
        "exist",
    ]


def test_native_grammar_rejects_collision_and_extra_suffix():
    assert module.native_program(program())
    assert not module.native_program(program() + ["not"])
    p = program()
    p[-2] = "filter_collision"
    assert not module.native_program(p)


def test_proxy_absence_never_flips_official_gold():
    a = annotation()
    for frame in a["motion_trajectory"]:
        frame["objects"][0]["inside_camera_view"] = frame["frame_id"] >= 20
    result = module.proxy_metadata(program(), a, "yes")
    assert result["branch"] == "target_transition_absent"
    assert result["gold_agrees_with_proxy"] is False
    assert result["official_target_unchanged"] == "yes"


def test_initial_visible_anchor_is_unknown_not_invented_entry():
    result = module.proxy_metadata(program(), annotation(), "no")
    assert result["branch"] == "unknown"
    assert result["gold_agrees_with_proxy"] is None


def test_actual_exposure_uses_only_completed_draws():
    rows = [
        SimpleNamespace(meta=SimpleNamespace(family_id=str(i), modality=["video"]))
        for i in range(1200)
    ]
    families = module.actual_video_families(rows, 42, 20, 12)
    assert len(families) == 240
    assert len(module.actual_video_families(rows, 42, 100, 12)) == 1200
