"""Controlled event-order gold is evaluated, never inferred from old negative labels."""

import importlib.util
from pathlib import Path

import pytest
from test_gold_inventory import sample

from data.schema import Sample

path = Path(__file__).resolve().parents[1] / ".scratch/s4/prepare-temporal-order-contrast.py"
spec = importlib.util.spec_from_file_location("order_prepare", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def annotation():
    return {
        "object_property": [
            {"object_id": i, "color": color, "material": "metal", "shape": "sphere"}
            for i, color in enumerate(["red", "blue", "green"])
        ],
        "motion_trajectory": [
            {
                "frame_id": f,
                "objects": [{"object_id": i, "inside_camera_view": True} for i in range(3)],
            }
            for f in range(128)
        ],
        "collision": [
            {"object_ids": [0, 1], "frame_id": 10},
            {"object_ids": [1, 2], "frame_id": 30},
        ],
    }


@pytest.mark.parametrize("failure", ["missing", "repeat", "tie", "close", "invisible", "ambiguous"])
def test_reject_unsafe_event_semantics(failure):
    a = annotation()
    if failure == "missing":
        a["collision"] = a["collision"][:1]
    elif failure == "repeat":
        a["collision"].append(a["collision"][0])
    elif failure == "tie":
        a["collision"][1]["frame_id"] = 10
    elif failure == "close":
        a["collision"][1]["frame_id"] = 25
    elif failure == "invisible":
        a["motion_trajectory"][10]["objects"][0]["inside_camera_view"] = False
    else:
        a["object_property"][1]["color"] = "red"
    assert module.safe_pairs(a) == []


def test_gap16_allowed_and_derived_both_truths_ignore_old_answer():
    a = annotation()
    a["collision"][1]["frame_id"] = 26
    choices = module.safe_pairs(a)
    assert len(choices) == 1
    value = sample("clevrer-train-00001-5", pool="train", modality="video")
    value["meta"]["family_id"] = "clevrer:1"
    value["questions"][0]["target"]["value"] = "no"
    source = Sample.model_validate(value)
    variants = module.make_pair(source, choices[0], "a" * 64, "annotation_00001.json", "b" * 64)
    assert {s.questions[0].target.value for s, _ in variants} == {"yes", "no"}
    assert len({s.id for s, _ in variants}) == 2
    for s, ledger in variants:
        program = ledger["program_provenance"]["program"]
        expected = "yes" if program["events"][0]["frame"] < program["events"][1]["frame"] else "no"
        assert s.questions[0].target.value == expected
        assert s.prefix == source.prefix and s.meta.family_id == source.meta.family_id
        assert ledger["derived_sample_sha256"] == module.canonical_sample_sha(s)
        text = s.questions[0].instruction
        assert "object_id" not in text and "frame" not in text
        assert not any(char.isdigit() for char in text)
    assert variants == module.make_pair(
        source, choices[0], "a" * 64, "annotation_00001.json", "b" * 64
    )


def test_nontrain_pair_rejected():
    value = sample("clevrer-train-00001-5", pool="dev", modality="video")
    with pytest.raises(ValueError, match="TRAIN"):
        module.make_pair(
            Sample.model_validate(value),
            module.safe_pairs(annotation())[0],
            "a" * 64,
            "annotation.json",
            "b" * 64,
        )
