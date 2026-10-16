"""Exact text/program pairs keep official opposite gold and actual draw adjacency."""

import copy
import hashlib
import json

import pytest
from test_reconcile_crossmodal_retention import module
from test_reconcile_native_event_condition import contracts
from test_temporal_curriculum_preparation import row

from scripts.s1_text_pools import canonical_sample_sha


def pair(code):
    left, right = row("left", target="yes"), row("right", target="no")
    right.prefix.media[0].sha256 = "a" * 64
    program = [
        "events",
        "events",
        "red",
        "filter_color",
        "unique",
        "filter_in",
        "unique",
        "filter_before",
        "blue",
        "filter_color",
        "filter_out",
        "exist",
    ]
    metadata = {
        "metadata_only": True,
        "model_nonmedia_question_sha256": code.matched_nonmedia_sha(left),
        "official_program_sha256": hashlib.sha256(
            json.dumps(program, sort_keys=True).encode()
        ).hexdigest(),
        "text_sha256": hashlib.sha256(left.questions[0].instruction.encode()).hexdigest(),
        "members": [
            {
                "item_id": s.id,
                "family_id": s.meta.family_id,
                "sample_sha256": canonical_sample_sha(s),
                "official_source_sha256": s.meta.source_sha256,
                "media_sha256": s.prefix.media[0].sha256,
                "original_gold": s.questions[0].target.value,
            }
            for s in (left, right)
        ],
    }
    return left, right, program, metadata


@pytest.mark.parametrize(
    "bad", [None, "text", "state", "qid", "program", "gold", "media", "binding"]
)
def test_matched_identity_rejects_every_nonmedia_or_truth_drift(bad):
    code = module()
    left, right, program, metadata = pair(code)
    other = copy.deepcopy(program)
    if bad == "text":
        right.questions[0].instruction += " changed"
    elif bad == "state":
        right.prefix.state += " changed"
    elif bad == "qid":
        right.questions[0].qid = "changed"
    elif bad == "program":
        other[-2] = "filter_in"
    elif bad == "gold":
        right.questions[0].target.value = "yes"
    elif bad == "media":
        right.prefix.media[0].sha256 = left.prefix.media[0].sha256
    elif bad == "binding":
        metadata["members"][1]["sample_sha256"] = "0" * 64
    if bad:
        with pytest.raises(ValueError):
            code.matched_pair_binding(left, right, program, other, metadata)
    else:
        code.matched_pair_binding(left, right, program, other, metadata)


def test_matched_metadata_only_ce_contract_and_own_path(tmp_path):
    code = module()
    prep, launch, final = contracts(code)
    prep.update(
        pair_count=30,
        pair_metadata_only=True,
        nonmedia_prefix_question_equal=True,
        actual_draw_pair_offsets=[0, 2, 4],
    )
    code.native_parent_binding(prep, launch, final, text_matched=True)
    prep["actual_draw_pair_offsets"] = [1, 3, 5]
    with pytest.raises(ValueError, match="matched_preparation_contract"):
        code.native_parent_binding(prep, launch, final, text_matched=True)
    paths = code.pilot_paths(tmp_path, video_text_matched=True)
    assert paths == [
        (
            "s4-video-text-matched",
            tmp_path / "reports/s4-video-text-matched-training-001.json",
            tmp_path / "artifacts/s4-video-text-matched-train-001",
            tmp_path / "artifacts/s4-video-text-matched-pilot-001",
        )
    ]


def test_matched_base_uses017_retry_and_validates_old_snapshot(tmp_path, monkeypatch):
    code = module()
    base = tmp_path / "artifacts/s4-pilot-exposure-017-retry-001"
    base.mkdir(parents=True)
    old = tmp_path / "old-snapshot.json"
    old.write_text("{}")
    inv = base / "inventory-manifest.json"
    inv.write_text("{}")
    report = base / "report.json"
    report.write_text(
        json.dumps(
            {
                "exit_code": 0,
                "unknown_update_bearing_inputs": 0,
                "all_bound_input_sha256": {str(old): code.sha(old)},
                "output_sha256": {"inventory-manifest.json": code.sha(inv)},
            }
        )
    )
    monkeypatch.setattr(code, "MATCHED_BASE_SHA", code.sha(report))
    path, _ = code.base_inventory(tmp_path, crossmodal_retention=False, video_text_matched=True)
    assert path == inv
    old.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="native_prior_bindings_mutated"):
        code.base_inventory(tmp_path, crossmodal_retention=False, video_text_matched=True)
