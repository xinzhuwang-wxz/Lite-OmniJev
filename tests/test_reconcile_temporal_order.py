"""Program-gold exposure uses30families, never rawcoverage60-family constraints."""

import hashlib
import json

import pytest
from test_reconcile_crossmodal_retention import module
from test_video_polarity import video


def program_sample():
    sample = video(1, "yes")
    program = {
        "schema_version": "clevrer-collision-order-program-v1",
        "operation": "less_frame",
        "events": [{"frame": 10}, {"frame": 30}],
        "result": "yes",
    }
    p = {
        "original_postfix_execution": "not_executed_source_binding_only",
        "program": program,
        "program_sha256": hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest(),
    }
    sample.questions[0].target.source = "clevrer_collision_annotation_less_frame_v1"
    sample.meta.source_sha256 = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
    return sample, {"program_provenance": p}


@pytest.mark.parametrize("bad", [None, "gold", "gap", "hash", "postfix"])
def test_program_truth_and_provenance_boundary(bad):
    code = module()
    sample, row = program_sample()
    if bad == "gold":
        sample.questions[0].target.value = "no"
    elif bad == "gap":
        row["program_provenance"]["program"]["events"][1]["frame"] = 11
    elif bad == "hash":
        row["program_provenance"]["program_sha256"] = "0" * 64
    elif bad == "postfix":
        row["program_provenance"]["original_postfix_execution"] = "executed"
    if bad:
        with pytest.raises(ValueError):
            code.order_program_binding(sample, row)
    else:
        code.order_program_binding(sample, row)


def test_order_paths_keep_all_coverage_history(tmp_path):
    code = module()
    paths = code.pilot_paths(tmp_path, temporal_order_contrast=True)
    assert paths[:-1] == code.pilot_paths(tmp_path, coverage_segment003=True)
    assert paths[-1][3] == tmp_path / "artifacts/s4-temporal-order-contrast-pilot-001"
    assert len(code.pilot_paths(tmp_path)) == 2


def test_order_parent_and_failed_update_budget():
    code = module()
    parent = {
        "path": "/f2db",
        "weights_sha256": "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3",
    }
    prep = {
        "status": "data_ready_only",
        "exit_code": 0,
        "parent_model": None,
        "training_authorized": False,
        "pending_parent_selection": True,
        "inventory_isolation": "isolated",
        "safe_family_count": 30,
        "video_families": 30,
        "video_rows": 60,
        "independently_evaluated_programs": 60,
    }
    launch = {
        "status": "failed",
        "exit_code": 1,
        "parent_selection": "root_selected_f2db_balanced_retention_and_full_video_evidence",
        "selected_init_adapter": parent,
    }
    final = {
        "status": "failed",
        "exit_code": 1,
        "steps_completed": 3,
        "init_adapter": parent,
        "config": {
            "seed": 42,
            "sample_count": 120,
            "steps": 10,
            "accumulation": 12,
            "midway_step": 5,
            "checkpoint_every": 5,
            "init_adapter": parent,
        },
    }
    code.claim_terminal(final, "s4-temporal-order-contrast")
    code.order_parent_binding(prep, launch, final)
    prep["video_families"] = 60
    with pytest.raises(ValueError, match="order_preparation_contract"):
        code.order_parent_binding(prep, launch, final)
