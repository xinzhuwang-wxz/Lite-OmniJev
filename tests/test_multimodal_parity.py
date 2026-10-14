"""Synthetic numerical gate tests; no model or GPU acceptance evidence."""

import hashlib
import math

import pytest

from eval.multimodal_parity import RawReadout, compare_rows, make_row


def rows(probabilities=(0.2, 0.2, 0.1)):
    return [
        make_row(
            id_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
            input_sha256="a" * 64,
            media_sha256=["b" * 64],
            token_ids=[10, 11, 12],
            raw_logprobs=[math.log(value) for value in probabilities],
        )
        for index in range(100)
    ]


def test_all_four_distributions_pass_without_granting_quality_acceptance():
    result = compare_rows(rows(), rows(), exact=True)
    assert result["status"] == "passed"
    assert result["max_differences"] == dict.fromkeys(
        ("raw_probabilities", "joint", "conditional", "p_U"), 0.0
    )
    assert result["release_or_stage_acceptance"] is False


def test_joint_difference_cannot_hide_behind_smaller_conditional_and_u_differences():
    reference = rows((0.5 * 0.001 * 0.999, 0.5 * 0.999 * 0.999, 0.5 * 0.001))
    candidate = rows((0.5 * 0.0002 * 0.9998, 0.5 * 0.9998 * 0.9998, 0.5 * 0.0002))
    result = compare_rows(reference, candidate)
    assert result["max_differences"]["conditional"] < 0.001
    assert result["max_differences"]["p_U"] < 0.001
    assert result["max_differences"]["raw_probabilities"] < 0.001
    assert result["max_differences"]["joint"] > 0.001
    assert result["status"] == "failed"


def test_raw_vocabulary_mass_and_exact_rerun_are_independent_checks():
    assert compare_rows(rows(), rows((0.2001, 0.2001, 0.10005)))["status"] == "passed"
    assert compare_rows(rows(), rows((0.2001, 0.2001, 0.10005)), exact=True)["status"] == "failed"
    assert compare_rows(rows(), rows((0.3, 0.3, 0.15)))["status"] == "failed"


def test_strict_threshold_does_not_admit_boundary_and_accepts_smaller_change():
    assert (
        compare_rows(rows((0.25, 0.25, 1e-20)), rows((0.2505, 0.2495, 1e-20)))["status"] == "failed"
    )
    assert (
        compare_rows(rows((0.25, 0.25, 1e-20)), rows((0.2501, 0.2499, 1e-20)))["status"] == "passed"
    )


def test_joint_keeps_tiny_answer_masses_even_when_u_rounds_to_one():
    row = make_row(
        id_sha256="a" * 64,
        input_sha256="b" * 64,
        media_sha256=[],
        token_ids=[1, 2, 3],
        raw_logprobs=[-40.0, -41.0, -0.00001],
    )
    assert row.p_U == 1.0
    assert row.joint[0] > 0 and row.joint[1] > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -9999.0, 0.1, True])
def test_invalid_or_clipped_logprobs_fail_closed(bad):
    with pytest.raises(ValueError):
        make_row(
            id_sha256="a" * 64,
            input_sha256="b" * 64,
            media_sha256=[],
            token_ids=[1, 2, 3],
            raw_logprobs=[-2.0, -3.0, bad],
        )


def test_untrusted_derived_fields_and_impossible_full_vocabulary_mass_are_rejected():
    changed = rows()[0].model_dump()
    changed["joint"][0] += 0.01
    with pytest.raises(ValueError):
        RawReadout.model_validate(changed)
    with pytest.raises(ValueError):
        rows((0.5, 0.5, 0.5))


def test_complete_unique_cohort_and_input_identity_are_required():
    with pytest.raises(ValueError, match="100"):
        compare_rows(rows()[:-1], rows())
    with pytest.raises(ValueError, match="unique"):
        compare_rows([rows()[0]] * 100, rows())
    changed = rows()
    changed[0].media_sha256 = ["c" * 64]
    with pytest.raises(ValueError, match="identity"):
        compare_rows(rows(), changed)
