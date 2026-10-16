"""CPU contracts and gradients only; no real-model quality or training evidence."""

import math

import pytest

from train.pair_objective import claim_pair_loss
from train.video_pair_objective import video_pair_loss

torch = pytest.importorskip("torch")


def binding():
    return {
        "pair_id": "opaque-pair-1",
        "positive_question_sha256": "a" * 64,
        "negative_question_sha256": "a" * 64,
        "positive_program_sha256": "b" * 64,
        "negative_program_sha256": "b" * 64,
    }


def arguments():
    return dict(
        positive_token_ids={"U": 0, "no": 2, "yes": 1},
        negative_token_ids={"yes": 3, "U": 1, "no": 0},
        positive_gold="yes",
        negative_gold="no",
        positive_family_id="video-family-1",
        negative_family_id="video-family-2",
        match_binding=binding(),
    )


def loss(positive, negative, **overrides):
    kwargs = arguments()
    kwargs.update(overrides)
    return video_pair_loss(positive, negative, **kwargs)


def test_conditional_likelihood_and_gradients_on_both_videos():
    positive = torch.tensor([7.0, 0.5, -0.5, 4.0], requires_grad=True)
    negative = torch.tensor([0.5, 8.0, 9.0, 0.0], requires_grad=True)
    result = loss(positive, negative)
    assert result.ndim == 0 and result.dtype == torch.float32
    assert result.item() == pytest.approx(math.log1p(math.exp(-1.5)))
    qp = positive[[1, 2, 0]].softmax(0)
    qn = negative[[3, 0, 1]].softmax(0)
    conditional = qp[0] * qn[1] / (qp[0] * qn[1] + qp[1] * qn[0])
    assert result.item() == pytest.approx(-conditional.log().item())
    result.backward()
    magnitude = 1 / (1 + math.exp(1.5))
    assert positive.grad.tolist() == pytest.approx([0, -magnitude, magnitude, 0])
    assert negative.grad.tolist() == pytest.approx([-magnitude, 0, 0, magnitude])
    assert (
        loss(positive.detach() - 0.1 * positive.grad, negative.detach() - 0.1 * negative.grad)
        < result
    )


def test_common_question_prior_and_vocab_offsets_cancel():
    positive = torch.tensor([0.0, 1.0, 0.0, 0.0])
    negative = torch.zeros(4)
    baseline = loss(positive, negative)
    # Add the same yes/no log-odds prior to both differently mapped videos.
    positive[1] += 8
    negative[3] += 8
    assert torch.equal(loss(positive, negative), baseline)
    assert torch.equal(loss(positive + 16, negative - 16), baseline)
    assert torch.equal(
        loss(positive, negative, negative_token_ids={"U": 1, "no": 0, "yes": 3}), baseline
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_fp32_reduction_retains_input_gradient(dtype):
    positive = torch.zeros(4, dtype=dtype, requires_grad=True)
    negative = torch.zeros(4, dtype=dtype, requires_grad=True)
    value = loss(positive, negative)
    assert value.dtype == torch.float32
    assert value.item() == pytest.approx(math.log(2))
    value.backward()
    assert positive.grad.dtype == negative.grad.dtype == dtype
    assert positive.grad[1] == -0.5 and negative.grad[3] == 0.5


@pytest.mark.parametrize("delta", [-10000.0, 10000.0])
def test_extreme_finite_differences_are_stable(delta):
    positive = torch.tensor([0.0, delta, 0.0, 0.0], requires_grad=True)
    negative = torch.zeros(4, requires_grad=True)
    value = loss(positive, negative)
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(positive.grad).all() and torch.isfinite(negative.grad).all()


@pytest.mark.parametrize(
    "override",
    [
        {"positive_gold": "no", "negative_gold": "yes"},
        {"negative_gold": "yes"},
        {"positive_gold": "U"},
        {"negative_family_id": "video-family-1"},
        {"positive_family_id": " "},
        {"negative_family_id": ""},
        {"negative_family_id": None},
        {"positive_token_ids": {"yes": 1, "no": 1, "U": 0}},
        {"positive_token_ids": {"yes": True, "no": 2, "U": 0}},
        {"positive_token_ids": {"yes": 4, "no": 2, "U": 0}},
        {"positive_token_ids": {"yes": -1, "no": 2, "U": 0}},
        {"negative_token_ids": {"yes": 3, "no": 0}},
        {"negative_token_ids": {"yes": 3, "no": 0, "U": 1, "extra": 2}},
        {"negative_token_ids": None},
        {"match_binding": None},
        {"match_binding": {}},
    ],
)
def test_reject_invalid_contract(override):
    with pytest.raises(ValueError):
        loss(torch.zeros(4), torch.zeros(4), **override)


@pytest.mark.parametrize(
    "key,value",
    [
        ("pair_id", " "),
        ("pair_id", None),
        ("positive_question_sha256", "c" * 64),
        ("negative_program_sha256", "c" * 64),
        ("positive_program_sha256", "g" * 64),
        ("negative_question_sha256", None),
    ],
)
def test_reject_unmatched_or_invalid_binding(key, value):
    data = binding()
    data[key] = value
    with pytest.raises(ValueError):
        loss(torch.zeros(4), torch.zeros(4), match_binding=data)


@pytest.mark.parametrize(
    "invalid",
    [
        [0.0, 0.0, 0.0],
        torch.zeros(1, 4),
        torch.zeros(2),
        torch.zeros(4, dtype=torch.int64),
        torch.tensor([0.0, 0.0, 0.0, math.nan]),
        torch.tensor([0.0, math.inf, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, -math.inf, 0.0]),
        torch.zeros(4, dtype=torch.complex64),
    ],
)
@pytest.mark.parametrize("side", ["positive", "negative"])
def test_reject_invalid_complete_vocab_vectors(invalid, side):
    with pytest.raises(ValueError):
        loss(
            invalid if side == "positive" else torch.zeros(4),
            invalid if side == "negative" else torch.zeros(4),
        )


def test_reject_device_mismatch_without_gpu():
    with pytest.raises(ValueError, match="share a device"):
        loss(torch.zeros(4), torch.zeros(4, device="meta"))


def test_reject_fp32_cast_or_difference_overflow():
    with pytest.raises(ValueError, match="nonfinite FP32"):
        loss(torch.tensor([0.0, 1e100, 0.0, 0.0], dtype=torch.float64), torch.zeros(4))
    with pytest.raises(ValueError, match="nonfinite FP32"):
        loss(torch.tensor([0.0, 3e38, -3e38, 0.0]), torch.zeros(4))


def test_existing_claim_contract_still_rejects_different_families():
    kwargs = arguments()
    kwargs.pop("match_binding")
    with pytest.raises(ValueError, match="share one nonempty component family"):
        claim_pair_loss(torch.zeros(4), torch.zeros(4), **kwargs)
