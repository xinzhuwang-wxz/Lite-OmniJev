"""CPU mathematics and contract tests; not GPU or training-quality evidence."""

import math

import pytest

from train.pair_objective import claim_pair_loss

torch = pytest.importorskip("torch")


def loss(positive, negative, **overrides):
    kwargs = dict(
        positive_token_ids={"U": 0, "no": 2, "yes": 1},
        negative_token_ids={"yes": 3, "U": 1, "no": 0},
        positive_gold="yes",
        negative_gold="no",
        positive_family_id="component-1",
        negative_family_id="component-1",
    )
    kwargs.update(overrides)
    return claim_pair_loss(positive, negative, **kwargs)


def test_formula_and_gradient_direction_use_explicit_token_mapping():
    positive = torch.tensor([7.0, 0.5, -0.5, 4.0], requires_grad=True)
    negative = torch.tensor([0.5, 8.0, 9.0, 0.0], requires_grad=True)
    result = loss(positive, negative)
    # s_positive=1, s_negative=-.5; softplus(-.5).
    assert result.dtype == torch.float32 and result.ndim == 0
    assert result.item() == pytest.approx(math.log1p(math.exp(-0.5)))
    result.backward()
    magnitude = 1 / (1 + math.exp(0.5))
    assert positive.grad.tolist() == pytest.approx([0, -magnitude, magnitude, 0])
    assert negative.grad.tolist() == pytest.approx([-magnitude, 0, 0, magnitude])
    assert loss(positive.detach() + 100, negative.detach() - 50).item() == pytest.approx(
        result.item()
    )
    assert (
        loss(positive, negative, negative_token_ids={"no": 0, "yes": 3, "U": 1}).item()
        == result.item()
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_inputs_produce_fp32_scalar_and_retain_gradients(dtype):
    positive = torch.zeros(4, dtype=dtype, requires_grad=True)
    negative = torch.zeros(4, dtype=dtype, requires_grad=True)
    result = loss(positive, negative)
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(math.log1p(math.exp(1)))
    result.backward()
    assert positive.grad.dtype == negative.grad.dtype == dtype
    assert torch.isfinite(positive.grad).all() and torch.isfinite(negative.grad).all()
    assert positive.grad[1] < 0 and positive.grad[2] > 0
    assert negative.grad[3] > 0 and negative.grad[0] < 0


@pytest.mark.parametrize(
    "override",
    [
        {"positive_gold": "no", "negative_gold": "yes"},
        {"positive_gold": "U"},
        {"negative_gold": "yes"},
        {"negative_family_id": "different"},
        {"positive_family_id": ""},
        {"positive_token_ids": {"yes": 0, "no": 0, "U": 1}},
        {"positive_token_ids": {"yes": -1, "no": 2, "U": 1}},
        {"positive_token_ids": {"yes": 4, "no": 2, "U": 1}},
        {"negative_token_ids": {"yes": True, "no": 2, "U": 1}},
        {"negative_token_ids": {"yes": 0, "no": 2, "u": 1}},
        {"negative_token_ids": {"yes": 0, "no": 2}},
        {"negative_token_ids": {"yes": 0, "no": 2, "U": 1, "extra": 3}},
    ],
)
def test_invalid_pair_contract_rejected(override):
    with pytest.raises(ValueError):
        loss(torch.zeros(4), torch.zeros(4), **override)


@pytest.mark.parametrize(
    "invalid",
    [
        torch.zeros(1, 4),
        torch.tensor([1, 2, 3, 4]),
        torch.zeros(2),
        torch.tensor([float("nan"), 0.0, 0.0, 0.0]),
        torch.tensor([0.0, float("inf"), 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, -float("inf")]),
    ],
)
def test_invalid_full_vocabulary_rejected_on_both_sides(invalid):
    with pytest.raises(ValueError):
        loss(invalid, torch.zeros(4))
    with pytest.raises(ValueError):
        loss(torch.zeros(4), invalid)


def test_finite_inputs_with_overflowing_fp32_margin_are_rejected():
    positive = torch.tensor([0.0, -3e38, 3e38, 0.0])
    with pytest.raises(ValueError, match="nonfinite FP32"):
        loss(positive, torch.zeros(4))


def test_same_mapping_for_both_inputs_is_supported():
    mapping = {"no": 1, "U": 2, "yes": 0}
    positive = torch.tensor([2.0, 0.0, 10.0], requires_grad=True)
    negative = torch.tensor([0.0, 1.0, -20.0], requires_grad=True)
    result = loss(positive, negative, positive_token_ids=mapping, negative_token_ids=mapping)
    assert result.item() == pytest.approx(math.log1p(math.exp(-2)))
    result.backward()
    assert positive.grad[2] == negative.grad[2] == 0
