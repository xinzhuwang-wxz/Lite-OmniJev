"""CPU analytic behavior, not training, calibration or GPU benefit evidence."""

import math

import pytest

from data.schema import Question
from train.decision_reward import expected_reward_loss

torch = pytest.importorskip("torch")
QUESTION = Question(qid="decision", type="noul", instruction="Synthetic controlled choice")
IDS = {"U": 4, "no": 1, "yes": 3}
REWARDS = {"no": -2.0, "U": -0.25, "yes": 1.0}


def loss(z, **overrides):
    kwargs = dict(question=QUESTION, token_ids=IDS, rewards=REWARDS)
    kwargs.update(overrides)
    return expected_reward_loss(z, **kwargs)


def test_expected_value_and_analytic_gradient_include_u_and_only_legal_tokens():
    z = torch.tensor([9.0, -0.4, -8.0, 0.7, 0.2], requires_grad=True)
    result = loss(z)
    q = torch.softmax(z.detach()[[3, 1, 4]], dim=0)
    r = torch.tensor([1.0, -2.0, -0.25])
    expectation = (q * r).sum()
    assert result.dtype == torch.float32 and result.ndim == 0
    assert result.item() == pytest.approx(-expectation.item(), abs=1e-7)
    result.backward()
    expected = torch.zeros(5)
    expected[[3, 1, 4]] = q * (expectation - r)
    assert torch.allclose(z.grad, expected, atol=1e-7)
    assert z.grad[4] != 0 and z.grad[0] == z.grad[2] == 0


def test_finite_difference_and_reward_shift_mapping_order_invariance():
    z = torch.tensor([1.0, -0.4, 4.0, 0.7, 0.2], requires_grad=True)
    gradient = torch.autograd.grad(loss(z), z)[0]
    for i in range(5):
        delta = torch.zeros_like(z)
        delta[i] = 1e-3
        numerical = (loss(z.detach() + delta) - loss(z.detach() - delta)) / (2e-3)
        assert numerical.item() == pytest.approx(gradient[i].item(), abs=8e-5)
    shifted = {k: v + 16 for k, v in REWARDS.items()}
    shifted_gradient = torch.autograd.grad(loss(z, rewards=shifted), z)[0]
    assert torch.allclose(gradient, shifted_gradient, atol=1e-7)
    assert loss(z, rewards=shifted).item() == pytest.approx(loss(z).item() - 16, abs=2e-6)
    assert loss(z, token_ids={"yes": 3, "no": 1, "U": 4}).item() == loss(z).item()


@pytest.mark.parametrize("constant", [0.0, 2.75, -3.0])
def test_constant_reward_exactly_zero_gradient(constant):
    z = torch.tensor([0.0, 2.0, 4.0, -0.3, 1.0], requires_grad=True)
    result = loss(z, rewards={k: constant for k in IDS})
    assert result.item() == -constant
    assert torch.equal(torch.autograd.grad(result, z)[0], torch.zeros_like(z))


def test_onehot_reward_is_gold_probability_scaled_legal_ce_gradient_not_brier():
    z = torch.tensor([0.0, -0.6, 3.0, 0.2, 0.4], requires_grad=True)
    reward_gradient = torch.autograd.grad(loss(z, rewards={"yes": 1.0, "no": 0.0, "U": 0.0}), z)[0]
    selected = z[[3, 1, 4]]
    q = selected.softmax(0)
    ce_gradient = torch.autograd.grad(-selected.log_softmax(0)[0], z, retain_graph=True)[0]
    assert torch.allclose(reward_gradient, q[0].detach() * ce_gradient, atol=1e-7)
    brier_gradient = torch.autograd.grad((q - torch.tensor([1.0, 0.0, 0.0])).square().sum(), z)[0]
    assert not torch.allclose(reward_gradient, brier_gradient, atol=1e-4)


def test_reference_kl_matches_standard_operator_and_is_stopgrad():
    z = torch.tensor([0.0, -0.4, 5.0, 0.7, 0.2], requires_grad=True)
    reference = torch.tensor([0.2, 0.3, 0.5], requires_grad=True)
    result = loss(z, reference_probs=reference, reference_kl_weight=0.3)
    logq = z[[3, 1, 4]].log_softmax(0)
    expected_kl = torch.nn.functional.kl_div(logq, reference.detach(), reduction="sum")
    assert result.item() == pytest.approx((loss(z) + 0.3 * expected_kl).item(), abs=1e-7)
    result.backward()
    assert reference.grad is None and torch.isfinite(z.grad).all()
    assert (
        loss(
            z.detach(), reference_probs={"no": 0.3, "U": 0.5, "yes": 0.2}, reference_kl_weight=0.3
        ).item()
        == result.item()
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_logits_keep_gradient_and_fp32_scalar(dtype):
    z = torch.zeros(5, dtype=dtype, requires_grad=True)
    result = loss(z)
    assert result.dtype == torch.float32
    result.backward()
    assert z.grad.dtype == dtype and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"rewards": {"yes": 1.0, "no": 0.0}},
        {"rewards": {"yes": 1.0, "no": 0.0, "U": math.nan}},
        {"rewards": {"yes": 1.0, "no": 0.0, "U": math.inf}},
        {"rewards": {"yes": 1.0, "no": 0.0, "U": True}},
        {"rewards": {"yes": 1.0, "no": 0.0, "U": 0.0, "extra": 0.0}},
        {"token_ids": {"yes": 3, "no": 1}},
        {"token_ids": {"yes": 3, "no": 1, "U": 3}},
        {"token_ids": {"yes": 3, "no": 1, "U": 5}},
        {"token_ids": {"yes": True, "no": 1, "U": 4}},
        {"reference_kl_weight": -0.1},
        {"reference_kl_weight": math.nan},
        {"reference_kl_weight": 0.1},
        {"reference_probs": [0.2, 0.3, 0.5]},
        {"reference_probs": [0.2, 0.3], "reference_kl_weight": 0.1},
        {"reference_probs": [0.2, 0.3, 0.6], "reference_kl_weight": 0.1},
        {"reference_probs": [0.2, -0.1, 0.9], "reference_kl_weight": 0.1},
        {"reference_probs": [0.2, math.inf, 0.8], "reference_kl_weight": 0.1},
        {"reference_probs": {"yes": 0.5, "no": 0.5}, "reference_kl_weight": 0.1},
    ],
)
def test_invalid_action_or_reward_reference_contract_rejected(overrides):
    with pytest.raises(ValueError):
        loss(torch.zeros(5), **overrides)


@pytest.mark.parametrize(
    "z",
    [
        torch.zeros(1, 5),
        torch.zeros(5, dtype=torch.long),
        torch.tensor([math.nan, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, math.inf, 0.0, 0.0]),
    ],
)
def test_invalid_full_vocabulary_rejected(z):
    with pytest.raises(ValueError):
        loss(z)


@pytest.mark.parametrize(
    "question,token_ids",
    [
        (
            Question(qid="c", type="choice", instruction="Synthetic", options=["x", "y"]),
            {"B": 2, "U": 0, "A": 1},
        ),
        (
            Question(qid="s", type="score", instruction="Synthetic", score_levels=3),
            {"2": 3, "U": 0, "0": 1, "1": 2},
        ),
    ],
)
def test_choice_and_score_use_contract_action_sets(question, token_ids):
    z = torch.zeros(5, requires_grad=True)
    r = {action: float(action == "U") for action in token_ids}
    result = expected_reward_loss(z, question=question, token_ids=token_ids, rewards=r)
    assert result.item() == pytest.approx(-1 / len(token_ids))
    result.backward()
    assert z.grad[token_ids["U"]] < 0


def test_action_dependent_business_cost_changes_direction_relative_to_gold_reward():
    z = torch.zeros(5, requires_grad=True)
    business = torch.autograd.grad(loss(z), z)[0][[3, 1, 4]]
    gold = torch.autograd.grad(loss(z, rewards={"yes": 1.0, "no": 0.0, "U": 0.0}), z)[0][[3, 1, 4]]
    # Uniform q: gold treats no and U equally; explicit costs do not.
    assert gold[1] == gold[2] and business[1] != business[2]


def test_complex_reference_rejected_instead_of_silently_discarding_imaginary_part():
    with pytest.raises(ValueError, match="real"):
        loss(
            torch.zeros(5),
            reference_probs=torch.tensor([0.2 + 1j, 0.3 + 0j, 0.5 + 0j]),
            reference_kl_weight=0.1,
        )
