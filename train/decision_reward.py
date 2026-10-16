"""CPU-tested finite-action expected reward prototype; no trainer/readout integration."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from data.schema import Question, answer_tokens


def _reward_values(rewards: Mapping[str, float], actions: tuple[str, ...]) -> list[float]:
    if not isinstance(rewards, Mapping) or set(rewards) != set(actions):
        raise ValueError("rewards must cover exactly all legal answers including U")
    if any(
        isinstance(r, bool) or not isinstance(r, Real) or not math.isfinite(float(r))
        for r in rewards.values()
    ):
        raise ValueError("rewards must be finite fixed real values")
    return [float(rewards[action]) for action in actions]


def expected_reward_loss(
    logits: Any,
    *,
    question: Question,
    token_ids: Mapping[str, int],
    rewards: Mapping[str, float],
    reference_probs: Any = None,
    reference_kl_weight: float = 0.0,
) -> Any:
    """Return FP32 -E_q[r] + weight*KL(stopgrad(reference)||q).

    q is the joint FP32 softmax of every legal answer token AND U, selected from
    one complete vocabulary vector. Question defines canonical action order;
    mapping insertion order has no effect. Rewards are fixed caller-specified
    action utilities/costs, not differentiable model scores or new supervision.
    A reference tensor/list uses answer_tokens(question)+('U',) order; a mapping
    must name exactly that action set. KL delegates to PyTorch's existing operator.

    Centering fixed rewards preserves the exact objective/gradient while making
    constant-reward gradients exactly zero. One-hot gold reward gives -q_gold,
    whose gradient equals q_gold times legal-CE gradient, not Brier or a proper
    calibration score. This helper changes no probability readout, classifier,
    training loop, full-vocabulary CE baseline, or optimizer.
    """
    torch = importlib.import_module("torch")
    if not isinstance(question, Question):
        raise ValueError("question must declare the legal action contract")
    # Revalidate in case a mutable contract was changed after construction.
    question = Question.model_validate(question.model_dump(mode="json"))
    actions = (*answer_tokens(question), "U")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 1
        or not logits.is_floating_point()
        or not bool(torch.isfinite(logits).all())
    ):
        raise ValueError("logits must be one finite floating vocabulary vector")
    if not isinstance(token_ids, Mapping) or set(token_ids) != set(actions):
        raise ValueError("token mapping must cover exactly all legal answers including U")
    ids = [token_ids[action] for action in actions]
    if any(type(i) is not int or not 0 <= i < logits.numel() for i in ids) or len(set(ids)) != len(
        ids
    ):
        raise ValueError("legal token IDs must be unique integer vocabulary indices")
    if (
        isinstance(reference_kl_weight, bool)
        or not isinstance(reference_kl_weight, Real)
        or not math.isfinite(float(reference_kl_weight))
        or reference_kl_weight < 0
    ):
        raise ValueError("reference KL weight must be finite and nonnegative")
    values = _reward_values(rewards, actions)
    selected = logits[ids].float()
    fixed = torch.tensor(values, device=selected.device, dtype=torch.float32)
    centered = fixed - fixed[0]
    if not bool(torch.isfinite(selected).all()) or not bool(torch.isfinite(centered).all()):
        raise ValueError("selected logits/rewards must remain finite in FP32")
    log_q = selected.log_softmax(dim=-1)
    q = log_q.exp()
    loss = -(q * centered).sum() - fixed[0]
    if reference_kl_weight:
        if reference_probs is None:
            raise ValueError("positive KL weight requires a complete reference distribution")
        if isinstance(reference_probs, Mapping):
            if set(reference_probs) != set(actions):
                raise ValueError("reference must cover exactly all legal answers including U")
            reference_probs = [reference_probs[action] for action in actions]
        reference = torch.as_tensor(reference_probs, device=selected.device).detach()
        if reference.is_complex():
            raise ValueError("reference must contain real probabilities")
        reference = reference.float()
        if (
            reference.shape != selected.shape
            or not bool(torch.isfinite(reference).all())
            or bool((reference < 0).any())
            or not bool(
                torch.isclose(reference.sum(), reference.new_tensor(1.0), atol=1e-6, rtol=0)
            )
        ):
            raise ValueError("reference must be a finite normalized joint action distribution")
        loss = loss + reference_kl_weight * torch.nn.functional.kl_div(
            log_q, reference, reduction="sum"
        )
    elif reference_probs is not None:
        raise ValueError("reference distribution supplied with zero KL weight")
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite expected reward objective")
    return loss
