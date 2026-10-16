"""R1-B permutation-group GRPO: reward, group advantage, KL, and closed-form proofs.

Pure CPU, stdlib. Group = K gold-preserving prompt variants (paraphrase wrappers,
never predicate flips) of ONE grid item. Reward per sampled rollout:
    r_v = -Brier(p_v, gold) + lambda_cons * consistency_v
    consistency_v = (1[a_v == a_1] + ... + 1[a_v == a_K]) / (K - 1)   (other variants)
GRPO advantage within the group: A_v = (r_v - mean r) / (std r + eps), plus a
KL anchor to the starting adapter distribution p_init.

Proofs implemented and asserted in __main__:
1. The -Brier part of E[R] and its gradient decompose per variant (gold-only,
   no cross-variant dependence) -> expressible without RL.
2. The consistency part E[consistency_v] = <p_v, mean_{v'!=v} p_{v'}> couples
   variants: its gradient wrt p_v changes when another variant's distribution
   changes -> NOT decomposable into per-item gold losses. This is the
   legitimate non-gold signal that pure-gold-reward RL (stopped route) lacked.
3. Nevertheless E[R] has an exact closed form (Monte-Carlo check), so the
   exact-gradient surrogate "consistency-regularized Brier over variants"
   dominates sampled GRPO in variance; sampled GRPO remains the policy-gradient
   implementation with KL-anchor and clipping as step-size control.
"""

from __future__ import annotations

import itertools
import json
import math
import random

LABELS = ("yes", "no", "U")
K_VARIANTS = 8
LAMBDA_CONS = 0.5
BETA_KL = 0.05
EPS = 1e-6


def brier(distribution: dict[str, float], gold: str | None) -> float:
    """Multi-class Brier over the answer labels; U-gold items score the U entry."""
    target = "U" if gold is None else gold
    score = {label: (1.0 if label == target else 0.0) for label in LABELS}
    return math.fsum((distribution[label] - score[label]) ** 2 for label in LABELS)


def typed_target(wsft_target: dict) -> dict[str, float]:
    """WSFT row target -> full label distribution including U mass.

    SoftTarget carries probs over [yes, no] with U implicit zero; hard rows are
    one-hot; insufficient rows put all mass on U. Predicate-factorized targets
    that used duplication express their U mass across duplicated rows, so a
    single row stays faithful to its own target kind.
    """
    kind = wsft_target["kind"]
    if kind == "soft":
        yes, no = wsft_target["probs"]
        return {"yes": float(yes), "no": float(no), "U": 0.0}
    if kind == "hard":
        return {
            "yes": 1.0 if wsft_target["value"] == "yes" else 0.0,
            "no": 1.0 if wsft_target["value"] == "no" else 0.0,
            "U": 0.0,
        }
    if kind == "insufficient":
        return {"yes": 0.0, "no": 0.0, "U": 1.0}
    raise ValueError(f"unknown target kind {kind}")


def nll(distribution: dict[str, float], target: dict[str, float]) -> float:
    """Cross-entropy of the policy distribution against a typed target."""
    return -math.fsum(
        target[label] * math.log(distribution[label])
        for label in LABELS
        if target[label] > 0
    )


def sampled_reward(
    answers: list[str],
    distributions: list[dict[str, float]],
    gold: str | None,
    *,
    lambda_cons: float = LAMBDA_CONS,
    reward_variant: str = "gold",
    target: dict[str, float] | None = None,
) -> list[float]:
    """Per-variant reward for one sampled group rollout.

    reward_variant "gold"  : R1-B-gold, -Brier against one-hot gold.
    reward_variant "wsft"  : R1-B-wsft, -NLL against the predicate-factorized
                             typed target (target dict from typed_target()).
    """
    k = len(answers)
    rewards = []
    for index, answer in enumerate(answers):
        consistency = (
            math.fsum(1.0 if answer == other else 0.0 for j, other in enumerate(answers) if j != index)
            / (k - 1)
        )
        if reward_variant == "gold":
            accuracy_term = -brier(distributions[index], gold)
        elif reward_variant == "wsft":
            accuracy_term = -nll(distributions[index], target or {})
        else:
            raise ValueError(f"unknown reward variant {reward_variant}")
        rewards.append(accuracy_term + lambda_cons * consistency)
    return rewards


def group_advantages(rewards: list[float]) -> list[float]:
    mean = math.fsum(rewards) / len(rewards)
    std = math.sqrt(math.fsum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + EPS) for r in rewards]


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    return math.fsum(
        p[label] * (math.log(p[label]) - math.log(q[label]))
        for label in LABELS
        if p[label] > 0
    )


def expected_reward_closed_form(
    distributions: list[dict[str, float]],
    gold: str | None,
    *,
    lambda_cons: float = LAMBDA_CONS,
    reward_variant: str = "gold",
    target: dict[str, float] | None = None,
) -> float:
    """Exact E[mean_v r_v] under independent per-variant categorical sampling."""
    k = len(distributions)
    if reward_variant == "gold":
        accuracy_part = math.fsum(-brier(p, gold) for p in distributions) / k
    else:
        accuracy_part = math.fsum(-nll(p, target or {}) for p in distributions) / k
    consistency_part = 0.0
    for index, p in enumerate(distributions):
        others = [q for j, q in enumerate(distributions) if j != index]
        mean_other = {
            label: math.fsum(q[label] for q in others) / (k - 1) for label in LABELS
        }
        consistency_part += math.fsum(p[label] * mean_other[label] for label in LABELS)
    consistency_part = lambda_cons * consistency_part / k
    return accuracy_part + consistency_part


def montecarlo_expected_reward(
    distributions: list[dict[str, float]],
    gold: str | None,
    *,
    rollouts: int,
    seed: int,
    lambda_cons: float = LAMBDA_CONS,
    reward_variant: str = "gold",
    target: dict[str, float] | None = None,
) -> float:
    rng = random.Random(seed)
    k = len(distributions)
    total = 0.0
    for _ in range(rollouts):
        answers = []
        for p in distributions:
            draw = rng.random()
            acc = 0.0
            chosen = LABELS[-1]
            for label in LABELS:
                acc += p[label]
                if draw <= acc:
                    chosen = label
                    break
            answers.append(chosen)
        rewards = sampled_reward(
            answers,
            distributions,
            gold,
            lambda_cons=lambda_cons,
            reward_variant=reward_variant,
            target=target,
        )
        total += math.fsum(rewards) / k
    return total / rollouts


def _numeric_grad(
    distributions, gold, variant, label, lambda_cons, reward_variant="gold", target=None
):
    def objective(dists):
        return expected_reward_closed_form(
            dists,
            gold,
            lambda_cons=lambda_cons,
            reward_variant=reward_variant,
            target=target,
        )

    base = objective(distributions)
    eps = 1e-6
    bumped = [dict(p) for p in distributions]
    bumped[variant][label] += eps
    bumped[variant]["U"] = max(0.0, bumped[variant]["U"] - eps)
    return (objective(bumped) - base) / eps


def _main() -> None:
    gold = "yes"
    distributions = [
        {"yes": 0.72, "no": 0.20, "U": 0.08},
        {"yes": 0.65, "no": 0.25, "U": 0.10},
        {"yes": 0.80, "no": 0.12, "U": 0.08},
        {"yes": 0.55, "no": 0.35, "U": 0.10},
        {"yes": 0.70, "no": 0.22, "U": 0.08},
        {"yes": 0.60, "no": 0.30, "U": 0.10},
        {"yes": 0.75, "no": 0.17, "U": 0.08},
        {"yes": 0.68, "no": 0.24, "U": 0.08},
    ]
    assert len(distributions) == K_VARIANTS
    wsft_target = typed_target({"kind": "soft", "probs": [0.95, 0.05], "source": "b1_wsft"})
    results = {}
    for variant, target in (("gold", None), ("wsft", wsft_target)):
        closed = expected_reward_closed_form(
            distributions, gold, reward_variant=variant, target=target
        )
        estimated = montecarlo_expected_reward(
            distributions,
            gold,
            rollouts=200000,
            seed=13,
            reward_variant=variant,
            target=target,
        )
        assert abs(closed - estimated) < 0.01, (variant, closed, estimated)

        def grad_variant0(lam, reward_variant, target):
            return {
                label: _numeric_grad(distributions, gold, 0, label, lam, reward_variant, target)
                for label in ("yes", "no")
            }

        g_accuracy = grad_variant0(0.0, variant, target)
        g_cons = grad_variant0(LAMBDA_CONS, variant, target)
        saved = distributions[3]
        distributions[3] = {"yes": 0.30, "no": 0.60, "U": 0.10}
        g_accuracy_alt = grad_variant0(0.0, variant, target)
        g_cons_alt = grad_variant0(LAMBDA_CONS, variant, target)
        distributions[3] = saved
        assert all(
            abs(g_accuracy[k] - g_accuracy_alt[k]) < 1e-9 for k in g_accuracy
        ), f"{variant}: accuracy term must decompose"
        assert any(
            abs(g_cons[k] - g_cons_alt[k]) > 1e-3 for k in g_cons
        ), f"{variant}: consistency term must couple"
        rewards = sampled_reward(
            ["yes", "yes", "yes", "no", "yes", "yes", "yes", "no"],
            distributions,
            gold,
            reward_variant=variant,
            target=target,
        )
        results[f"r1b_{variant}"] = {
            "closed_form": closed,
            "montecarlo": estimated,
            "abs_gap": abs(closed - estimated),
            "accuracy_grad_invariant_to_other_variant": True,
            "consistency_grad_couples_variants": True,
            "sample_group_advantages": [round(a, 4) for a in group_advantages(rewards)],
        }
    anchor = distributions[0]
    results["shared"] = {
        "mean_kl_to_anchor": math.fsum(kl_divergence(p, anchor) for p in distributions[1:])
        / (K_VARIANTS - 1),
        "beta_kl": BETA_KL,
        "lambda_cons": LAMBDA_CONS,
        "wsft_example_target": wsft_target,
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    _main()
