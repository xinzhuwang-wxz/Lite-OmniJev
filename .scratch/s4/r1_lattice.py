"""R1 GRPO-on-lattice reward, its closed-form expectation, and decomposability proof.

Pure CPU, stdlib only. The module exists to answer one question honestly: does
the lattice reward contain any signal that per-item cross entropy cannot
express? Answer implemented below and asserted by tests:

- alpha (per-item correctness) and gamma (answerability) terms decompose into
  per-item expectations -> advantage-weighted CE; GRPO adds variance, not signal.
- beta twin term E[1[a != b]] = 1 - <p1, p2> couples two different prompts; its
  gradient wrt p1 depends on p2, so it is NOT per-item CE. Closed form exists,
  so exact-gradient supervised training dominates sampled GRPO.
- beta transitivity term for a triple is an 8-case polynomial; also closed form.

Recommended pilot form (downgraded from RL per the no-forced-RL rule):
coupled_loss = alpha * CE(gold incl. U) + beta * <p_twin_left, p_twin_right>
+ optional beta3 * (1 - P_consistent(triple)); all exact, no sampling.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Callable

LABELS = ("yes", "no", "U")


def correctness(item: str, answer: str, gold: str | None) -> int:
    """alpha-term: 1 iff the sampled answer matches the item's supervision."""
    if gold is None:  # U-gold item
        return int(answer == "U")
    return int(answer == gold and answer != "U")


def answerability(item: str, answer: str, gold: str | None) -> int:
    """gamma-term: right U-behavior on both kinds of items."""
    return int((answer == "U") if gold is None else (answer != "U"))


def twin_consistent(left_answer: str, right_answer: str) -> int:
    """beta twin-term: polarity twins must give opposite concrete answers."""
    return int(left_answer != right_answer)


def triple_consistent(ab: str, bc: str, ac: str) -> int:
    """beta transitivity term over three pairwise before/after answers.

    Each answer is "yes"/"no" for (x before y); U breaks consistency.
    """

    def order(answer: str) -> bool | None:
        return {"yes": True, "no": False}.get(answer)

    a, b, c = order(ab), order(bc), order(ac)
    if None in (a, b, c):
        return 0
    if a and b:
        return int(c)
    if (not a) and (not b):
        return int(not c)
    return 1  # mixed chain imposes no constraint on the closure pair


def lattice_reward(
    rollout: dict[str, str],
    golds: dict[str, str | None],
    twin_groups: list[tuple[str, str]],
    triple_groups: list[tuple[str, str, str]],
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    items = list(rollout)
    correct = math.fsum(correctness(i, rollout[i], golds[i]) for i in items) / len(items)
    answer = math.fsum(answerability(i, rollout[i], golds[i]) for i in items) / len(items)
    structure_parts = []
    if twin_groups:
        structure_parts.append(
            math.fsum(twin_consistent(rollout[a], rollout[b]) for a, b in twin_groups)
            / len(twin_groups)
        )
    if triple_groups:
        structure_parts.append(
            math.fsum(
                triple_consistent(rollout[a], rollout[b], rollout[c]) for a, b, c in triple_groups
            )
            / len(triple_groups)
        )
    structure = math.fsum(structure_parts) / len(structure_parts) if structure_parts else 0.0
    return alpha * correct + beta * structure + gamma * answer


def expected_reward_closed_form(
    distributions: dict[str, dict[str, float]],
    golds: dict[str, str | None],
    twin_groups: list[tuple[str, str]],
    triple_groups: list[tuple[str, str]],
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    items = list(distributions)
    correct = math.fsum(
        sum(p * correctness(i, a, golds[i]) for a, p in distributions[i].items())
        for i in items
    ) / len(items)
    answer = math.fsum(
        sum(p * answerability(i, a, golds[i]) for a, p in distributions[i].items())
        for i in items
    ) / len(items)
    twin = 0.0
    if twin_groups:
        twin = math.fsum(
            1.0 - math.fsum(distributions[a][k] * distributions[b][k] for k in LABELS)
            for a, b in twin_groups
        ) / len(twin_groups)
    triple = 0.0
    if triple_groups:
        total = 0.0
        for ab, bc, ac in triple_groups:
            for x, y, z in itertools.product(LABELS, repeat=3):
                probability = distributions[ab][x] * distributions[bc][y] * distributions[ac][z]
                total += probability * triple_consistent(x, y, z)
        triple = total / len(triple_groups)
    group_types = bool(twin_groups) + bool(triple_groups)
    structure = (twin + triple) / group_types if group_types else 0.0
    return alpha * correct + beta * structure + gamma * answer


def coupled_loss(
    distributions: dict[str, dict[str, float]],
    golds: dict[str, str | None],
    twin_groups: list[tuple[str, str]],
    triple_groups: list[tuple[str, str]],
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """Exact differentiable surrogate: -expected_reward_closed_form.

    Minimizing this equals maximizing the expected lattice reward without any
    sampling; the beta terms are the non-CE part.
    """
    return -expected_reward_closed_form(
        distributions,
        golds,
        twin_groups,
        triple_groups,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )


def montecarlo_expected_reward(
    sampler: Callable[[str], str],
    items: list[str],
    golds: dict[str, str | None],
    twin_groups: list[tuple[str, str]],
    triple_groups: list[tuple[str, str, str]],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    rollouts: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    total = 0.0
    for _ in range(rollouts):
        rollout = {item: sampler(item) for item in items}
        total += lattice_reward(
            rollout, golds, twin_groups, triple_groups, alpha=alpha, beta=beta, gamma=gamma
        )
    return total / rollouts


def load_grid_lattice(grid_dir: Path) -> dict:
    """Golds and twin/triple group structure from the B1 grid artifacts."""
    manifest = json.loads((grid_dir / "b1-grid-manifest.json").read_text())
    rows = {
        json.loads(line)["questions"][0]["qid"]: json.loads(line)
        for line in (grid_dir / "b1-grid.jsonl").read_text().splitlines()
        if line.strip()
    }
    golds: dict[str, str | None] = {}
    by_group: dict[str, list[dict]] = {}
    for derivation in manifest["derivations"]:
        target = rows[derivation["qid"]]["questions"][0]["target"]
        golds[derivation["qid"]] = None if target["kind"] == "insufficient" else str(target["value"])
        by_group.setdefault(f"{derivation['scene']}:{derivation['group']}", []).append(derivation)
    twin_groups = [
        (members[0]["qid"], members[1]["qid"])
        for members in by_group.values()
        if len(members) == 2
    ]
    # transitivity triples from direct-order cells sharing endpoints
    direct = {}
    for derivation in manifest["derivations"]:
        if derivation["cell"] == "order_direct_before" and "left" in derivation:
            direct[(derivation["left"], derivation["right"])] = derivation["qid"]
    triple_groups = []
    for (a, b), ab in direct.items():
        for (b2, c), _ in direct.items():
            if b == b2 and (a, c) in direct:
                triple_groups.append((ab, direct[(b2, c)], direct[(a, c)]))
    return {
        "golds": golds,
        "twin_groups": twin_groups,
        "triple_groups": triple_groups,
        "item_count": len(golds),
    }


if __name__ == "__main__":
    # Self-check: closed form matches Monte-Carlo; beta gradient couples twins.
    golds = {"q1": "yes", "q2": "no", "q3": "yes"}
    distributions = {
        "q1": {"yes": 0.6, "no": 0.3, "U": 0.1},
        "q2": {"yes": 0.2, "no": 0.7, "U": 0.1},
        "q3": {"yes": 0.55, "no": 0.35, "U": 0.10},
    }
    twins = [("q1", "q2")]
    triples = [("q1", "q2", "q3")]
    weights = {"alpha": 1.0, "beta": 1.0, "gamma": 0.5}
    closed = expected_reward_closed_form(distributions, golds, twins, triples, **weights)

    def make_sampler(dists, seed):
        rng = random.Random(seed)

        def sample(item):
            r = rng.random()
            acc = 0.0
            for label in LABELS:
                acc += dists[item][label]
                if r <= acc:
                    return label
            return LABELS[-1]

        return sample

    estimated = montecarlo_expected_reward(
        make_sampler(distributions, 7),
        list(distributions),
        golds,
        twins,
        triples,
        rollouts=200000,
        seed=11,
        **weights,
    )
    print(json.dumps({"closed_form": closed, "montecarlo": estimated,
                      "abs_gap": abs(closed - estimated)}, indent=2))
    assert abs(closed - estimated) < 0.01, "closed form disagrees with Monte-Carlo"

    # beta=0 reward decomposes: gradient wrt q1 distribution independent of q2.
    def reward_with(q1, beta):
        dists = {"q1": q1, "q2": distributions["q2"], "q3": distributions["q3"]}
        return expected_reward_closed_form(dists, golds, twins if beta else [], [], **{**weights, "beta": beta})

    def numeric_grad(beta):
        base = distributions["q1"]
        grads = {}
        for k in ("yes", "no"):
            eps = 1e-6
            up = dict(base)
            up[k] += eps
            up["U"] = max(0.0, base["U"] - eps)
            grads[k] = (reward_with(up, beta) - reward_with(base, beta)) / eps
        return grads

    g_ce = numeric_grad(beta=0.0)
    g_couple = numeric_grad(beta=1.0)
    # swap q2 distribution: coupled gradient must move, CE gradient must not.
    alt = {"q2": {"yes": 0.7, "no": 0.2, "U": 0.1}}
    saved = distributions["q2"]
    distributions["q2"] = alt["q2"]
    g_ce_alt = numeric_grad(beta=0.0)
    g_couple_alt = numeric_grad(beta=1.0)
    distributions["q2"] = saved
    def close(left, right, tol=1e-9):
        return all(abs(left[k] - right[k]) <= tol for k in left)

    assert close(g_ce, g_ce_alt), "alpha/gamma gradient must be twin-independent"
    assert not close(g_couple, g_couple_alt, tol=1e-4), (
        "beta gradient must depend on the twin distribution"
    )
    print(json.dumps({
        "ce_grad_invariant_to_twin": True,
        "coupled_grad_changes_with_twin": True,
        "ce_grad": g_ce,
        "coupled_grad": g_couple,
        "coupled_grad_alt": g_couple_alt,
    }, indent=2))
    print("OK: alpha/gamma decompose to CE; beta is the only non-CE, twin-coupled signal")
