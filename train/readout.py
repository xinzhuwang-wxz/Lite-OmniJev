"""Read complete legal-token logits; never pass truncated/sampled top-k output."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Readout:
    probabilities: dict[str, float]
    p_insufficient: float
    confidence: float
    prediction: str
    joint_probabilities: dict[str, float]


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    maximum = max(values)
    # Subtract before division so tiny temperatures cannot overflow positive logits.
    weights = [math.exp((x - maximum) / temperature) for x in values]
    total = math.fsum(weights)
    return [x / total for x in weights]


def read_probabilities(
    logits: Mapping[str, float], options: Sequence[str], *, temperature: float = 1.0
) -> Readout:
    """Normalize U jointly, then answers conditionally; tie-break in option order.

    Callers must supply every legal token's original logit from the same position.
    Provenance cannot be recovered from numbers: backends must reject top-k output.
    Temperature is an explicit calibrated scalar, not a sampler transformation.
    """
    if isinstance(options, str) or not 2 <= len(options) <= 20:
        raise ValueError("options must contain 2–20 distinct tokens")
    if any(not isinstance(x, str) or not x for x in options):
        raise ValueError("answer tokens must be nonempty strings")
    if len(set(options)) != len(options) or "U" in options:
        raise ValueError("answer tokens must be distinct and exclude U")
    if set(logits) != {*options, "U"}:
        raise ValueError("logits must contain exactly all legal answers plus U")
    values = [logits[x] for x in (*options, "U")]
    if any(
        isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x)
        for x in values
    ):
        raise ValueError("logits must be finite numbers")
    if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    joint = _softmax(values, temperature)
    # A second softmax avoids 0/0 when U dominates and all answer masses underflow.
    conditional = _softmax(values[:-1], temperature)
    maximum = max(conditional)
    k = len(options)
    return Readout(
        dict(zip(options, conditional, strict=True)),
        joint[-1],
        min(1.0, max(0.0, (maximum - 1 / k) / (1 - 1 / k))),
        options[conditional.index(maximum)],
        dict(zip((*options, "U"), joint, strict=True)),
    )
