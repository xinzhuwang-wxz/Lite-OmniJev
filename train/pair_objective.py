"""Gold controlled-claim pair objective; no readout or training-loop integration."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


def _validate_logits(logits: Any, token_ids: Mapping[str, int], name: str) -> None:
    torch = importlib.import_module("torch")
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 1
        or not logits.is_floating_point()
        or logits.numel() < 3
    ):
        raise ValueError(f"{name} requires a full floating vocabulary vector")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{name} contains nonfinite vocabulary logits")
    if not isinstance(token_ids, Mapping) or set(token_ids) != {"yes", "no", "U"}:
        raise ValueError(f"{name} requires exactly yes/no/U token mapping")
    ids = list(token_ids.values())
    if any(type(i) is not int or not 0 <= i < logits.numel() for i in ids):
        raise ValueError(f"{name} token IDs must be integer vocabulary indices")
    if len(set(ids)) != 3:
        raise ValueError(f"{name} yes/no/U token IDs must be unique")


def claim_pair_loss(
    positive_logits: Any,
    negative_logits: Any,
    *,
    positive_token_ids: Mapping[str, int],
    negative_token_ids: Mapping[str, int],
    positive_gold: str,
    negative_gold: str,
    positive_family_id: str,
    negative_family_id: str,
) -> Any:
    """Return finite FP32 softplus(1 - (s_positive - s_negative)), s=yes-no.

    Callers provide original complete vocabulary logits for two controlled Noul
    claims sharing a component family. Explicit gold must be yes then no; this
    helper cannot infer semantic correctness from tensors. U is validated but
    does not enter the pair score. Gradients flow through the FP32 cast, including
    for FP16/BF16 inputs. There is no temperature, classification head, teacher,
    optimizer update, or probability transformation here.
    """
    if positive_gold != "yes" or negative_gold != "no":
        raise ValueError("controlled pair requires explicit positive yes and negative no gold")
    if (
        not isinstance(positive_family_id, str)
        or not positive_family_id.strip()
        or not isinstance(negative_family_id, str)
        or positive_family_id != negative_family_id
    ):
        raise ValueError("controlled pair must share one nonempty component family")
    _validate_logits(positive_logits, positive_token_ids, "positive")
    _validate_logits(negative_logits, negative_token_ids, "negative")
    if positive_logits.device != negative_logits.device:
        raise ValueError("controlled pair logits must share a device")
    torch = importlib.import_module("torch")
    positive = positive_logits.float()
    negative = negative_logits.float()
    score_positive = positive[positive_token_ids["yes"]] - positive[positive_token_ids["no"]]
    score_negative = negative[negative_token_ids["yes"]] - negative[negative_token_ids["no"]]
    margin_loss = 1.0 - (score_positive - score_negative)
    if not bool(torch.isfinite(margin_loss)):
        raise ValueError("nonfinite FP32 pair margin")
    loss = torch.nn.functional.softplus(margin_loss)
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite FP32 pair loss")
    return loss
