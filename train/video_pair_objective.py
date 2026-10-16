"""CPU-prepared cross-video ranking objective; no trainer or inference integration."""

from __future__ import annotations

import importlib
import re
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


def video_pair_loss(
    positive_logits: Any,
    negative_logits: Any,
    *,
    positive_token_ids: Mapping[str, int],
    negative_token_ids: Mapping[str, int],
    positive_gold: str,
    negative_gold: str,
    positive_family_id: str,
    negative_family_id: str,
    match_binding: Mapping[str, str],
) -> Any:
    """Return FP32 softplus(-(d_positive-d_negative)), where d=z_yes-z_no.

    Explicit gold is yes then no on distinct TRAIN video families. match_binding
    requires an opaque nonempty pair_id and positive/negative_question_sha256 and
    positive/negative_program_sha256; each pair of lowercase SHA256 hashes must
    match. This checks declared identity, not source truth: the caller must verify
    the hashes against original question/nonmedia input and official programs,
    TRAIN ledgers, opposite official gold and distinct media SHA before calling.
    Hashes/IDs are metadata only, never model input. No label or U is inferred.

    This is one conditional assignment term, not two bidirectional terms. U and
    unrelated vocabulary logits receive no pair gradient. Keep joint legal CE
    outside this helper to anchor absolute probabilities and U; the caller owns
    weight/normalization, checkpoint identity and matched exposure. No probability
    readout, margin parameter, model update or existing same-family guard changes.
    """
    if positive_gold != "yes" or negative_gold != "no":
        raise ValueError("video pair requires explicit positive yes and negative no gold")
    if (
        not isinstance(positive_family_id, str)
        or not positive_family_id.strip()
        or not isinstance(negative_family_id, str)
        or not negative_family_id.strip()
        or positive_family_id == negative_family_id
    ):
        raise ValueError("video pair requires distinct nonempty family IDs")
    keys = {
        "pair_id",
        "positive_question_sha256",
        "negative_question_sha256",
        "positive_program_sha256",
        "negative_program_sha256",
    }
    if not isinstance(match_binding, Mapping) or set(match_binding) != keys:
        raise ValueError("video pair requires complete question/program match binding")
    pair_id = match_binding["pair_id"]
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise ValueError("video pair requires a nonempty opaque pair_id")
    for kind in ("question", "program"):
        left = match_binding[f"positive_{kind}_sha256"]
        right = match_binding[f"negative_{kind}_sha256"]
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or re.fullmatch(r"[0-9a-f]{64}", left) is None
            or left != right
        ):
            raise ValueError(f"video pair requires matching {kind} SHA256 hashes")
    torch = importlib.import_module("torch")
    if (
        isinstance(positive_logits, torch.Tensor)
        and isinstance(negative_logits, torch.Tensor)
        and positive_logits.device != negative_logits.device
    ):
        raise ValueError("video pair logits must share a device")
    _validate_logits(positive_logits, positive_token_ids, "positive")
    _validate_logits(negative_logits, negative_token_ids, "negative")
    positive = positive_logits.float()
    negative = negative_logits.float()
    d_positive = positive[positive_token_ids["yes"]] - positive[positive_token_ids["no"]]
    d_negative = negative[negative_token_ids["yes"]] - negative[negative_token_ids["no"]]
    delta = d_positive - d_negative
    if not bool(torch.isfinite(delta)):
        raise ValueError("nonfinite FP32 video pair difference")
    loss = torch.nn.functional.softplus(-delta)
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite FP32 video pair loss")
    return loss
