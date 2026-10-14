"""Strict S0 numerical comparison, separate from model/source or quality admission."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Self

from pydantic import Field, model_validator

from data.schema import Contract, Sha256
from train.readout import read_probabilities

THRESHOLD = 1e-3


def _derive(token_ids: list[int], logprobs: list[float]) -> dict[str, Any]:
    if (
        not 3 <= len(token_ids) <= 21
        or len(logprobs) != len(token_ids)
        or len(set(token_ids)) != len(token_ids)
        or any(type(value) is not int or value < 0 for value in token_ids)
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value > 0
            or value <= -9999
            for value in logprobs
        )
    ):
        raise ValueError("complete unique answer IDs and finite unclipped raw logprobs required")
    raw = [math.exp(value) for value in logprobs]
    if math.fsum(raw) > 1 + 1e-6:
        raise ValueError("answer mass cannot exceed the full vocabulary probability mass")
    labels = [str(token) for token in token_ids[:-1]]
    readout = read_probabilities(dict(zip([*labels, "U"], logprobs, strict=True)), labels)
    return {
        "raw_probabilities": raw,
        "joint": list(readout.joint_probabilities.values()),
        "conditional": list(readout.probabilities.values()),
        "p_U": readout.p_insufficient,
    }


class RawReadout(Contract):
    """One generation position; answer IDs are ordered with U last."""

    id_sha256: Sha256
    input_sha256: Sha256
    media_sha256: list[Sha256]
    token_ids: list[int] = Field(min_length=3, max_length=21)
    raw_logprobs: list[float] = Field(min_length=3, max_length=21)
    raw_probabilities: list[float]
    joint: list[float]
    conditional: list[float]
    p_U: float

    @model_validator(mode="after")
    def recompute_distributions(self) -> Self:
        for name, expected in _derive(self.token_ids, self.raw_logprobs).items():
            if getattr(self, name) != expected:
                raise ValueError("derived probability distribution does not match raw logprobs")
        return self


def make_row(
    *,
    id_sha256: str,
    input_sha256: str,
    media_sha256: list[str],
    token_ids: list[int],
    raw_logprobs: list[float],
) -> RawReadout:
    return RawReadout.model_validate(
        {
            "id_sha256": id_sha256,
            "input_sha256": input_sha256,
            "media_sha256": media_sha256,
            "token_ids": token_ids,
            "raw_logprobs": raw_logprobs,
            **_derive(token_ids, raw_logprobs),
        }
    )


def _cohort(rows: Sequence[RawReadout | dict[str, Any]]) -> list[RawReadout]:
    if len(rows) != 100:
        raise ValueError("exactly 100 rows required")
    validated = [
        RawReadout.model_validate(row.model_dump() if isinstance(row, RawReadout) else row)
        for row in rows
    ]
    if len({row.id_sha256 for row in validated}) != 100:
        raise ValueError("100 unique sample identities required")
    return validated


def compare_rows(
    reference: Sequence[RawReadout | dict[str, Any]],
    candidate: Sequence[RawReadout | dict[str, Any]],
    *,
    exact: bool = False,
) -> dict[str, Any]:
    left, right = _cohort(reference), _cohort(candidate)
    maxima = dict.fromkeys(("raw_probabilities", "joint", "conditional", "p_U"), 0.0)
    identical = True
    for expected, actual in zip(left, right, strict=True):
        if any(
            getattr(expected, name) != getattr(actual, name)
            for name in ("id_sha256", "input_sha256", "media_sha256", "token_ids")
        ):
            raise ValueError("cohort input/media/token identity mismatch")
        for name in ("raw_probabilities", "joint", "conditional"):
            maxima[name] = max(
                maxima[name],
                *(
                    abs(a - b)
                    for a, b in zip(getattr(expected, name), getattr(actual, name), strict=True)
                ),
            )
        maxima["p_U"] = max(maxima["p_U"], abs(expected.p_U - actual.p_U))
        identical = identical and expected.model_dump() == actual.model_dump()
    passed = all(value < THRESHOLD for value in maxima.values()) and (not exact or identical)
    return {
        "status": "passed" if passed else "failed",
        "exit_code": 0 if passed else 1,
        "samples": 100,
        "threshold_exclusive": THRESHOLD,
        "max_differences": maxima,
        "exact_rerun_required": exact,
        "rows_identical": identical,
        "release_or_stage_acceptance": False,
        "scope": "Numerical only; source/model/runtime/GPU provenance required separately",
    }
