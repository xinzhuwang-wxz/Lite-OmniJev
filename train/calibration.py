"""Bounded scalar temperature preparation; no stage acceptance or automatic file discovery."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from data.schema import Question, answer_tokens
from train.readout import Readout, read_probabilities

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Cohort = Literal["text", "image", "screenshot", "audio", "video", "crossmodal"]
Primitive = Literal["choice", "score", "noul"]
MIN_T = 0.05
MAX_T = 20.0


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


class Identity(StrictModel):
    base_revision: Revision
    adapter_weights_sha256: Sha256
    official_template_sha256: Sha256
    answer_token_map_sha256: Sha256
    raw_logits_receipt_sha256: Sha256
    cohort: Cohort
    primitive: Primitive
    seed: Annotated[int, Field(ge=0)]
    code_revision: Revision


class Binding(Identity):
    calibration_data_sha256: Sha256


class LogitRow(StrictModel):
    pool: Literal["calibration"]
    logits_kind: Literal["raw_logits"]
    complete_legal_logits: bool
    options: Annotated[list[str], Field(min_length=2, max_length=20)]
    logits: dict[str, FiniteFloat]
    target: str

    @model_validator(mode="after")
    def validate_readout(self) -> LogitRow:
        if not self.complete_legal_logits:
            raise ValueError("complete legal logits required")
        read_probabilities(self.logits, self.options)
        if self.target not in self.logits:
            raise ValueError("target must be an answer token or U")
        if not math.isfinite((max(self.logits.values()) - min(self.logits.values())) / MIN_T):
            raise ValueError("logit span is unsafe for bounded temperature arithmetic")
        return self


class CalibrationInput(StrictModel):
    schema_version: Literal[1]
    pool: Literal["calibration"]
    identity: Identity
    rows: Annotated[list[LogitRow], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_primitive(self) -> CalibrationInput:
        for row in self.rows:
            validate_options(self.identity.primitive, row.options)
        return self


def validate_options(primitive: Primitive, options: list[str]) -> None:
    question = Question(
        qid="calibration-contract",
        type=primitive,
        instruction="Validate canonical answer tokens",
        options=options if primitive == "choice" else None,
        score_levels=len(options) if primitive == "score" else None,
    )
    if tuple(options) != answer_tokens(question):
        raise ValueError("options differ from canonical primitive answer tokens")


class TemperatureArtifact(StrictModel):
    schema_version: Literal[1]
    method: Literal["bounded_scalar_joint_nll"]
    fitter_implementation_sha256: Sha256
    binding: Binding
    temperature: Annotated[FiniteFloat, Field(ge=MIN_T, le=MAX_T)]
    temperature_min: FiniteFloat
    temperature_max: FiniteFloat
    rows: Annotated[int, Field(ge=1)]
    nll_before: Annotated[FiniteFloat, Field(ge=0)]
    nll_after: Annotated[FiniteFloat, Field(ge=0)]
    stage_acceptance: Literal["not_evaluated"]

    @model_validator(mode="after")
    def no_nll_regression(self) -> TemperatureArtifact:
        if self.temperature_min != MIN_T or self.temperature_max != MAX_T:
            raise ValueError("temperature bounds differ from frozen baseline")
        if self.nll_after > self.nll_before + 1e-12:
            raise ValueError("temperature fit must not worsen calibration NLL")
        return self


def joint_nll(rows: list[LogitRow], temperature: float) -> float:
    """Stable log-sum-exp loss, including U; this does not produce a new probability readout."""
    losses = []
    for row in rows:
        maximum = max(row.logits.values())
        shifted = [(value - maximum) / temperature for value in row.logits.values()]
        losses.append(
            (maximum - row.logits[row.target]) / temperature
            + math.log(math.fsum(math.exp(value) for value in shifted))
        )
    return math.fsum(loss / len(losses) for loss in losses)


def fit_temperature(data: CalibrationInput, data_sha256: str) -> TemperatureArtifact:
    """Deterministic 80-step golden-section fit in inverse temperature (convex NLL)."""
    binding = Binding(**data.identity.model_dump(), calibration_data_sha256=data_sha256)
    left, right = 1 / MAX_T, 1 / MIN_T
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    y1, y2 = joint_nll(data.rows, 1 / x1), joint_nll(data.rows, 1 / x2)
    for _ in range(80):
        if y1 <= y2:
            right, x2, y2 = x2, x1, y1
            x1 = right - ratio * (right - left)
            y1 = joint_nll(data.rows, 1 / x1)
        else:
            left, x1, y1 = x1, x2, y2
            x2 = left + ratio * (right - left)
            y2 = joint_nll(data.rows, 1 / x2)
    candidates = [1.0, MIN_T, MAX_T, 1 / x1, 1 / x2]
    temperature = min(candidates, key=lambda value: joint_nll(data.rows, value))
    return TemperatureArtifact(
        schema_version=1,
        method="bounded_scalar_joint_nll",
        fitter_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binding=binding,
        temperature=temperature,
        temperature_min=MIN_T,
        temperature_max=MAX_T,
        rows=len(data.rows),
        nll_before=joint_nll(data.rows, 1.0),
        nll_after=joint_nll(data.rows, temperature),
        stage_acceptance="not_evaluated",
    )


def save_temperature(artifact: TemperatureArtifact, path: Path) -> None:
    with path.open("x") as stream:
        stream.write(artifact.model_dump_json(indent=2) + "\n")


def load_temperature(path: Path, expected: Binding) -> TemperatureArtifact:
    artifact = TemperatureArtifact.model_validate(json.loads(path.read_text()))
    if artifact.binding != expected:
        raise ValueError("calibration binding mismatch")
    return artifact


def apply_temperature(
    artifact: TemperatureArtifact, expected: Binding, logits: dict[str, float], options: list[str]
) -> Readout:
    if artifact.binding != expected:
        raise ValueError("calibration binding mismatch")
    validate_options(artifact.binding.primitive, options)
    return read_probabilities(logits, options, temperature=artifact.temperature)
