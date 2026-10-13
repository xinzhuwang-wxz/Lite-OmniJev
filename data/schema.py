"""Strict, versioned data contract shared by builders, training and evaluation."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Pool = Literal["train", "dev", "calibration", "holdout"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class HardTarget(Contract):
    kind: Literal["hard"]
    value: str | int
    source: Nonempty


class SoftTarget(Contract):
    kind: Literal["soft"]
    probs: list[float] = Field(min_length=2, max_length=20)
    source: Nonempty

    @model_validator(mode="after")
    def distribution(self) -> Self:
        if any(p < 0 or p > 1 for p in self.probs):
            raise ValueError("soft probabilities must lie in [0, 1]")
        if not math.isclose(math.fsum(self.probs), 1.0, abs_tol=1e-9):
            raise ValueError("soft probabilities must sum to 1; U is implicit zero")
        return self


class InsufficientTarget(Contract):
    kind: Literal["insufficient"]


Target = Annotated[HardTarget | SoftTarget | InsufficientTarget, Field(discriminator="kind")]


class Question(Contract):
    qid: Nonempty
    type: Literal["choice", "score", "noul"]
    instruction: Nonempty
    options: list[Nonempty] | None = None
    score_levels: int | None = None
    score_descriptions: list[Nonempty] | None = None
    target: Target | None = None
    pair_of: Nonempty | None = None

    @model_validator(mode="after")
    def legal_answers(self) -> Self:
        if self.type == "choice":
            if self.options is None or not 2 <= len(self.options) <= 20:
                raise ValueError("choice requires 2–20 options")
            if len(set(self.options)) != len(self.options):
                raise ValueError("choice options must be distinct")
            if self.score_levels is not None or self.score_descriptions is not None:
                raise ValueError("choice cannot specify score levels or descriptions")
        elif self.type == "score":
            if self.score_levels is None or not 2 <= self.score_levels <= 10:
                raise ValueError("score requires explicit score_levels between 2 and 10")
            if self.options is not None:
                raise ValueError("score cannot specify options")
            if (
                self.score_descriptions is not None
                and len(self.score_descriptions) != self.score_levels
            ):
                raise ValueError("score_descriptions must match score_levels")
        elif (
            self.options is not None
            or self.score_levels is not None
            or self.score_descriptions is not None
        ):
            raise ValueError("noul has fixed yes/no answers")
        tokens = answer_tokens(self)
        if isinstance(self.target, SoftTarget) and len(self.target.probs) != len(tokens):
            raise ValueError("soft target length must match real answers, excluding U")
        if isinstance(self.target, HardTarget):
            value = self.target.value
            if self.type == "score":
                if type(value) is not int or str(value) not in tokens:
                    raise ValueError("score hard target must be an integer in range")
            elif type(value) is not str or value not in tokens:
                raise ValueError("hard target must name a legal answer token")
        return self


def answer_tokens(question: Question) -> tuple[str, ...]:
    """Canonical answer order; U is separate and never a real answer."""
    if question.type == "choice":
        return tuple(chr(65 + i) for i in range(len(question.options or [])))
    if question.type == "score":
        return tuple(str(i) for i in range(question.score_levels or 0))
    return ("yes", "no")


class Media(Contract):
    type: Literal["image", "audio", "video"]
    uri: Nonempty
    sha256: Sha256


class Prefix(Contract):
    state: Nonempty
    media: list[Media] = Field(default_factory=list)


class Metadata(Contract):
    modality: list[Literal["text", "image", "audio", "video"]] = Field(min_length=1)
    source_dataset: Nonempty
    source_revision: Nonempty
    source_sha256: Sha256
    license: Nonempty
    usage: Literal["research", "commercial"]
    pool: Pool
    family_id: Nonempty
    data_version: Nonempty

    @model_validator(mode="after")
    def declared_provenance(self) -> Self:
        if self.license.casefold() in {"unknown", "n/a", "none", "unlicensed"}:
            raise ValueError("a known license is required")
        if len(set(self.modality)) != len(self.modality):
            raise ValueError("modalities must be distinct")
        return self


class Sample(Contract):
    id: Nonempty
    prefix: Prefix
    questions: list[Question] = Field(min_length=1)
    meta: Metadata

    @model_validator(mode="after")
    def sample_integrity(self) -> Self:
        qids = {q.qid for q in self.questions}
        if len(qids) != len(self.questions):
            raise ValueError("question IDs must be unique within a sample")
        for question in self.questions:
            if question.target is None:
                raise ValueError("dataset samples require a target for every question")
            if question.pair_of is not None:
                if question.pair_of not in qids or question.pair_of == question.qid:
                    raise ValueError("pair_of must reference a different question in this sample")
        if any(m.type not in self.meta.modality for m in self.prefix.media):
            raise ValueError("media types must be declared in metadata modality")
        return self


def validate_pool_isolation(samples: list[Sample]) -> None:
    """Refuse family leakage and repeated sample IDs across an assembled corpus."""
    families: dict[str, Pool] = {}
    ids: set[str] = set()
    for sample in samples:
        if sample.id in ids:
            raise ValueError(f"duplicate sample ID: {sample.id}")
        ids.add(sample.id)
        family = sample.meta.family_id
        if family in families and families[family] != sample.meta.pool:
            raise ValueError(f"sample family crosses pools: {family}")
        families[family] = sample.meta.pool
