"""Local text-only decision gateway; model validation and production controls are separate gates."""

from __future__ import annotations

import asyncio
import json
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from data.schema import Prefix, Question, answer_tokens
from scripts.check_tokens import OfficialTokenizerAdapter
from train.readout import read_probabilities
from train.template import ChatTokenizer, check_answer_tokens, render_prompt


class InputQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice", "score", "noul"]
    instructions: str = Field(min_length=1, max_length=8192)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def validate_criteria(self) -> InputQuestion:
        if not self.instructions.strip():
            raise ValueError("instructions cannot be blank")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= 20:
                raise ValueError("choice requires 2–20 option IDs mapped to descriptions")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score requires 2–10 ordered descriptions")
        elif self.criteria is not None:
            if not isinstance(self.criteria, dict) or set(self.criteria) != {"true", "false"}:
                raise ValueError("noul criteria must contain true and false descriptions")
        if self.criteria is not None:
            values = (
                list(self.criteria.values()) if isinstance(self.criteria, dict) else self.criteria
            )
            if any(not text.strip() for text in values) or len(
                {text.strip() for text in values}
            ) != len(values):
                raise ValueError("criteria descriptions must be nonempty and distinct")
            if isinstance(self.criteria, dict) and any(not key.strip() for key in self.criteria):
                raise ValueError("option IDs must be nonempty")
        return self


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state: JsonValue
    model: Literal["jev-latest", "lite-omnijev-e4b-dev"] = "lite-omnijev-e4b-dev"
    questions: Annotated[dict[str, InputQuestion], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def bounded_state(self) -> DecisionRequest:
        encoded = json.dumps(self.state, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode()) > 65536:
            raise ValueError("text state exceeds 64 KiB")
        if any(not qid.strip() for qid in self.questions):
            raise ValueError("question IDs cannot be blank")
        return self


@dataclass(frozen=True)
class Settings:
    upstream_url: str | None = None
    upstream_model: str = "google/gemma-4-E4B-it"
    model_version: str = "untrained-e4b-pinned"
    model_dir: Path | None = None
    raw_logprobs_confirmed: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        directory = os.getenv("LITE_MODEL_DIR")
        return cls(
            upstream_url=os.getenv("LITE_UPSTREAM_URL"),
            upstream_model=os.getenv("LITE_UPSTREAM_MODEL", "google/gemma-4-E4B-it"),
            model_version=os.getenv("LITE_MODEL_VERSION", "untrained-e4b-pinned"),
            model_dir=Path(directory) if directory else None,
            raw_logprobs_confirmed=os.getenv("LITE_RAW_LOGPROBS_CONFIRMED") == "1",
        )


def core_question(qid: str, item: InputQuestion) -> tuple[Question, list[str]]:
    data: dict[str, Any] = {"qid": qid, "type": item.type, "instruction": item.instructions}
    labels: list[str]
    if item.type == "choice":
        assert isinstance(item.criteria, dict)
        labels = list(item.criteria)
        data["options"] = list(item.criteria.values())
    elif item.type == "score":
        assert isinstance(item.criteria, list)
        labels = [str(index) for index in range(len(item.criteria))]
        data.update(score_levels=len(labels), score_descriptions=item.criteria)
    else:
        labels = ["yes", "no"]
        if item.criteria:
            data["instruction"] += "\nAnswer meanings: " + json.dumps(
                item.criteria, ensure_ascii=False
            )
    return Question.model_validate(data), labels


def create_app(
    settings: Settings | None = None,
    *,
    tokenizer: ChatTokenizer | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    runtime_tokenizer = tokenizer

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        nonlocal runtime_tokenizer
        if runtime_tokenizer is None and settings.model_dir:
            runtime_tokenizer = OfficialTokenizerAdapter(settings.model_dir)
        async with httpx.AsyncClient(timeout=30, transport=transport) as client:
            application.state.client = client
            yield

    application = FastAPI(
        title="Lite-OmniJev local text gateway",
        description="Local text contracts. Production controls and GPU gates pending.",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "inference_configured": bool(
                settings.upstream_url and runtime_tokenizer and settings.raw_logprobs_confirmed
            ),
            "model_version": settings.model_version,
            "scope": "local-text-only",
            "model_quality_verified": False,
        }

    async def decide(
        qid: str, item: InputQuestion, prefix: Prefix, client: httpx.AsyncClient
    ) -> tuple[str, dict[str, Any], int, int]:
        assert runtime_tokenizer is not None and settings.upstream_url is not None
        question, labels = core_question(qid, item)
        try:
            token_ids = check_answer_tokens(runtime_tokenizer, question)
            prompt = render_prompt(runtime_tokenizer, prefix, question)
        except ValueError as error:
            raise HTTPException(503, "Official tokenizer/template preflight failed") from error
        try:
            response = await client.post(
                settings.upstream_url.rstrip("/") + "/v1/completions",
                json={
                    "model": settings.upstream_model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "logprobs": 1,
                    "logprob_token_ids": list(token_ids.values()),
                    "return_tokens_as_token_ids": True,
                },
            )
            response.raise_for_status()
            body = response.json()
            raw = body["choices"][0]["logprobs"]["top_logprobs"][0]
            logits: dict[str, float] = {}
            for token, token_id in token_ids.items():
                value = raw[f"token_id:{token_id}"]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= -9999
                    or value > 0
                ):
                    raise ValueError("non-finite or clipped raw logprob")
                logits[token] = float(value)
            count = body["usage"]["prompt_tokens"]
            generated = body["usage"]["completion_tokens"]
            if type(count) is not int or count < 0 or type(generated) is not int or generated < 0:
                raise ValueError("invalid upstream token accounting")
        except httpx.TimeoutException as error:
            raise HTTPException(504, "Model backend timed out; retry when it is ready") from error
        except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError) as error:
            raise HTTPException(
                502, "Backend did not return complete raw answer logprobs"
            ) from error
        readout = read_probabilities(logits, answer_tokens(question))
        values = list(readout.probabilities.values())
        probabilities = dict(zip(labels, values, strict=True))
        result: dict[str, Any] = {
            "type": item.type,
            "probabilities": probabilities,
            "confidence": readout.confidence,
            "p_insufficient": readout.p_insufficient,
            "calibration": "unfitted",
            "confidence_semantics": "normalized_peak_non_U",
        }
        if item.type == "choice":
            result["choice"] = labels[values.index(max(values))]
        elif item.type == "noul":
            result["noul"] = probabilities["yes"]
        else:
            assert isinstance(item.criteria, list)
            result["score"] = sum(index * probability for index, probability in enumerate(values))
            result["legend"] = {str(i): description for i, description in enumerate(item.criteria)}
        return qid, result, count, generated

    @application.post("/v1/systemone")
    async def systemone(request: DecisionRequest) -> dict[str, Any]:
        if (
            not settings.upstream_url
            or runtime_tokenizer is None
            or not settings.raw_logprobs_confirmed
        ):
            raise HTTPException(
                503, "Configure a pinned tokenizer and vLLM backend with verified raw_logprobs mode"
            )
        state = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(
                request.state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        if not state.strip():
            raise HTTPException(422, "state cannot be blank")
        prefix = Prefix(state=state)
        results = await asyncio.gather(
            *(
                decide(qid, item, prefix, application.state.client)
                for qid, item in request.questions.items()
            )
        )
        return {
            "model": settings.model_version,
            "answers": {qid: answer for qid, answer, _, _ in results},
            "usage": {"input_tokens": sum(count for _, _, count, _ in results), "output_tokens": 0},
            "backend_usage": {
                "input_tokens": sum(count for _, _, count, _ in results),
                "output_tokens": sum(generated for _, _, _, generated in results),
            },
            "validation_status": "GPU parity and calibration pending",
        }

    return application


app = create_app()
