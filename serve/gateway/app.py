"""Local decision gateway; CPU media integration does not establish model quality."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from starlette.responses import Response

from data.schema import Media, Prefix, Question, answer_tokens
from scripts.check_tokens import OfficialTokenizerAdapter
from serve.vllm_chat import VllmChatClient, VllmChatProtocolError
from train.multimodal import prepare_decision
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
    media: list[Media] = Field(default_factory=list, max_length=8)
    video_frames: Literal[8, 16] = 16
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
    media_root: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        directory = os.getenv("LITE_MODEL_DIR")
        media_root = os.getenv("LITE_MEDIA_ROOT")
        return cls(
            upstream_url=os.getenv("LITE_UPSTREAM_URL"),
            upstream_model=os.getenv("LITE_UPSTREAM_MODEL", "google/gemma-4-E4B-it"),
            model_version=os.getenv("LITE_MODEL_VERSION", "untrained-e4b-pinned"),
            model_dir=Path(directory) if directory else None,
            raw_logprobs_confirmed=os.getenv("LITE_RAW_LOGPROBS_CONFIRMED") == "1",
            media_root=Path(media_root) if media_root else None,
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
    processor: Any = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    runtime_tokenizer = tokenizer
    runtime_processor = processor
    canonical_mode = settings.media_root is not None
    preparation_lock = asyncio.Semaphore(1)
    preparation_jobs: set[asyncio.Task[Any]] = set()
    shutting_down = False

    def preparation_finished(job: asyncio.Task[Any]) -> None:
        preparation_jobs.discard(job)
        if not job.cancelled():
            # Retrieve failures even when the HTTP waiter has disconnected; never log content.
            job.exception()

    def trusted_root_configured() -> bool:
        root = settings.media_root
        return bool(
            root is not None
            and root.is_dir()
            and not root.is_symlink()
            and root.absolute() == root.resolve()
        )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        nonlocal runtime_tokenizer, runtime_processor, shutting_down
        if runtime_tokenizer is None and settings.model_dir and not canonical_mode:
            try:
                runtime_tokenizer = OfficialTokenizerAdapter(settings.model_dir)
            except (ImportError, OSError, ValueError):
                runtime_tokenizer = None
        if (
            canonical_mode
            and trusted_root_configured()
            and runtime_processor is None
            and settings.model_dir
        ):
            try:
                transformers = importlib.import_module("transformers")
                runtime_processor = transformers.AutoProcessor.from_pretrained(
                    settings.model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                )
            except Exception:
                runtime_processor = None
        async with httpx.AsyncClient(timeout=30, transport=transport) as client:
            application.state.client = client
            chat_client = (
                VllmChatClient(
                    settings.upstream_url,
                    settings.upstream_model,
                    raw_logprobs_confirmed=True,
                    transport=transport,
                )
                if settings.upstream_url and settings.raw_logprobs_confirmed and canonical_mode
                else None
            )
            application.state.chat_client = chat_client
            try:
                yield
            finally:
                shutting_down = True
                # Jobs own the CPU lock until their actual worker thread has finished.
                shutdown_cancelled = False
                if preparation_jobs:
                    drain = asyncio.gather(*tuple(preparation_jobs), return_exceptions=True)
                    while not drain.done():
                        try:
                            await asyncio.shield(drain)
                        except asyncio.CancelledError:
                            shutdown_cancelled = True
                if chat_client is not None:
                    await chat_client.aclose()
                if shutdown_cancelled:
                    raise asyncio.CancelledError

    application = FastAPI(
        title="Lite-OmniJev local text gateway",
        description="Local development contracts. Production controls and GPU gates pending.",
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def server_timing(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        request.state.timings = {"preprocess": 0.0, "upstream": 0.0, "queue": 0.0}
        response = await call_next(request)
        phases = {**request.state.timings, "total": time.perf_counter() - started}
        response.headers["Server-Timing"] = ", ".join(
            f"{label};dur={max(0.0, phases[label]) * 1000:.3f}"
            for label in ("preprocess", "upstream", "total", "queue")
        )
        return response

    @application.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        # Pydantic's default input/context fields may contain the original state or media URI.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {
                        "type": "value_error",
                        "loc": ["body"],
                        "msg": "Invalid decision request fields",
                    }
                ]
            },
        )

    @application.get("/health")
    def health() -> dict[str, Any]:
        configured = bool(
            settings.upstream_url
            and settings.raw_logprobs_confirmed
            and (
                runtime_processor is not None and trusted_root_configured()
                if canonical_mode
                else runtime_tokenizer is not None
            )
        )
        return {
            "status": "ok",
            "inference_configured": configured,
            "model_version": settings.model_version,
            "mode": "canonical-chat" if canonical_mode else "legacy-text-completions",
            "scope": "local-canonical-chat-development" if canonical_mode else "local-text-only",
            "model_quality_verified": False,
        }

    def format_answer(
        item: InputQuestion, question: Question, labels: list[str], logits: dict[str, float]
    ) -> dict[str, Any]:
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
        return result

    async def decide_canonical(
        qid: str,
        item: InputQuestion,
        prefix: Prefix,
        video_frames: Literal[8, 16],
        timings: dict[str, float],
    ) -> tuple[str, dict[str, Any], int, int]:
        question, labels = core_question(qid, item)
        media_root = settings.media_root
        assert media_root is not None

        queued_at = time.perf_counter()
        one_timing = {"preprocess": 0.0, "queue": 0.0}

        def prepare() -> Any:
            started = time.perf_counter()
            one_timing["queue"] = started - queued_at
            try:
                try:
                    check_answer_tokens(runtime_processor.tokenizer, question)
                except Exception:
                    raise HTTPException(
                        503, "Official processor/tokenizer preflight unavailable"
                    ) from None
                try:
                    prepared = prepare_decision(
                        prefix,
                        question,
                        runtime_processor,
                        media_root,
                        profile="canonical_http",
                        video_frames=video_frames,
                    )
                    # HTTP needs canonical bytes and exact IDs, not duplicate HF media tensors.
                    return replace(prepared, inputs={"input_ids": prepared.inputs["input_ids"]})
                except Exception:
                    raise HTTPException(
                        422, "Invalid media or unsupported S0 input combination"
                    ) from None
            finally:
                one_timing["preprocess"] = time.perf_counter() - started

        async def preparation_job() -> Any:
            async with preparation_lock:
                try:
                    return await asyncio.to_thread(prepare)
                finally:
                    timings["preprocess"] += one_timing["preprocess"]
                    timings["queue"] += one_timing["queue"]

        if shutting_down:
            raise HTTPException(503, "Gateway is shutting down")
        job = asyncio.create_task(preparation_job())
        preparation_jobs.add(job)
        job.add_done_callback(preparation_finished)
        # Shield the lock-owning task, not merely the thread await inside a request-owned lock.
        prepared = await asyncio.shield(job)
        upstream_started = time.perf_counter()
        try:
            raw = await application.state.chat_client.read_logprobs(prepared)
        except VllmChatProtocolError as error:
            if str(error) == "upstream_timeout":
                raise HTTPException(
                    504, "Model backend timed out; retry when it is ready"
                ) from None
            raise HTTPException(
                502, "Backend did not return aligned complete raw logprobs"
            ) from None
        finally:
            timings["upstream"] += time.perf_counter() - upstream_started
        return (
            qid,
            format_answer(item, question, labels, raw.raw_logprobs),
            raw.usage["prompt_tokens"],
            raw.usage["completion_tokens"],
        )

    async def decide(
        qid: str,
        item: InputQuestion,
        prefix: Prefix,
        client: httpx.AsyncClient,
        timings: dict[str, float],
    ) -> tuple[str, dict[str, Any], int, int]:
        assert runtime_tokenizer is not None and settings.upstream_url is not None
        preprocessing_started = time.perf_counter()
        question, labels = core_question(qid, item)
        try:
            token_ids = check_answer_tokens(runtime_tokenizer, question)
            prompt = render_prompt(runtime_tokenizer, prefix, question)
        except ValueError as error:
            raise HTTPException(503, "Official tokenizer/template preflight failed") from error
        finally:
            timings["preprocess"] += time.perf_counter() - preprocessing_started
        upstream_started = time.perf_counter()
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
        finally:
            timings["upstream"] += time.perf_counter() - upstream_started
        return qid, format_answer(item, question, labels, logits), count, generated

    @application.post("/v1/systemone")
    async def systemone(request: DecisionRequest, http_request: Request) -> dict[str, Any]:
        use_canonical = canonical_mode or bool(request.media)
        if not settings.upstream_url or not settings.raw_logprobs_confirmed:
            raise HTTPException(503, "Configure a pinned backend with verified raw_logprobs mode")
        if use_canonical and (
            runtime_processor is None
            or not trusted_root_configured()
            or application.state.chat_client is None
        ):
            raise HTTPException(503, "Configure a trusted media root and pinned local processor")
        if not use_canonical and runtime_tokenizer is None:
            raise HTTPException(503, "Configure a pinned tokenizer and verified text backend")
        state = (
            request.state
            if isinstance(request.state, str)
            else json.dumps(
                request.state, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        )
        if not state.strip():
            raise HTTPException(422, "state cannot be blank")
        prefix = Prefix(state=state, media=request.media)
        if use_canonical:
            # Validate all question contracts before making any upstream request.
            for qid, item in request.questions.items():
                core_question(qid, item)
            results = []
            for qid, item in request.questions.items():
                results.append(
                    await decide_canonical(
                        qid, item, prefix, request.video_frames, http_request.state.timings
                    )
                )
        else:
            results = await asyncio.gather(
                *(
                    decide(qid, item, prefix, application.state.client, http_request.state.timings)
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
