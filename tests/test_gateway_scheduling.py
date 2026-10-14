"""Public API scheduling behavior; fake preparation/readout, never GPU performance."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from serve.gateway.app import Settings, create_app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class PreparedFixture:
    question: Any
    inputs: dict[str, Any]


@pytest.mark.parametrize("limit", [1, 3])
async def test_api_primes_first_then_bounds_and_orders_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: int
) -> None:
    import serve.gateway.app as gateway

    monkeypatch.setattr(gateway, "check_answer_tokens", lambda *args: {})
    monkeypatch.setattr(
        gateway,
        "prepare_decision",
        lambda prefix, question, *args, **kwargs: PreparedFixture(question, {"input_ids": [[1]]}),
    )
    active = maximum = 0
    first_done = False
    starts: list[str] = []
    completions: list[str] = []
    wave = asyncio.Event()

    async def read(self: Any, prepared: PreparedFixture) -> Any:
        nonlocal active, maximum, first_done
        qid = prepared.question.qid
        starts.append(qid)
        if qid == "q0":
            first_done = True
        else:
            assert first_done
            active += 1
            maximum = max(maximum, active)
            if active >= limit:
                wave.set()
            await asyncio.wait_for(wave.wait(), 2)
            await asyncio.sleep((8 - int(qid[1:])) * 0.001)
            active -= 1
        completions.append(qid)
        values = {"A": -1.0, "B": -2.0, "U": -3.0}
        if int(qid[1:]) % 2:
            values["A"], values["B"] = values["B"], values["A"]
        return SimpleNamespace(
            raw_logprobs=values, usage={"prompt_tokens": 1, "completion_tokens": 1}
        )

    monkeypatch.setattr(gateway.VllmChatClient, "read_logprobs", read)
    app = create_app(
        Settings(
            upstream_url="http://fixture",
            media_root=tmp_path,
            raw_logprobs_confirmed=True,
            question_concurrency=limit,
        ),
        processor=SimpleNamespace(tokenizer=object()),
    )
    questions = {
        f"q{i}": {
            "type": "choice",
            "instructions": f"Question {i}",
            "criteria": {"left": "L", "right": "R"},
        }
        for i in range(8)
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://fixture"
        ) as client:
            response = await client.post(
                "/v1/systemone", json={"state": "Public state", "questions": questions}
            )
    assert response.status_code == 200
    body = response.json()
    assert list(body["answers"]) == list(questions)
    assert [body["answers"][f"q{i}"]["choice"] for i in range(8)] == [
        "right" if i % 2 else "left" for i in range(8)
    ]
    assert body["backend_usage"] == {"input_tokens": 8, "output_tokens": 8}
    assert starts[0] == completions[0] == "q0"
    assert maximum == limit
    fields = {x.strip().split(";dur=")[0] for x in response.headers["server-timing"].split(",")}
    if limit == 1:
        assert completions == list(questions)
        assert fields == {"preprocess", "upstream", "queue", "total"}
    else:
        assert fields == {"preprocess_work", "upstream_work", "queue_work", "total"}
        assert response.headers["x-lite-execution-mode"] == "bounded-scheduled-v1"
        assert response.headers["x-lite-question-concurrency"] == str(limit)


@pytest.mark.parametrize("value", [0, 11, True, 1.5])
def test_invalid_concurrency_rejected_before_server_start(value: Any) -> None:
    with pytest.raises(ValueError):
        Settings(question_concurrency=value)


async def test_partial_upstream_failure_cancels_siblings_without_partial_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import serve.gateway.app as gateway

    monkeypatch.setattr(gateway, "check_answer_tokens", lambda *args: {})
    monkeypatch.setattr(
        gateway,
        "prepare_decision",
        lambda prefix, question, *args, **kwargs: PreparedFixture(question, {"input_ids": [[1]]}),
    )
    sibling_started = asyncio.Event()
    active = 0
    cancelled = 0

    async def read(self: Any, prepared: PreparedFixture) -> Any:
        nonlocal active, cancelled
        if prepared.question.qid == "q0":
            return SimpleNamespace(
                raw_logprobs={"yes": -1.0, "no": -2.0, "U": -3.0},
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )
        if prepared.question.qid == "q1":
            await asyncio.wait_for(sibling_started.wait(), 2)
            raise gateway.VllmChatProtocolError("PRIVATE failure")
        active += 1
        sibling_started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled += 1
            raise
        finally:
            active -= 1

    monkeypatch.setattr(gateway.VllmChatClient, "read_logprobs", read)
    app = create_app(
        Settings(
            upstream_url="http://fixture",
            media_root=tmp_path,
            raw_logprobs_confirmed=True,
            question_concurrency=2,
        ),
        processor=SimpleNamespace(tokenizer=object()),
    )
    payload = {
        "state": "Public state",
        "questions": {f"q{i}": {"type": "noul", "instructions": f"Question {i}"} for i in range(6)},
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://fixture"
        ) as client:
            response = await asyncio.wait_for(client.post("/v1/systemone", json=payload), 3)
    assert response.status_code == 502
    assert "answers" not in response.json() and "PRIVATE" not in response.text
    assert active == 0 and cancelled >= 1


async def test_image_reuse_is_created_once_per_request_and_never_shared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys
    import types

    import serve.gateway.app as gateway

    original = SimpleNamespace(tokenizer=object())
    copies: list[Any] = []
    prepared_processors: list[Any] = []
    module = types.ModuleType("train.request_media_cache")

    def factory(processor: Any) -> Any:
        assert processor is original
        private = SimpleNamespace(tokenizer=original.tokenizer)
        copies.append(private)
        return private

    module.__dict__["request_image_processor"] = factory
    monkeypatch.setitem(sys.modules, "train.request_media_cache", module)
    monkeypatch.setattr(gateway, "check_answer_tokens", lambda *args: {})

    def prepare(prefix: Any, question: Any, processor: Any, *args: Any, **kwargs: Any) -> Any:
        prepared_processors.append(processor)
        return PreparedFixture(question, {"input_ids": [[1]]})

    async def read(self: Any, prepared: Any) -> Any:
        return SimpleNamespace(
            raw_logprobs={"yes": -1.0, "no": -2.0, "U": -3.0},
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )

    monkeypatch.setattr(gateway, "prepare_decision", prepare)
    monkeypatch.setattr(gateway.VllmChatClient, "read_logprobs", read)
    app = create_app(
        Settings(
            upstream_url="http://fixture",
            media_root=tmp_path,
            raw_logprobs_confirmed=True,
            question_concurrency=3,
            request_image_reuse=True,
        ),
        processor=original,
    )
    payload = {
        "state": "Public state",
        "media": [{"type": "image", "uri": "public.png", "sha256": "0" * 64}],
        "questions": {
            f"q{i}": {"type": "noul", "instructions": "Public question"} for i in range(3)
        },
    }
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://fixture"
        ) as client:
            assert (await client.post("/v1/systemone", json=payload)).status_code == 200
            assert (await client.post("/v1/systemone", json=payload)).status_code == 200
    assert len(copies) == 2 and copies[0] is not copies[1]
    assert all(p is copies[0] for p in prepared_processors[:3])
    assert all(p is copies[1] for p in prepared_processors[3:])
