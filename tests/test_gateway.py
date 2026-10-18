import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from serve.gateway.app import Settings, create_app


class FixtureTokenizer:
    """Explicit transport fixture, never a model quality test."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [{"A": 1, "B": 2, "U": 3, "yes": 4, "no": 5, "0": 6, "1": 7}[text]]

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs["enable_thinking"] is False
        return "<official-template-fixture>" + conversation[0]["content"]


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


REQUEST = {
    "state": "Order was shipped",
    "questions": {
        "shipped": {
            "type": "choice",
            "instructions": "Shipment status?",
            "criteria": {"sent": "Shipped", "waiting": "Waiting"},
        },
        "yesno": {"type": "noul", "instructions": "Was it shipped?"},
        "score": {"type": "score", "instructions": "Progress?", "criteria": ["Not sent", "Sent"]},
    },
}


def response_for(request: httpx.Request, bad: str | None = None) -> httpx.Response:
    payload = json.loads(request.content)
    assert request.url.path == "/v1/completions"
    assert payload["temperature"] == 1
    assert payload["max_tokens"] == 1
    assert payload["return_tokens_as_token_ids"] is True
    assert "allowed_token_ids" not in payload
    ids = payload["logprob_token_ids"]
    logits = {f"token_id:{token_id}": -float(index + 1) for index, token_id in enumerate(ids)}
    if bad == "missing":
        logits.pop(f"token_id:{ids[-1]}")
    if bad == "clipped":
        logits[f"token_id:{ids[0]}"] = -9999.0
    if bad == "positive":
        logits[f"token_id:{ids[0]}"] = 1.0
    if bad == "timeout":
        raise httpx.ReadTimeout("fixture")
    return httpx.Response(
        200,
        json={
            "choices": [{"logprobs": {"top_logprobs": [logits]}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1},
        },
    )


@asynccontextmanager
async def client_for(
    bad: str | None = None, *, diagnostics: bool = False
) -> AsyncIterator[httpx.AsyncClient]:
    application = create_app(
        Settings(
            upstream_url="http://fixture",
            raw_logprobs_confirmed=True,
            internal_readout_diagnostics=diagnostics,
        ),
        tokenizer=FixtureTokenizer(),
        transport=httpx.MockTransport(lambda request: response_for(request, bad)),
    )
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://local-test"
        ) as client:
            yield client


async def test_unconfigured_backend_is_explicit_not_fake_prediction() -> None:
    application = create_app(Settings())
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://local-test"
        ) as client,
    ):
        assert (await client.get("/health")).json()["inference_configured"] is False
        assert (await client.get("/docs")).status_code == 200
        assert (await client.post("/v1/systemone", json=REQUEST)).status_code == 503


async def test_three_primitives_map_complete_token_probabilities() -> None:
    async with client_for() as client:
        response = await client.post("/v1/systemone", json=REQUEST)
        assert response.status_code == 200
        body = response.json()
        assert body["usage"] == {"input_tokens": 36, "output_tokens": 0}
        assert body["backend_usage"] == {"input_tokens": 36, "output_tokens": 3}
        assert body["answers"]["shipped"]["choice"] == "sent"
        for answer in body["answers"].values():
            assert sum(answer["probabilities"].values()) == pytest.approx(1)
            assert "p_insufficient" not in answer
            assert answer["confidence_semantics"] == "normalized_peak"
            assert answer["calibration"] == "unfitted"
        assert 0 < body["answers"]["score"]["score"] < 1
        assert body["answers"]["yesno"]["noul"] > 0.5


@pytest.mark.parametrize(
    "bad,status", [("missing", 502), ("clipped", 502), ("positive", 502), ("timeout", 504)]
)
async def test_backend_failures_do_not_fabricate_answer(bad: str, status: int) -> None:
    async with client_for(bad) as client:
        response = await client.post("/v1/systemone", json=REQUEST)
        assert response.status_code == status
        assert "answers" not in response.json()


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "?", "criteria": {"only": "One"}},
        {"type": "score", "instructions": "?", "criteria": ["One"]},
        {"type": "noul", "instructions": "?", "criteria": {"unknown": "Maybe"}},
        {"type": "choice", "instructions": " ", "criteria": {"a": "A", "b": "B"}},
    ],
)
async def test_bad_requests_are_rejected_before_model_call(question: dict[str, Any]) -> None:
    async with client_for() as client:
        response = await client.post(
            "/v1/systemone", json={"state": "example", "questions": {"q": question}}
        )
        assert response.status_code == 422


async def test_score_legend_preserves_sdk_criteria_values() -> None:
    async with client_for() as client:
        response = await client.post("/v1/systemone", json=REQUEST)
        assert response.json()["answers"]["score"]["legend"] == {"0": "Not sent", "1": "Sent"}


async def test_whitespace_equivalent_options_return_422() -> None:
    async with client_for() as client:
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "example",
                "questions": {
                    "q": {"type": "choice", "instructions": "?", "criteria": {"a": " A", "b": "A"}}
                },
            },
        )
        assert response.status_code == 422


async def test_internal_readout_requires_operator_opt_in() -> None:
    async with client_for(diagnostics=True) as client:
        body = (await client.post("/v1/systemone", json=REQUEST)).json()
    for answer in body["answers"].values():
        assert 0 < answer["p_insufficient"] < 1
        assert answer["confidence_semantics"] == "normalized_peak_non_U"


async def test_playground_is_available_without_a_model() -> None:
    async with client_for() as client:
        response = await client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "决策工作台" in response.text
    assert "https://cdn" not in response.text
