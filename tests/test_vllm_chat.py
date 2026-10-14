"""Protocol fixtures and opt-in real CPU media; never a real backend/quality claim."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import io
import json
import os
import wave
from pathlib import Path
from typing import Any

import httpx
import pytest

from data.schema import Prefix
from scripts.s0_prepare import read_manifest
from serve.vllm_chat import VllmChatClient, VllmChatProtocolError
from train.multimodal import PreparedDecision, prepare_case, prepare_decision

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class IdRow:
    def tolist(self) -> list[int]:
        return [10, 20, 30]


def prepared_fixture() -> PreparedDecision:
    prepared = PreparedDecision(
        messages=[{"role": "user", "content": [{"type": "text", "text": "PRIVATE INPUT"}]}],
        prompt="PRIVATE INPUT",
        inputs={"input_ids": [IdRow()]},
        answer_token_ids={"A": 1, "B": 2, "U": 3},
        summary={"input_profile": "canonical-http-v1"},
        http_messages=[{"role": "user", "content": [{"type": "text", "text": "PRIVATE INPUT"}]}],
        http_media_io_kwargs={},
    )

    prepared.summary.update(
        answer_token_mapping_sha256=hashlib.sha256(
            json.dumps(prepared.answer_token_ids, sort_keys=True).encode()
        ).hexdigest(),
        http_messages_sha256=hashlib.sha256(
            json.dumps(prepared.http_messages, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        http_media_io_kwargs_sha256=hashlib.sha256(b"{}").hexdigest(),
    )
    return prepared


def response_body(
    ids: list[int] | None = None, prompt_ids: list[int] | None = None
) -> dict[str, Any]:
    ids = ids or [1, 2, 3]
    prompt_ids = prompt_ids or [10, 20, 30]
    return {
        "model": "pinned-alias",
        "prompt_token_ids": prompt_ids,
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": 1,
            "total_tokens": len(prompt_ids) + 1,
        },
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"role": "assistant", "content": "mock"},
                "logprobs": {
                    "content": [
                        {
                            "token": f"token_id:{ids[0]}",
                            "logprob": -1.0,
                            "top_logprobs": [
                                {"token": f"token_id:{token}", "logprob": -float(i + 1)}
                                for i, token in enumerate(ids)
                            ],
                        }
                    ]
                },
            }
        ],
    }


async def test_native_chat_requests_all_ids_without_top_k_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["logprob_token_ids"] == [1, 2, 3]
        assert payload["logprobs"] is True and payload["top_logprobs"] == 0
        assert payload["max_completion_tokens"] == 1 and payload["stream"] is False
        assert payload["n"] == 1 and payload["temperature"] == payload["top_p"] == 1
        assert payload["top_k"] == -1 and payload["min_p"] == 0
        assert payload["presence_penalty"] == payload["frequency_penalty"] == 0
        assert payload["repetition_penalty"] == 1 and payload["seed"] == 42
        assert payload["return_token_ids"] is True
        assert payload["return_tokens_as_token_ids"] is True
        assert payload["add_generation_prompt"] is True and payload["add_special_tokens"] is False
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "allowed_token_ids" not in payload and "prompt" not in payload
        body = response_body()
        body["choices"][0]["logprobs"]["content"][0]["top_logprobs"].reverse()
        return httpx.Response(200, json=body)

    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.read_logprobs(prepared_fixture())
    assert result.raw_logprobs == {"A": -1.0, "B": -2.0, "U": -3.0}
    assert result.summary["expanded_prompt_ids_match_hf"] is True
    assert "PRIVATE INPUT" not in json.dumps(result.summary)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_u",
        "duplicate",
        "negative_id",
        "leading_zero_id",
        "nan",
        "positive",
        "sentinel",
        "wrong_model",
        "missing_prompt_ids",
        "same_length_wrong_ids",
        "prompt_bool",
        "usage_prompt",
        "usage_completion",
        "usage_total",
        "usage_bool",
        "two_positions",
        "two_choices",
        "unsolicited_token",
        "missing_logprobs",
        "text_only_tokens",
    ],
)
async def test_rejects_incomplete_or_transformed_protocol(failure: str) -> None:
    body = response_body()
    content = body["choices"][0]["logprobs"]["content"]
    entries = content[0]["top_logprobs"]
    if failure == "missing_u":
        entries.pop()
    elif failure == "duplicate":
        entries[2]["token"] = entries[0]["token"]
    elif failure == "negative_id":
        entries[2]["token"] = "token_id:-3"
    elif failure == "leading_zero_id":
        entries[2]["token"] = "token_id:03"
    elif failure == "nan":
        entries[0]["logprob"] = float("nan")
    elif failure == "positive":
        entries[0]["logprob"] = 0.1
    elif failure == "sentinel":
        entries[0]["logprob"] = -9999
    elif failure == "wrong_model":
        body["model"] = "other-revision-alias"
    elif failure == "missing_prompt_ids":
        body.pop("prompt_token_ids")
    elif failure == "same_length_wrong_ids":
        body["prompt_token_ids"][0] = 11
    elif failure == "prompt_bool":
        body["prompt_token_ids"][0] = True
    elif failure == "usage_prompt":
        body["usage"]["prompt_tokens"] = 2
    elif failure == "usage_completion":
        body["usage"]["completion_tokens"] = 2
    elif failure == "usage_total":
        body["usage"]["total_tokens"] = 9
    elif failure == "usage_bool":
        body["usage"]["completion_tokens"] = True
    elif failure == "two_positions":
        content.append(copy.deepcopy(content[0]))
    elif failure == "two_choices":
        body["choices"].append(copy.deepcopy(body["choices"][0]))
    elif failure == "unsolicited_token":
        entries.append({"token": "token_id:99", "logprob": -3.0})
    elif failure == "missing_logprobs":
        body["choices"][0].pop("logprobs")
    elif failure == "text_only_tokens":
        entries[0]["token"] = "A"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=json.dumps(body), headers={"content-type": "application/json"}
        )

    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(VllmChatProtocolError) as caught:
            await client.read_logprobs(prepared_fixture())
    assert "PRIVATE INPUT" not in str(caught.value)


async def test_one_extra_sampled_token_is_allowed() -> None:
    body = response_body()
    position = body["choices"][0]["logprobs"]["content"][0]
    position["token"] = "token_id:99"
    position["logprob"] = -7.0
    position["top_logprobs"].append({"token": "token_id:99", "logprob": -7.0})
    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
    ) as client:
        result = await client.read_logprobs(prepared_fixture())
    assert list(result.raw_logprobs) == ["A", "B", "U"]


async def test_requires_attestation_and_canonical_profile_before_http() -> None:
    with pytest.raises(VllmChatProtocolError, match="not_confirmed"):
        VllmChatClient("http://fixture", "pinned-alias", raw_logprobs_confirmed=False)
    prepared = prepared_fixture()
    prepared.summary["input_profile"] = "hf-only-v1"
    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(lambda _: pytest.fail("must not send")),
    ) as client:
        with pytest.raises(VllmChatProtocolError, match="canonical_http"):
            await client.read_logprobs(prepared)


@pytest.fixture(scope="module")
def real_processor() -> Any:
    model_dir = os.getenv("S0_LOCAL_MODEL_DIR")
    if not model_dir or not os.getenv("S0_CANONICAL_MEDIA_ROOT"):
        pytest.skip("explicit pinned metadata/real public CPU media fixture required")
    transformers = importlib.import_module("transformers")
    return transformers.AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False
    )


@pytest.mark.parametrize(
    "case_index", [0, 20, 40, 60, 80, 90, "api_images", "api_image_audio", "api_video_audio"]
)
async def test_real_media_bytes_roundtrip_to_identical_hf_tensors(
    real_processor: Any, case_index: int | str
) -> None:
    np = importlib.import_module("numpy")
    image_module = importlib.import_module("PIL.Image")
    cases = read_manifest(Path(__file__).parent / "fixtures/s0-prepare-public-001/manifest.json")
    asset_root = Path(os.environ["S0_CANONICAL_MEDIA_ROOT"])
    if isinstance(case_index, str):
        source_indices = {
            "api_images": [20, 40],
            "api_image_audio": [20, 60],
            "api_video_audio": [90, 60],
        }[case_index]
        prefix = Prefix(
            state="Public API input preparation fixture",
            media=[cases[index].prefix.media[0] for index in source_indices],
        )
        prepared = prepare_decision(
            prefix,
            cases[0].question,
            real_processor,
            asset_root,
            profile="canonical_http",
            video_frames=16,
        )
        assert "provenance" not in prepared.summary and "id" not in prepared.summary
        expected_frames = 16
    else:
        prepared = prepare_case(
            cases[case_index], real_processor, asset_root, profile="canonical_http"
        )
        expected_frames = cases[case_index].video_frames

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"] == prepared.http_messages
        images, audio, videos = [], [], []
        for part in payload["messages"][0]["content"]:
            if part["type"] == "image_url":
                data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1], validate=True)
                with image_module.open(io.BytesIO(data)) as image:
                    images.append(image.convert("RGB"))
            elif part["type"] == "input_audio":
                assert part["input_audio"]["format"] == "wav"
                data = base64.b64decode(part["input_audio"]["data"], validate=True)
                with wave.open(io.BytesIO(data), "rb") as wav:
                    assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (
                        1,
                        16000,
                        2,
                    )
                    pcm = wav.readframes(wav.getnframes())
                audio.append(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
            elif part["type"] == "video_url":
                chunks = part["video_url"]["url"].removeprefix("data:video/jpeg;base64,").split(",")
                frames = []
                for chunk in chunks:
                    with image_module.open(
                        io.BytesIO(base64.b64decode(chunk, validate=True))
                    ) as image:
                        assert image.format == "JPEG"
                        frames.append(image.convert("RGB"))
                videos.append(frames)
        kwargs: dict[str, Any] = {
            "text": [prepared.prompt],
            "return_tensors": "pt",
            "return_mm_token_type_ids": True,
            "truncation": False,
            "add_special_tokens": False,
        }
        if images:
            kwargs["images"] = [images]
        if audio:
            kwargs.update(audio=audio, sampling_rate=16000)
        if videos:
            io_kwargs = payload["media_io_kwargs"]["video"]
            assert io_kwargs["do_sample_frames"] is False
            assert len(io_kwargs["frames_indices"]) == len(videos[0]) == expected_frames
            kwargs.update(
                videos=videos,
                video_metadata=prepared.summary["video_metadata"],
                do_sample_frames=False,
            )
        reconstructed = real_processor(**kwargs)
        assert set(reconstructed) == set(prepared.inputs)
        for key, value in reconstructed.items():
            assert (
                hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                == prepared.summary["tensors"][key]["sha256"]
            )
        ids = payload["logprob_token_ids"]
        assert ids == list(prepared.answer_token_ids.values())
        return httpx.Response(
            200, json=response_body(ids, prepared.inputs["input_ids"][0].tolist())
        )

    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        result = await client.read_logprobs(prepared)
    assert set(result.raw_logprobs) == set(prepared.answer_token_ids)


@pytest.mark.parametrize("failure", ["http_status", "timeout", "invalid_json"])
async def test_transport_failures_do_not_echo_response_or_input(failure: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("PRIVATE INPUT", request=request)
        return httpx.Response(500 if failure == "http_status" else 200, content=b"PRIVATE OUTPUT")

    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(VllmChatProtocolError) as caught:
            await client.read_logprobs(prepared_fixture())
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize(
    "mapping", [{"A": 1, "B": 2}, {"A": 1, "B": 1, "U": 3}, {"A": True, "B": 2, "U": 3}]
)
async def test_invalid_mapping_never_reaches_http(mapping: dict[str, int]) -> None:
    prepared = prepared_fixture()
    prepared.answer_token_ids.clear()
    prepared.answer_token_ids.update(mapping)
    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(lambda _: pytest.fail("must not send")),
    ) as client:
        with pytest.raises(VllmChatProtocolError, match="mapping"):
            await client.read_logprobs(prepared)


async def test_mutated_canonical_bytes_are_rejected_before_http() -> None:
    prepared = prepared_fixture()
    assert prepared.http_messages is not None
    prepared.http_messages[0]["content"][0]["text"] = "tampered input"
    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(lambda _: pytest.fail("must not send")),
    ) as client:
        with pytest.raises(VllmChatProtocolError, match="binding_mismatch"):
            await client.read_logprobs(prepared)


@pytest.mark.parametrize("case_index", [0, 20, 40, 60, 80, 90])
async def test_default_hf_profile_preserves_original_real_tensor_hashes(
    real_processor: Any, case_index: int
) -> None:
    root = Path(__file__).parent.parent
    cases = read_manifest(root / "tests/fixtures/s0-prepare-public-001/manifest.json")
    baseline = json.loads((root / "reports/s0-prepare-mac-cpu-004.json").read_text())
    prepared = prepare_case(
        cases[case_index], real_processor, Path(os.environ["S0_CANONICAL_MEDIA_ROOT"])
    )
    assert prepared.http_messages is None and prepared.http_media_io_kwargs is None
    assert prepared.summary["tensors"] == baseline["cases"][case_index]["tensors"]


async def test_changed_answer_mapping_cannot_send_request() -> None:
    prepared = prepared_fixture()
    prepared.summary["answer_token_mapping_sha256"] = hashlib.sha256(
        json.dumps(prepared.answer_token_ids, sort_keys=True).encode()
    ).hexdigest()
    prepared.answer_token_ids["A"], prepared.answer_token_ids["B"] = 2, 1
    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(lambda _: pytest.fail("mutated mapping must not send")),
    ) as client:
        with pytest.raises(VllmChatProtocolError, match="answer_token_mapping"):
            await client.read_logprobs(prepared)


async def test_raw_answer_mass_cannot_exceed_full_vocabulary_mass() -> None:
    body = response_body()
    body["choices"][0]["logprobs"]["content"][0]["logprob"] = -0.1
    for entry in body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        entry["logprob"] = -0.1
    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
    ) as client:
        with pytest.raises(VllmChatProtocolError, match="probability_mass"):
            await client.read_logprobs(prepared_fixture())


@pytest.mark.parametrize("sampled_in_top", [False, True])
async def test_rejects_sampled_probability_inconsistent_with_unique_raw_mass(
    sampled_in_top: bool,
) -> None:
    import math

    body = response_body()
    position = body["choices"][0]["logprobs"]["content"][0]
    for entry in position["top_logprobs"]:
        entry["logprob"] = math.log(0.2)
    position["token"] = "token_id:1" if sampled_in_top else "token_id:99"
    position["logprob"] = math.log(0.9)
    async with VllmChatClient(
        "http://fixture",
        "pinned-alias",
        raw_logprobs_confirmed=True,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)),
    ) as client:
        with pytest.raises(VllmChatProtocolError):
            await client.read_logprobs(prepared_fixture())
