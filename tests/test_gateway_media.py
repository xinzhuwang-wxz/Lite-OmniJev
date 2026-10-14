"""Public gateway behavior with actual licensed CPU media and a mock upstream only."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import io
import json
import math
import os
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts.s0_prepare import read_manifest
from serve.gateway.app import Settings, create_app
from train.multimodal import file_hash

pytestmark = pytest.mark.anyio

QUESTIONS = {
    "choice": {
        "type": "choice",
        "instructions": "Choose a description.",
        "criteria": {"first": "First", "second": "Second"},
    },
    "noul": {"type": "noul", "instructions": "Is evidence sufficient?"},
    "score": {
        "type": "score",
        "instructions": "Rate visibility.",
        "criteria": ["Low", "Middle", "High"],
    },
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def real_processor() -> Any:
    if not os.getenv("S0_LOCAL_MODEL_DIR") or not os.getenv("S0_CANONICAL_MEDIA_ROOT"):
        pytest.skip("explicit licensed public media and pinned metadata CPU fixture required")
    transformers = importlib.import_module("transformers")
    return transformers.AutoProcessor.from_pretrained(
        os.environ["S0_LOCAL_MODEL_DIR"], local_files_only=True, trust_remote_code=False
    )


def assert_safe_timing(response: httpx.Response) -> None:
    header = response.headers["server-timing"]
    fields = dict(part.strip().split(";dur=") for part in header.split(","))
    assert set(fields) == {"preprocess", "upstream", "total", "queue"}
    assert all(math.isfinite(float(value)) and float(value) >= 0 for value in fields.values())
    assert "PRIVATE" not in header and "data:" not in header


class NativeChatFixture:
    """Decode emitted native bytes independently; no real service/model logits."""

    def __init__(self, processor: Any, failure: str | None = None) -> None:
        self.processor = processor
        self.failure = failure
        self.prompt_counts: list[int] = []
        self.part_types: list[list[str]] = []
        self.expected_summary: dict[str, Any] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.failure == "timeout":
            raise httpx.ReadTimeout("PRIVATE UPSTREAM INPUT", request=request)
        if self.failure == "http_status":
            return httpx.Response(500, content=b"PRIVATE UPSTREAM RESPONSE")
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == "pinned-fixture-model"
        assert payload["max_completion_tokens"] == 1
        assert payload["top_logprobs"] == 0 and payload["logprobs"] is True
        assert payload["top_k"] == -1 and payload["top_p"] == payload["temperature"] == 1
        assert payload["return_token_ids"] is True
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        np = importlib.import_module("numpy")
        image_module = importlib.import_module("PIL.Image")
        images, audio, videos, content = [], [], [], []
        parts = payload["messages"][0]["content"]
        self.part_types.append([part["type"] for part in parts])
        for part in parts:
            kind = part["type"]
            if kind == "image_url":
                uri = part["image_url"]["url"]
                assert uri.startswith("data:image/png;base64,")
                with image_module.open(
                    io.BytesIO(base64.b64decode(uri.split(",", 1)[1], validate=True))
                ) as image:
                    assert image.format == "PNG"
                    images.append(image.convert("RGB"))
                content.append({"type": "image"})
            elif kind == "input_audio":
                assert part["input_audio"]["format"] == "wav"
                data = base64.b64decode(part["input_audio"]["data"], validate=True)
                with wave.open(io.BytesIO(data), "rb") as wav:
                    assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (
                        1,
                        2,
                        16000,
                    )
                    pcm = wav.readframes(wav.getnframes())
                audio.append(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
                content.append({"type": "audio"})
            elif kind == "video_url":
                uri = part["video_url"]["url"]
                assert uri.startswith("data:video/jpeg;base64,")
                frames = []
                for chunk in uri.removeprefix("data:video/jpeg;base64,").split(","):
                    with image_module.open(
                        io.BytesIO(base64.b64decode(chunk, validate=True))
                    ) as image:
                        assert image.format == "JPEG"
                        frames.append(image.convert("RGB"))
                videos.append(frames)
                content.append({"type": "video"})
            else:
                assert kind == "text"
                content.append(part)
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        kwargs: dict[str, Any] = {
            "text": [prompt],
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
            metadata = dict(payload["media_io_kwargs"]["video"])
            assert metadata.pop("do_sample_frames") is False
            assert len(metadata["frames_indices"]) == len(videos[0])
            kwargs.update(videos=videos, video_metadata=[metadata], do_sample_frames=False)
        inputs = self.processor(**kwargs)
        prompt_ids = inputs["input_ids"][0].tolist()
        self.prompt_counts.append(len(prompt_ids))
        if self.expected_summary is not None:
            expected = self.expected_summary
            assert (
                hashlib.sha256(
                    json.dumps(payload["messages"], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                == expected["http_messages_sha256"]
            )
            assert (
                hashlib.sha256(
                    json.dumps(payload.get("media_io_kwargs", {}), sort_keys=True).encode()
                ).hexdigest()
                == expected["http_media_io_kwargs_sha256"]
            )
            assert set(inputs) == set(expected["tensors"])
            for key, value in inputs.items():
                assert (
                    hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                    == expected["tensors"][key]["sha256"]
                )
        ids = payload["logprob_token_ids"]
        assert len(ids) >= 3 and len(set(ids)) == len(ids)
        entries = [
            {"token": f"token_id:{token}", "logprob": -float(i + 1)} for i, token in enumerate(ids)
        ]
        position = {"token": f"token_id:{ids[0]}", "logprob": -1.0, "top_logprobs": entries}
        body = {
            "model": payload["model"],
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
                    "logprobs": {"content": [position]},
                }
            ],
        }
        if self.failure == "missing_u":
            entries.pop()
        elif self.failure == "prompt_ids":
            body["prompt_token_ids"][0] += 1
        elif self.failure == "wrong_model":
            body["model"] = "different-model"
        elif self.failure == "usage":
            body["usage"]["prompt_tokens"] += 1
        elif self.failure == "sentinel":
            entries[0]["logprob"] = -9999
        elif self.failure == "nan":
            entries[0]["logprob"] = float("nan")
        elif self.failure == "two_positions":
            body["choices"][0]["logprobs"]["content"].append(copy.deepcopy(position))
        return httpx.Response(
            200, content=json.dumps(body), headers={"content-type": "application/json"}
        )


@asynccontextmanager
async def gateway(
    processor: Any, root: Path | None, *, failure: str | None = None, configured: bool = True
) -> AsyncIterator[tuple[httpx.AsyncClient, NativeChatFixture]]:
    upstream = NativeChatFixture(processor, failure)
    application = create_app(
        Settings(
            upstream_url="http://fixture" if configured else None,
            upstream_model="pinned-fixture-model",
            raw_logprobs_confirmed=configured,
            media_root=root,
        ),
        processor=processor,
        transport=httpx.MockTransport(upstream),
    )
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://gateway-test"
        ) as client,
    ):
        yield client, upstream


def request_case(index: int) -> dict[str, Any]:
    cases = read_manifest(Path(__file__).parent / "fixtures/s0-prepare-public-001/manifest.json")
    case = cases[index]
    return {
        "state": case.prefix.state,
        "media": [media.model_dump() for media in case.prefix.media],
        "video_frames": case.video_frames,
        "questions": QUESTIONS,
    }


@pytest.mark.parametrize(
    "index,part",
    [
        (0, "text"),
        (20, "image_url"),
        (40, "image_url"),
        (60, "input_audio"),
        (80, "video_url"),
        (90, "video_url"),
    ],
)
async def test_real_five_categories_and_multiquestion_canonical_chat(
    real_processor: Any, index: int, part: str
) -> None:
    async with gateway(real_processor, Path(os.environ["S0_CANONICAL_MEDIA_ROOT"])) as (
        client,
        upstream,
    ):
        health = (await client.get("/health")).json()
        assert health["mode"] == "canonical-chat" and health["inference_configured"] is True
        response = await client.post("/v1/systemone", json=request_case(index))
    assert response.status_code == 200
    assert_safe_timing(response)
    body = response.json()
    assert set(body["answers"]) == set(QUESTIONS)
    assert body["usage"] == {"input_tokens": sum(upstream.prompt_counts), "output_tokens": 0}
    assert body["backend_usage"] == {
        "input_tokens": sum(upstream.prompt_counts),
        "output_tokens": 3,
    }
    assert all(part in parts for parts in upstream.part_types)
    for answer in body["answers"].values():
        assert sum(answer["probabilities"].values()) == pytest.approx(1)
        assert "U" not in answer["probabilities"] and 0 < answer["p_insufficient"] < 1
        assert answer["calibration"] == "unfitted"
    assert body["answers"]["choice"]["choice"] == "first"
    assert body["answers"]["score"]["legend"] == {"0": "Low", "1": "Middle", "2": "High"}
    assert "GPU parity" in body["validation_status"]
    assert request_case(index)["state"] not in response.text


@pytest.mark.parametrize(
    "failure,status",
    [
        ("missing_u", 502),
        ("prompt_ids", 502),
        ("wrong_model", 502),
        ("usage", 502),
        ("sentinel", 502),
        ("nan", 502),
        ("two_positions", 502),
        ("timeout", 504),
        ("http_status", 502),
    ],
)
async def test_upstream_protocol_and_timeout_errors_are_private(
    real_processor: Any, failure: str, status: int
) -> None:
    async with gateway(
        real_processor, Path(os.environ["S0_CANONICAL_MEDIA_ROOT"]), failure=failure
    ) as (client, _):
        request = request_case(20)
        request["state"] = "PRIVATE CUSTOMER STATE"
        response = await client.post("/v1/systemone", json=request)
    assert response.status_code == status and "answers" not in response.json()
    assert_safe_timing(response)
    assert "PRIVATE" not in response.text and "image.png" not in response.text


@pytest.mark.parametrize(
    "failure",
    [
        "hash",
        "escape",
        "url",
        "absolute",
        "symlink",
        "empty",
        "oversize",
        "corrupt_audio",
        "long_audio",
        "wrong_kind",
        "multiple_audio",
        "multiple_video",
        "image_video",
    ],
)
async def test_bad_media_rejected_before_any_upstream_call(
    real_processor: Any, tmp_path: Path, failure: str
) -> None:
    request = request_case(20)
    request["state"] = "PRIVATE CUSTOMER STATE"
    request["media"][0]["sha256"] = "a" * 64
    root = Path(os.environ["S0_CANONICAL_MEDIA_ROOT"])
    if failure == "escape":
        request["media"][0]["uri"] = "../PRIVATE_SECRET.png"
    elif failure == "url":
        request["media"][0]["uri"] = "https://example.invalid/PRIVATE_SECRET"
    elif failure == "absolute":
        request["media"][0]["uri"] = str(root / "image.png")
    elif failure in {"symlink", "empty", "oversize", "corrupt_audio", "long_audio"}:
        root = tmp_path
        asset = root / "PRIVATE_SECRET"
        if failure == "symlink":
            asset.symlink_to(Path(os.environ["S0_CANONICAL_MEDIA_ROOT"]) / "image.png")
        elif failure == "empty":
            asset.write_bytes(b"")
        elif failure == "oversize":
            with asset.open("wb") as handle:
                handle.write(
                    (Path(os.environ["S0_CANONICAL_MEDIA_ROOT"]) / "image.png").read_bytes()
                )
                handle.truncate(64 * 1024 * 1024 + 1)
        elif failure == "corrupt_audio":
            asset.write_bytes(b"not a WAV")
        else:
            with wave.open(str(asset), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b"\0\0" * (31 * 8000))
        request["media"][0]["uri"] = asset.name
        request["media"][0]["sha256"] = file_hash(asset)
        if failure in {"corrupt_audio", "long_audio"}:
            request["media"][0]["type"] = "audio"
            request["media"][0]["sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    elif failure == "wrong_kind":
        request["media"] = request_case(60)["media"]
        request["media"][0]["type"] = "video"
    elif failure == "multiple_audio":
        request["media"] = request_case(60)["media"] * 2
    elif failure == "multiple_video":
        request["media"] = request_case(80)["media"] * 2
    elif failure == "image_video":
        request["media"] += request_case(80)["media"]
    async with gateway(real_processor, root) as (client, upstream):
        response = await client.post("/v1/systemone", json=request)
    assert response.status_code == 422 and not upstream.prompt_counts
    assert_safe_timing(response)
    assert "PRIVATE" not in response.text and "answers" not in response.json()


@pytest.mark.parametrize("failure", ["root", "processor", "backend", "root_symlink"])
async def test_missing_configuration_returns_503(
    real_processor: Any, tmp_path: Path, failure: str
) -> None:
    root = Path(os.environ["S0_CANONICAL_MEDIA_ROOT"])
    if failure == "root":
        root = None
    elif failure == "root_symlink":
        link = tmp_path / "linked-root"
        link.symlink_to(root, target_is_directory=True)
        root = link
    async with gateway(
        None if failure == "processor" else real_processor, root, configured=failure != "backend"
    ) as (client, upstream):
        response = await client.post("/v1/systemone", json=request_case(20))
        health = (await client.get("/health")).json()
    assert response.status_code == 503 and not upstream.prompt_counts
    assert health["inference_configured"] is False
    assert "answers" not in response.json()


async def test_schema_error_does_not_echo_private_input() -> None:
    application = create_app(Settings())
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://fixture"
        ) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "PRIVATE CUSTOMER STATE",
                "media": [
                    {"type": "image", "uri": "PRIVATE PATH", "sha256": "PRIVATE INVALID HASH"}
                ],
                "video_frames": 4,
                "questions": QUESTIONS,
            },
        )
    assert response.status_code == 422 and "PRIVATE" not in response.text


async def test_fixed_100_gateway_requests_match_canonical_reference(real_processor: Any) -> None:
    root = Path(__file__).parent.parent
    cases = read_manifest(root / "tests/fixtures/s0-prepare-public-001/manifest.json")
    reference_path = Path(
        os.environ.get(
            "S0_CANONICAL_REFERENCE", str(root / "reports/s0-canonical-http-cpu-003.json")
        )
    )
    reference = json.loads(reference_path.read_text())
    assert reference["status"] == "passed" and reference["complete_100_cases"]
    assert (
        reference["manifest_sha256"]
        == hashlib.sha256(
            (root / "tests/fixtures/s0-prepare-public-001/manifest.json").read_bytes()
        ).hexdigest()
    )
    async with gateway(real_processor, Path(os.environ["S0_CANONICAL_MEDIA_ROOT"])) as (
        client,
        upstream,
    ):
        for case, expected in zip(cases, reference["cases"], strict=True):
            upstream.expected_summary = expected
            question = {"type": case.question.type, "instructions": case.question.instruction}
            if case.question.type == "choice":
                question["criteria"] = {
                    chr(65 + i): label for i, label in enumerate(case.question.options)
                }
            elif case.question.type == "score":
                question["criteria"] = case.question.score_descriptions or [
                    str(i) for i in range(case.question.score_levels)
                ]
            request = {
                "state": case.prefix.state,
                "media": [media.model_dump() for media in case.prefix.media],
                "video_frames": case.video_frames,
                "questions": {case.question.qid: question},
            }
            response = await client.post("/v1/systemone", json=request)
            assert response.status_code == 200
            assert_safe_timing(response)
            assert response.json()["backend_usage"]["output_tokens"] == 1
    assert len(upstream.prompt_counts) == 100


async def test_lifespan_loads_real_processor_from_local_metadata_only(
    real_processor: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = importlib.import_module("transformers")
    original = transformers.AutoProcessor.from_pretrained
    calls = []

    def load(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load)
    upstream = NativeChatFixture(real_processor)
    model_dir = Path(os.environ["S0_LOCAL_MODEL_DIR"])
    application = create_app(
        Settings(
            upstream_url="http://fixture",
            upstream_model="pinned-fixture-model",
            raw_logprobs_confirmed=True,
            model_dir=model_dir,
            media_root=Path(os.environ["S0_CANONICAL_MEDIA_ROOT"]),
        ),
        transport=httpx.MockTransport(upstream),
    )
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://fixture"
        ) as client,
    ):
        response = await client.post("/v1/systemone", json=request_case(40))
    assert response.status_code == 200 and len(calls) == 1
    assert calls[0] == ((model_dir,), {"local_files_only": True, "trust_remote_code": False})


async def test_legacy_lifespan_does_not_force_processor_load(
    real_processor: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = importlib.import_module("transformers")
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("legacy must not load processor"),
    )
    application = create_app(
        Settings(
            upstream_url="http://fixture",
            raw_logprobs_confirmed=True,
            model_dir=Path(os.environ["S0_LOCAL_MODEL_DIR"]),
        ),
        tokenizer=real_processor.tokenizer,
    )
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://fixture"
        ) as client,
    ):
        health = (await client.get("/health")).json()
        assert (
            health["mode"] == "legacy-text-completions" and health["inference_configured"] is True
        )
        assert "Lite-OmniJev local text gateway" in (await client.get("/openapi.json")).text


async def test_failed_local_processor_load_does_not_fallback_to_legacy(
    real_processor: Any, tmp_path: Path
) -> None:
    application = create_app(
        Settings(
            upstream_url="http://fixture",
            raw_logprobs_confirmed=True,
            model_dir=tmp_path,
            media_root=Path(os.environ["S0_CANONICAL_MEDIA_ROOT"]),
        ),
        tokenizer=real_processor.tokenizer,
        transport=httpx.MockTransport(lambda _: pytest.fail("must not reach upstream")),
    )
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://fixture"
        ) as client,
    ):
        response = await client.post(
            "/v1/systemone", json={"state": "PRIVATE STATE", "questions": QUESTIONS}
        )
        assert response.status_code == 503 and "PRIVATE" not in response.text
        assert (await client.get("/health")).json()["inference_configured"] is False


async def test_media_root_environment_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LITE_MEDIA_ROOT", str(tmp_path))
    assert Settings.from_env().media_root == tmp_path


async def test_ten_questions_use_sequential_canonical_requests(real_processor: Any) -> None:
    import asyncio

    active = 0
    maximum_active = 0
    upstream = NativeChatFixture(real_processor)

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(active, maximum_active)
        try:
            # Controlled scheduling overlap, not a performance or latency assertion.
            await asyncio.sleep(0.05)
            return upstream(request)
        finally:
            active -= 1

    application = create_app(
        Settings(
            upstream_url="http://fixture",
            upstream_model="pinned-fixture-model",
            raw_logprobs_confirmed=True,
            media_root=Path(os.environ["S0_CANONICAL_MEDIA_ROOT"]),
        ),
        processor=real_processor,
        transport=httpx.MockTransport(handle),
    )
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://fixture"
        ) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "PUBLIC SEQUENTIAL TEST",
                "questions": {f"q-{index}": QUESTIONS["choice"] for index in range(10)},
            },
        )
    assert response.status_code == 200 and maximum_active == 1 and len(upstream.prompt_counts) == 10
    assert response.json()["backend_usage"]["output_tokens"] == 10
    assert_safe_timing(response)


async def test_unconfigured_response_timing_headers_do_not_claim_inference() -> None:
    application = create_app(Settings())
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://fixture"
        ) as client,
    ):
        response = await client.post(
            "/v1/systemone", json={"state": "PRIVATE STATE", "questions": QUESTIONS}
        )
    assert response.status_code == 503 and "PRIVATE" not in response.text
    assert_safe_timing(response)


async def test_cancelled_request_keeps_cpu_lock_until_thread_finishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import asyncio
    import threading
    from types import SimpleNamespace

    gateway = importlib.import_module("serve.gateway.app")
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    active = 0
    maximum_active = 0
    calls = 0
    guard = threading.Lock()

    def blocked_prepare(*args: Any, **kwargs: Any) -> Any:
        nonlocal active, maximum_active, calls
        with guard:
            calls += 1
            ordinal = calls
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            (first_entered if ordinal == 1 else second_entered).set()
            assert release.wait(5), "test must release bounded worker"
            raise ValueError("PRIVATE THIRD PARTY MEDIA ERROR")
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(gateway, "check_answer_tokens", lambda *args: {})
    monkeypatch.setattr(gateway, "prepare_decision", blocked_prepare)
    application = create_app(
        Settings(
            upstream_url="http://fixture",
            raw_logprobs_confirmed=True,
            media_root=tmp_path,
        ),
        processor=SimpleNamespace(tokenizer=object()),
        transport=httpx.MockTransport(lambda request: pytest.fail("no upstream on bad media")),
    )
    payload = {"state": "PUBLIC CANCEL TEST", "questions": {"q": QUESTIONS["choice"]}}
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://fixture"
        ) as client:
            first = asyncio.create_task(client.post("/v1/systemone", json=payload))
            second = None
            try:
                assert await asyncio.to_thread(first_entered.wait, 2)
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(client.post("/v1/systemone", json=payload))
                # Scheduling window detects overlap, not a performance threshold.
                assert not await asyncio.to_thread(second_entered.wait, 0.1)
                assert maximum_active == 1
            finally:
                release.set()
                if second is not None:
                    response = await asyncio.wait_for(second, 3)
            assert response.status_code == 422 and "PRIVATE" not in response.text
    assert active == 0 and calls == 2 and maximum_active == 1


@pytest.mark.parametrize("cancel_shutdown", [False, True])
async def test_shutdown_drains_cancelled_request_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cancel_shutdown: bool
) -> None:
    import asyncio
    import threading
    from types import SimpleNamespace

    gateway = importlib.import_module("serve.gateway.app")
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked_prepare(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        try:
            assert release.wait(5), "test must release bounded worker"
            raise ValueError("PRIVATE DECODER FAILURE")
        finally:
            finished.set()

    monkeypatch.setattr(gateway, "check_answer_tokens", lambda *args: {})
    monkeypatch.setattr(gateway, "prepare_decision", blocked_prepare)
    application = create_app(
        Settings(upstream_url="http://fixture", raw_logprobs_confirmed=True, media_root=tmp_path),
        processor=SimpleNamespace(tokenizer=object()),
        transport=httpx.MockTransport(lambda request: pytest.fail("no upstream on bad media")),
    )
    lifespan = application.router.lifespan_context(application)
    await lifespan.__aenter__()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://fixture"
    ) as client:
        request = asyncio.create_task(
            client.post(
                "/v1/systemone", json={"state": "PUBLIC", "questions": {"q": QUESTIONS["choice"]}}
            )
        )
        shutdown = None
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
            await asyncio.sleep(0)
            assert not shutdown.done() and not finished.is_set()
            if cancel_shutdown:
                shutdown.cancel()
                await asyncio.sleep(0)
                assert not shutdown.done() and not finished.is_set()
        finally:
            release.set()
            if shutdown is not None:
                if cancel_shutdown:
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(shutdown, 3)
                else:
                    await asyncio.wait_for(shutdown, 3)
            else:
                await lifespan.__aexit__(None, None, None)
    assert finished.is_set()


async def test_media_reference_budget_rejects_before_preparation_or_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    gateway = importlib.import_module("serve.gateway.app")
    monkeypatch.setattr(
        gateway, "prepare_decision", lambda *args, **kwargs: pytest.fail("no CPU preparation")
    )
    application = create_app(
        Settings(upstream_url="http://fixture", raw_logprobs_confirmed=True, media_root=tmp_path),
        processor=SimpleNamespace(tokenizer=object()),
        transport=httpx.MockTransport(lambda request: pytest.fail("no upstream")),
    )
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://fixture"
        ) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "PRIVATE STATE",
                "questions": QUESTIONS,
                "media": [{"type": "image", "uri": "PRIVATE.png", "sha256": "0" * 64}] * 9,
            },
        )
    assert response.status_code == 422 and "PRIVATE" not in response.text
    assert_safe_timing(response)


async def test_canonical_gateway_releases_media_tensors_before_backend_call(
    real_processor: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weakref

    import serve.gateway.app as gateway_module

    original_prepare = gateway_module.prepare_decision
    media_refs: list[Any] = []

    def observed_prepare(*args: Any, **kwargs: Any) -> Any:
        prepared = original_prepare(*args, **kwargs)
        media_refs.append(weakref.ref(prepared.inputs["pixel_values"]))
        return prepared

    upstream = NativeChatFixture(real_processor)

    def handler(request: httpx.Request) -> httpx.Response:
        assert media_refs and all(reference() is None for reference in media_refs)
        return upstream(request)

    monkeypatch.setattr(gateway_module, "prepare_decision", observed_prepare)
    app = create_app(
        Settings(
            upstream_url="http://fixture",
            upstream_model="pinned-fixture-model",
            raw_logprobs_confirmed=True,
            media_root=Path(os.environ["S0_CANONICAL_MEDIA_ROOT"]),
        ),
        processor=real_processor,
        transport=httpx.MockTransport(handler),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway-test"
        ) as client,
    ):
        result = await client.post("/v1/systemone", json=request_case(20))
    assert result.status_code == 200
    assert set(result.json()["answers"]) == set(QUESTIONS)
