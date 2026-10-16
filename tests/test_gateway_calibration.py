"""Synthetic CPU registry/HTTP contracts; no real tokenizer, model or private calibration claim."""

import hashlib
import json

import httpx
import pytest

from data.schema import Media, Question
from scripts.check_tokens import ANSWERS
from serve.gateway import calibration
from serve.gateway.app import Settings, create_app
from train.calibration import Binding, TemperatureArtifact
from train.readout import read_probabilities


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    model = tmp_path / ("a" * 40)
    model.mkdir()
    payloads = {
        "config.json": json.dumps(
            {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]}
        ).encode(),
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "chat_template.jinja": b"synthetic-template",
        "model.safetensors": b"synthetic-base",
    }
    for name, payload in payloads.items():
        (model / name).write_bytes(payload)
    assets = tmp_path / "assets.json"
    assets.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "repo_id": "google/gemma-4-E4B-it",
                        "revision": model.name,
                        "license": "gemma",
                        "files": [
                            {
                                "path": name,
                                "kind": "weights" if name.endswith(".safetensors") else "metadata",
                                "size": len(payload),
                                "actual_sha256": digest(payload),
                            }
                            for name, payload in payloads.items()
                        ],
                    }
                ],
            }
        )
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic-adapter")
    (adapter / "adapter_config.json").write_text("{}")
    mapping = {token: index for index, token in enumerate(ANSWERS)}
    monkeypatch.setattr(
        calibration,
        "probe",
        lambda path: {
            "status": "passed",
            "template_sha256": digest((path / "chat_template.jinja").read_bytes()),
            "answer_token_ids": mapping,
        },
    )
    argv = [
        "/runtime/bin/vllm",
        "serve",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8014",
        "--logprobs-mode",
        "raw_logprobs",
        "--enable-lora",
        "--lora-modules",
        f"decision={adapter}",
    ]
    monkeypatch.setattr(calibration, "_cmdline", lambda pid: tuple(argv))
    server = tmp_path / "server.json"
    server.write_text(json.dumps({"ready": True, "argv": argv, "server_pid": 42}))
    entries = []
    for cohort, primitive in sorted(calibration.REQUIRED_GROUPS):
        binding = Binding(
            base_revision=model.name,
            adapter_weights_sha256=digest(b"synthetic-adapter"),
            official_template_sha256=digest(b"synthetic-template"),
            answer_token_map_sha256=digest(json.dumps(mapping, sort_keys=True).encode()),
            raw_logits_receipt_sha256="9" * 64,
            cohort=cohort,
            primitive=primitive,
            seed=42,
            code_revision="e" * 40,
            calibration_data_sha256="f" * 64,
        )
        artifact = TemperatureArtifact(
            schema_version=1,
            method="bounded_scalar_joint_nll",
            fitter_implementation_sha256="1" * 64,
            binding=binding,
            temperature=2.0,
            temperature_min=0.05,
            temperature_max=20.0,
            rows=4,
            nll_before=0.8,
            nll_after=0.7,
            stage_acceptance="not_evaluated",
        )
        path = tmp_path / f"{cohort}-{primitive}.json"
        path.write_text(artifact.model_dump_json())
        entries.append(
            {
                "path": path.name,
                "sha256": digest(path.read_bytes()),
                "binding": binding.model_dump(),
            }
        )
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_version": "synthetic-model",
                "calibration_version": "synthetic-calibration",
                "groups": entries,
            }
        )
    )
    return {
        "path": registry_path,
        "serving_receipt": server,
        "asset_manifest": assets,
        "model_dir": model,
        "upstream_url": "http://127.0.0.1:8014",
        "upstream_model": "decision",
    }


def load(bundle):
    return calibration.load_registry(**bundle)


def update_registry(bundle, change):
    path = bundle["path"]
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "field",
    [
        "base_revision",
        "adapter_weights_sha256",
        "official_template_sha256",
        "answer_token_map_sha256",
    ],
)
def test_actual_deployment_binding_mismatch_rejected(bundle, field):
    update_registry(
        bundle,
        lambda value: value["groups"][0]["binding"].update(
            {field: "0" * (40 if field == "base_revision" else 64)}
        ),
    )
    with pytest.raises(ValueError, match="actual deployment files"):
        load(bundle)


def test_missing_duplicate_and_wrong_artifact_group_rejected(bundle):
    original = bundle["path"].read_bytes()
    update_registry(bundle, lambda value: value["groups"].pop())
    with pytest.raises(ValueError, match="missing configured"):
        load(bundle)
    bundle["path"].write_bytes(original)
    update_registry(bundle, lambda value: value["groups"].append(value["groups"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        load(bundle)
    bundle["path"].write_bytes(original)
    update_registry(bundle, lambda value: value["groups"][0]["binding"].update(seed=43))
    with pytest.raises(ValueError, match="binding mismatch"):
        load(bundle)


def test_operator_alias_or_hash_without_observed_process_is_insufficient(bundle, monkeypatch):
    monkeypatch.setattr(calibration, "_cmdline", lambda pid: ("different-process",))
    with pytest.raises(ValueError, match="actual local process"):
        load(bundle)


def test_historical_receipt_and_remote_endpoint_rejected(bundle):
    receipt = json.loads(bundle["serving_receipt"].read_text())
    receipt["server_exit_code"] = 0
    bundle["serving_receipt"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="live local"):
        load(bundle)
    del receipt["server_exit_code"]
    bundle["serving_receipt"].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="local endpoint"):
        load({**bundle, "upstream_url": "http://remote:8014"})


def test_real_local_file_tampering_rejected(bundle):
    (bundle["model_dir"] / "model.safetensors").write_bytes(b"different-base")
    with pytest.raises(ValueError, match="mismatch"):
        load(bundle)


def test_artifact_checksum_and_path_escape_rejected(bundle):
    original = bundle["path"].read_bytes()
    update_registry(bundle, lambda value: value["groups"][0].update(sha256="0" * 64))
    with pytest.raises(ValueError, match="checksum"):
        load(bundle)
    bundle["path"].write_bytes(original)
    update_registry(bundle, lambda value: value["groups"][0].update(path="../outside.json"))
    with pytest.raises(ValueError, match="escapes"):
        load(bundle)


@pytest.mark.parametrize(
    "types,expected",
    [
        ([], "text"),
        (["image"], "image"),
        (["image", "image"], "image"),
        (["video"], "video"),
        (["audio"], "audio"),
        (["image", "audio"], "crossmodal"),
        (["video", "image"], "crossmodal"),
    ],
)
def test_cohort_uses_actual_media_types(types, expected):
    media = [Media(type=kind, uri="synthetic", sha256="a" * 64) for kind in types]
    assert calibration.infer_cohort(media) == expected


def test_registry_applies_joint_u_then_conditional_and_rejects_runtime_mapping(bundle):
    registry = load(bundle)
    question = Question(qid="q", type="choice", instruction="synthetic", options=["a", "b"])
    logits = {"A": -1.0, "B": -2.0, "U": -3.0}
    ids = {token: registry.token_map[token] for token in logits}
    result, metadata = registry.read([], question, logits, ids)
    assert result == read_probabilities(logits, ["A", "B"], temperature=2)
    assert result.p_insufficient > read_probabilities(logits, ["A", "B"]).p_insufficient
    assert result.probabilities["A"] < read_probabilities(logits, ["A", "B"]).probabilities["A"]
    assert sum(result.probabilities.values()) == pytest.approx(1)
    assert metadata["calibration_version"] == "synthetic-calibration"
    ids["A"] = ids["B"]
    with pytest.raises(ValueError, match="runtime answer token"):
        registry.read([], question, logits, ids)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FixtureTokenizer:
    template = "synthetic-template"

    def encode(self, text, **kwargs):
        return [ANSWERS.index(text)]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return self.template + messages[0]["content"]


@pytest.mark.anyio
@pytest.mark.parametrize("configured", [False, True])
async def test_http_default_and_explicit_registry_temperature(bundle, configured):
    registry = load(bundle)
    settings = Settings(
        upstream_url=bundle["upstream_url"],
        upstream_model="decision",
        model_version="synthetic-model",
        model_dir=bundle["model_dir"],
        raw_logprobs_confirmed=True,
    )

    def respond(request):
        payload = json.loads(request.content)
        assert payload["temperature"] == 1.0
        ids = payload["logprob_token_ids"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "logprobs": {
                            "top_logprobs": [
                                {f"token_id:{i}": -float(n + 1) for n, i in enumerate(ids)}
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1},
            },
        )

    app = create_app(
        settings,
        tokenizer=FixtureTokenizer(),
        transport=httpx.MockTransport(respond),
        calibration_registry=registry if configured else None,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture"
        ) as client,
    ):
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "synthetic",
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "synthetic",
                        "criteria": {"first": "a", "second": "b"},
                    }
                },
            },
        )
    assert response.status_code == 200
    answer = response.json()["answers"]["q"]
    expected = read_probabilities(
        {"A": -1.0, "B": -2.0, "U": -3.0}, ["A", "B"], temperature=2 if configured else 1
    )
    assert answer["p_insufficient"] == pytest.approx(expected.p_insufficient)
    assert answer["probabilities"]["first"] == pytest.approx(expected.probabilities["A"])
    assert answer["calibration"] == ("scalar_temperature" if configured else "unfitted")
    assert ("calibration_artifact_sha256" in answer) == configured


def test_gateway_wrong_model_version_rejected(bundle):
    registry = load(bundle)
    with pytest.raises(ValueError, match="gateway settings"):
        create_app(
            Settings(
                upstream_url=bundle["upstream_url"],
                upstream_model="decision",
                model_version="wrong",
                model_dir=bundle["model_dir"],
                raw_logprobs_confirmed=True,
            ),
            calibration_registry=registry,
        )


def test_changed_process_or_artifact_rejected_after_load(bundle, monkeypatch):
    registry = load(bundle)
    original_cmd = registry.process_command
    monkeypatch.setattr(calibration, "_cmdline", lambda pid: ("restarted",))
    with pytest.raises(ValueError, match="process identity changed"):
        registry.ensure_current()
    monkeypatch.setattr(calibration, "_cmdline", lambda pid: original_cmd)
    artifact = bundle["path"].parent / "text-choice.json"
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="files changed"):
        registry.ensure_current()


def test_all18_groups_include_unroutable_screenshot_bindings(bundle):
    registry = load(bundle)
    assert len(registry.groups) == 18
    assert len(calibration.REACHABLE_COHORTS) * len(calibration.PRIMITIVES) == 15
    assert all(("screenshot", primitive) in registry.groups for primitive in calibration.PRIMITIVES)


def test_runtime_template_and_tokenizer_mismatch_rejected(bundle):
    registry = load(bundle)
    with pytest.raises(ValueError, match="rendering template"):
        registry.verify_runtime(FixtureTokenizer(), "different-template")

    class DifferentTokenizer:
        def encode(self, text, **kwargs):
            return [-1]

    with pytest.raises(ValueError, match="gateway tokenizer differs"):
        registry.verify_runtime(DifferentTokenizer(), FixtureTokenizer.template)


def test_request_read_does_not_rehash_model_files(bundle, monkeypatch):
    registry = load(bundle)
    monkeypatch.setattr(calibration, "verified_assets", lambda *args: pytest.fail("rehash"))
    monkeypatch.setattr(calibration, "file_sha256", lambda *args: pytest.fail("rehash"))
    question = Question(qid="q", type="noul", instruction="synthetic")
    logits = {"yes": -1.0, "no": -2.0, "U": -3.0}
    registry.read([], question, logits, {token: registry.token_map[token] for token in logits})


@pytest.mark.anyio
async def test_stopped_process_returns503_without_upstream_call(bundle, monkeypatch):
    registry = load(bundle)
    app = create_app(
        Settings(
            upstream_url=bundle["upstream_url"],
            upstream_model="decision",
            model_version="synthetic-model",
            model_dir=bundle["model_dir"],
            raw_logprobs_confirmed=True,
        ),
        tokenizer=FixtureTokenizer(),
        transport=httpx.MockTransport(lambda request: pytest.fail("must not call backend")),
        calibration_registry=registry,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://fixture"
        ) as client,
    ):
        monkeypatch.setattr(calibration, "_cmdline", lambda pid: ("restarted",))
        response = await client.post(
            "/v1/systemone",
            json={
                "state": "synthetic",
                "questions": {"q": {"type": "noul", "instructions": "synthetic"}},
            },
        )
        assert response.status_code == 503
