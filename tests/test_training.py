"""CPU plumbing evidence only; these tests do not validate real E4B/GPU quality."""

import json
import math
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from scripts.train_smoke import main
from train.checkpoint import load_checkpoint, save_checkpoint, verify_checkpoint
from train.monitoring import (
    MODE_ENV,
    OFFICIAL_API_KEY_ENV,
    MonitoringError,
    SwanLabConfig,
    SwanLabTelemetry,
    start_swanlab,
)
from train.trainer import (
    SmokeConfig,
    attach_lora,
    data_bytes,
    decision_loss,
    log_training_metrics,
    prepare_examples,
    run,
    runtime_requirements,
    smoke_samples,
    verified_assets,
)


def test_smoke_samples_are_reproducible_text_train_and_all_targets():
    config = SmokeConfig()
    first = smoke_samples(config)
    assert data_bytes(first) == data_bytes(smoke_samples(config))
    assert len(first) == 100
    assert {s.questions[0].type for s in first} == {"choice", "noul", "score"}
    assert sum(s.questions[0].target.kind == "insufficient" for s in first) == 25
    assert all(s.meta.pool == "train" and len(s.questions) == 1 for s in first)


@pytest.mark.parametrize(
    "changes",
    [
        {"steps": 11},
        {"sample_count": 99},
        {"rank": 8},
        {"modality": "image"},
        {"learning_rate": math.nan},
        {"unknown": 1},
    ],
)
def test_config_refuses_scope_changes(changes):
    with pytest.raises(ValidationError):
        SmokeConfig.model_validate(changes)


def test_training_data_cannot_read_holdout_or_media(tmp_path):
    samples = smoke_samples(SmokeConfig())
    samples[0].meta.pool = "holdout"
    path = tmp_path / "dataset.jsonl"
    path.write_bytes(data_bytes(samples))
    with pytest.raises(ValueError, match="training-pool"):
        smoke_samples(SmokeConfig(), path)


class TextTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "official:" + messages[0]["content"]

    def encode(self, text, **kwargs):
        if text in {"yes", "no"}:
            return [2000 if text == "yes" else 2001]
        for word, token in [("yes", 2000), ("no", 2001)]:
            if text.startswith("official:") and text.endswith(word):
                return [ord(c) for c in text[: -len(word)]] + [token]
        return [ord(c) for c in text]


def test_examples_align_truth_and_abstention_without_supervision_in_prompt():
    examples = prepare_examples(smoke_samples(SmokeConfig()), TextTokenizer(), 2048)
    assert len(examples) == 100
    assert sum(examples[0]["target"]) == 1
    assert examples[3]["target"] == [0.0, 0.0, 1.0]
    assert "target" not in "".join(chr(i) for i in examples[0]["input_ids"])
    with pytest.raises(ValueError, match="truncation"):
        prepare_examples(smoke_samples(SmokeConfig()), TextTokenizer(), 32)


def test_generation_boundary_drift_is_rejected():
    class Drifting(TextTokenizer):
        def encode(self, text, **kwargs):
            ids = super().encode(text, **kwargs)
            return ids + [99] if text.startswith("official:") and text.endswith("A") else ids

    with pytest.raises(ValueError, match="generation boundary"):
        prepare_examples(smoke_samples(SmokeConfig()), Drifting(), 2048)


def test_loss_gradient_only_touches_legal_logits_and_moves_towards_target():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([99.0, 0.3, -0.1, 0.0, 100.0], requires_grad=True)
    loss = decision_loss(logits, [1, 2, 3], [1.0, 0.0, 0.0])
    expected = -torch.log_softmax(logits[[1, 2, 3]], dim=0)[0]
    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad[0] == 0 and logits.grad[4] == 0
    assert logits.grad[1] < 0 and logits.grad[2] > 0 and logits.grad[3] > 0
    updated = logits.detach() - 0.1 * logits.grad
    assert decision_loss(updated, [1, 2, 3], [1.0, 0.0, 0.0]) < loss
    insufficient = torch.zeros(3, requires_grad=True)
    decision_loss(insufficient, [0, 1, 2], [0.0, 0.0, 1.0]).backward()
    assert insufficient.grad[2] < 0


@pytest.mark.parametrize(
    "ids,target",
    [
        ([0, 0, 2], [1.0, 0.0, 0.0]),
        ([0, 1, 3], [1.0, 0.0, 0.0]),
        ([0, 1, 2], [0.2, 0.2, 0.2]),
        ([0, 1, 2], [math.nan, 0.0, 1.0]),
    ],
)
def test_loss_rejects_malformed_supervision(ids, target):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        decision_loss(torch.zeros(3), ids, target)


def test_loss_rejects_nonfinite_selected_logits():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="nonfinite"):
        decision_loss(torch.tensor([0.0, math.nan, 0.0]), [0, 1, 2], [1.0, 0.0, 0.0])


def test_peft_updates_only_decoder_lora_and_saves_safetensors(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")

    class Tiny(torch.nn.Module):
        """Contract fixture, not a substitute for the real Gemma smoke."""

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            layer = torch.nn.ModuleDict(
                {
                    name: torch.nn.Linear(4, 4, bias=False)
                    for name in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    )
                }
            )
            self.model.language_model.layers = torch.nn.ModuleList([layer])
            self.model.vision_tower = torch.nn.Linear(4, 4)
            self.lm_head = torch.nn.Linear(4, 5)

        def forward(self, x):
            for projection in self.model.language_model.layers[0].values():
                x = torch.tanh(projection(x))
            return self.lm_head(x)

    base = Tiny()
    original = {name: p.detach().clone() for name, p in base.named_parameters()}
    model = attach_lora(base, SmokeConfig())
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name, _ in trainable)
    optimizer = torch.optim.SGD([p for _, p in trainable], lr=0.1)
    loss = decision_loss(model(torch.ones(4)), [0, 1, 2], [1.0, 0.0, 0.0])
    loss.backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for _, p in trainable)
    optimizer.step()
    for name, p in model.named_parameters():
        if "lora_" not in name:
            original_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            assert torch.equal(p, original[original_name])
    model.save_pretrained(tmp_path, safe_serialization=True, save_embedding_layers=False)
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert not list(tmp_path.glob("*.bin"))


def test_missing_weights_preflight_is_structured_and_output_is_not_created(tmp_path):
    output = tmp_path / "run"
    report = run(
        SmokeConfig(), tmp_path / ("a" * 40), tmp_path / "missing.json", output, execute=False
    )
    assert report["status"] == "blocked" and report["exit_code"] == 1
    assert report["blockers"] and report["steps_completed"] == 0
    assert not output.exists()
    assert report["monitoring"]["enabled"] is False
    assert not any("monitoring:" in blocker for blocker in report["blockers"])


def test_swanlab_online_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name == "swanlab" else None
    )
    config = SwanLabConfig(mode="online", failure_policy="fail", logdir=tmp_path)
    assert config.blockers(env={}) == [
        f"monitoring: {OFFICIAL_API_KEY_ENV} is required for online mode"
    ]
    assert config.blockers(env={OFFICIAL_API_KEY_ENV: "present"}) == []


def test_swanlab_disabled_mode_never_requires_sdk(monkeypatch, tmp_path):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    config = SwanLabConfig.from_env({MODE_ENV: "disabled"})
    assert config.enabled is False
    assert config.blockers(env={}) == []
    report = run(
        SmokeConfig(),
        tmp_path / ("a" * 40),
        tmp_path / "missing.json",
        tmp_path / "out",
        execute=False,
        monitoring=config,
    )
    assert report["monitoring"]["mode"] == "disabled"
    assert not any("SwanLab" in blocker for blocker in report["blockers"])


def test_swanlab_env_opt_in_defaults_to_offline():
    config = SwanLabConfig.from_env({"LITE_OMNIJEV_SWANLAB": "1"})
    assert config.enabled is True
    assert config.mode == "offline"
    assert config.describe()["project"] == "Lite-OmniJev"
    assert config.describe()["workspace"] == "maxen"


def test_training_metric_wiring_uses_real_loop_values():
    records = []

    class Recorder:
        def log(self, metrics, step=None):
            records.append((step, metrics))

        def finish(self, report):
            del report

    report = {"steps_completed": 7}
    config = SmokeConfig(learning_rate=0.0002)
    log_training_metrics(Recorder(), config, report, mean_loss=1.25, grad_norm=0.75)
    assert records == [
        (
            7,
            {
                "train/loss": 1.25,
                "train/learning_rate": 0.0002,
                "train/grad_norm": 0.75,
                "train/steps_completed": 7.0,
            },
        )
    ]


def test_swanlab_failure_policy_warn_records_error(monkeypatch, tmp_path):
    sentinel = "sk-SENTINEL-should-not-appear"

    class BrokenRun:
        def log(self, metrics, step=None):
            del metrics, step
            raise RuntimeError(f"network down with {sentinel}")

    config = SwanLabConfig(mode="offline", failure_policy="warn", logdir=tmp_path)
    telemetry = SwanLabTelemetry(config, BrokenRun())
    telemetry.log({"train/loss": 1.0}, step=1)
    report = {}
    monkeypatch.setitem(
        sys.modules,
        "swanlab",
        SimpleNamespace(finish=lambda: None),
    )
    telemetry.finish(report)
    assert report["monitoring_error"] == "RuntimeError: SwanLab telemetry log failed"
    assert sentinel not in json.dumps(report)


def test_swanlab_failure_policy_fail_raises(tmp_path):
    sentinel = "sk-SENTINEL-should-not-appear"

    class BrokenRun:
        def log(self, metrics, step=None):
            del metrics, step
            raise RuntimeError(f"sdk refused metric {sentinel}")

    config = SwanLabConfig(mode="offline", failure_policy="fail", logdir=tmp_path)
    telemetry = SwanLabTelemetry(config, BrokenRun())
    with pytest.raises(MonitoringError) as excinfo:
        telemetry.log({"train/loss": 1.0}, step=1)
    assert str(excinfo.value) == "RuntimeError: SwanLab telemetry log failed"
    assert sentinel not in str(excinfo.value)


def test_swanlab_init_failure_report_is_sanitized(monkeypatch, tmp_path):
    sentinel = "sk-SENTINEL-should-not-appear"
    fake = SimpleNamespace(init=lambda **kwargs: (_ for _ in ()).throw(ValueError(sentinel)))
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    report = {
        "code_revision": "abcdef1234567890",
        "scope": "real smoke",
        "model_revision": "f" * 40,
        "config": SmokeConfig().model_dump(),
        "resume_supported": False,
    }
    config = SwanLabConfig(mode="offline", failure_policy="warn", logdir=tmp_path)
    telemetry = start_swanlab(config, report)
    telemetry.finish(report)
    assert report["monitoring_error"] == "ValueError: SwanLab telemetry init failed"
    assert sentinel not in json.dumps(report)


def test_trainer_error_from_monitoring_is_sanitized(monkeypatch, tmp_path):
    sentinel = "sk-SENTINEL-should-not-appear"
    fake = SimpleNamespace(init=lambda **kwargs: (_ for _ in ()).throw(ValueError(sentinel)))
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: object() if name == "swanlab" else None
    )
    monkeypatch.setattr("train.trainer.verified_assets", lambda model_dir, manifest_path: {})
    monkeypatch.setattr("train.trainer.runtime_requirements", lambda: {})
    monkeypatch.setattr("train.trainer.OfficialTokenizerAdapter", lambda model_dir: object())
    monkeypatch.setattr(
        "train.trainer.prepare_examples",
        lambda samples, tokenizer, max_length: [
            {"input_ids": [1], "token_ids": [2, 3], "target": [1, 0]}
        ],
    )
    report = run(
        SmokeConfig(),
        tmp_path / ("a" * 40),
        tmp_path / "missing.json",
        tmp_path / "out",
        execute=True,
        monitoring=SwanLabConfig(mode="offline", failure_policy="fail", logdir=tmp_path),
    )
    encoded = json.dumps(report)
    assert sentinel not in encoded
    assert report["error"] == "MonitoringError: ValueError: SwanLab telemetry init failed"


def test_start_swanlab_passes_project_workspace_and_mode(monkeypatch, tmp_path):
    calls = []

    class Run:
        def log(self, metrics, step=None):
            del metrics, step

    fake = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs) or Run())
    monkeypatch.setitem(sys.modules, "swanlab", fake)
    report = {
        "code_revision": "abcdef1234567890",
        "scope": "real smoke",
        "model_revision": "f" * 40,
        "config": SmokeConfig().model_dump(),
        "resume_supported": False,
    }
    config = SwanLabConfig(mode="offline", failure_policy="fail", logdir=tmp_path)
    telemetry = start_swanlab(config, report)
    telemetry.log({"train/loss": 1.0}, step=1)
    assert calls[0]["project"] == "Lite-OmniJev"
    assert calls[0]["workspace"] == "maxen"
    assert calls[0]["mode"] == "offline"
    assert calls[0]["public"] is False


def test_run_refuses_existing_output_without_overwriting(tmp_path):
    marker = tmp_path / "owned.txt"
    marker.write_text("preserve")
    report = run(SmokeConfig(), tmp_path, tmp_path / "missing", tmp_path, execute=True)
    assert report["exit_code"] == 1
    assert marker.read_text() == "preserve"
    assert not (tmp_path / "run.json").exists()


def test_run_resume_requires_checkpoint_under_output_dir(tmp_path):
    report = run(
        SmokeConfig(),
        tmp_path / ("a" * 40),
        tmp_path / "manifest.json",
        tmp_path / "out",
        execute=True,
        resume_from=tmp_path / "elsewhere" / "step-0001",
    )
    assert report["status"] == "blocked"
    assert "resume checkpoint must be inside output-dir/checkpoints" in report["error"]


def test_checkpoint_resume_matches_uninterrupted_cpu_run(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    config = SmokeConfig().model_dump()
    model_revision = "b" * 40
    data_sha256 = "data-hash"
    asset_hashes = {"model.safetensors": "asset-hash"}
    order = list(range(10))

    def train_until(output_dir, *, stop_after=None, resume_from=None):
        random = __import__("random")
        random.seed(1234)
        torch.manual_seed(1234)
        model = torch.nn.Linear(3, 3, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        losses = []
        start = 0
        if resume_from is not None:
            checkpoint = load_checkpoint(resume_from)
            start = verify_checkpoint(
                checkpoint,
                config=config,
                model_revision=model_revision,
                data_sha256=data_sha256,
                asset_hashes=asset_hashes,
                order=order,
            )
            losses = list(checkpoint["losses"])
            from train.checkpoint import restore_checkpoint

            restore_checkpoint(model, optimizer, checkpoint)
        for step in range(start, 10):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(3) + order[step] / 100
            target = [1.0, 0.0, 0.0] if step % 2 == 0 else [0.0, 1.0, 0.0]
            loss = decision_loss(model(x), [0, 1, 2], target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            checkpoint_path = save_checkpoint(
                output_dir,
                step=step + 1,
                config=config,
                model_revision=model_revision,
                data_sha256=data_sha256,
                asset_hashes=asset_hashes,
                order=order,
                losses=losses,
                model=model,
                optimizer=optimizer,
            )
            if stop_after == step + 1:
                return model, losses, checkpoint_path
        return model, losses, checkpoint_path

    uninterrupted, uninterrupted_losses, _ = train_until(tmp_path / "full")
    _, _, checkpoint_path = train_until(tmp_path / "partial", stop_after=4)
    resumed, resumed_losses, _ = train_until(tmp_path / "partial", resume_from=checkpoint_path)
    assert resumed_losses == pytest.approx(uninterrupted_losses)
    for left, right in zip(uninterrupted.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(left, right)


def test_checkpoint_refuses_provenance_mismatch(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    torch.manual_seed(1)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones(2)).sum()
    loss.backward()
    optimizer.step()
    path = save_checkpoint(
        tmp_path,
        step=1,
        config=SmokeConfig().model_dump(),
        model_revision="c" * 40,
        data_sha256="data-a",
        asset_hashes={"model": "hash"},
        order=[0],
        losses=[float(loss.detach())],
        model=model,
        optimizer=optimizer,
    )
    checkpoint = load_checkpoint(path)
    with pytest.raises(ValueError, match="data hash"):
        verify_checkpoint(
            checkpoint,
            config=SmokeConfig().model_dump(),
            model_revision="c" * 40,
            data_sha256="data-b",
            asset_hashes={"model": "hash"},
            order=[0],
        )
    with pytest.raises(FileExistsError):
        save_checkpoint(
            tmp_path,
            step=1,
            config=SmokeConfig().model_dump(),
            model_revision="c" * 40,
            data_sha256="data-a",
            asset_hashes={"model": "hash"},
            order=[0],
            losses=[float(loss.detach())],
            model=model,
            optimizer=optimizer,
        )


def test_cli_invalid_config_is_structured(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text('{"steps": 100}')
    report = tmp_path / "report.json"
    code = main(
        [
            "preflight",
            "--config",
            str(config),
            "--model-dir",
            str(tmp_path),
            "--asset-manifest",
            str(tmp_path / "manifest.json"),
            "--report",
            str(report),
        ]
    )
    assert code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert json.loads(report.read_text())["exit_code"] == 1


def test_unpinned_or_wrong_model_assets_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": []}))
    with pytest.raises(ValueError, match="pinned"):
        verified_assets(tmp_path / "main", manifest)


def test_missing_optional_dependencies_are_reportable(monkeypatch):
    import importlib.metadata

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        runtime_requirements()


def test_runtime_refuses_non_cuda_even_with_dependencies(monkeypatch):
    import importlib
    import importlib.metadata
    from types import SimpleNamespace

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            Gemma4ForConditionalGeneration=object,
            get_peft_model=object,
            LoraConfig=object,
        ),
    )
    with pytest.raises(ValueError, match="CUDA is required"):
        runtime_requirements()


def test_soft_target_remains_distribution_with_zero_abstention():
    from data.schema import Sample

    samples = smoke_samples(SmokeConfig())
    payload = samples[0].model_dump()
    payload["questions"][0]["target"] = {
        "kind": "soft",
        "probs": [0.6, 0.3, 0.1],
        "source": "teacher:test-only",
    }
    example = prepare_examples([Sample.model_validate(payload)], TextTokenizer(), 2048)[0]
    assert example["target"] == [0.6, 0.3, 0.1, 0.0]


def test_duplicate_samples_are_rejected_before_training(tmp_path):
    samples = smoke_samples(SmokeConfig())
    samples[1] = samples[0]
    path = tmp_path / "duplicates.jsonl"
    path.write_bytes(data_bytes(samples))
    with pytest.raises(ValueError, match="duplicate sample ID"):
        smoke_samples(SmokeConfig(), path)


def test_installed_transformers_exposes_real_gemma_class():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    assert transformers.Gemma4ForConditionalGeneration.__name__ == "Gemma4ForConditionalGeneration"


def test_asset_manifest_must_cover_exactly_indexed_shards(tmp_path):
    import hashlib

    revision = "a" * 40
    model_dir = tmp_path / revision
    model_dir.mkdir()
    contents = {
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "config.json": json.dumps(
            {"model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]}
        ),
        "model-00001-of-00002.safetensors": "first-test-fixture",
        "model-00002-of-00002.safetensors": "second-test-fixture",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"weight": "model-00001-of-00002.safetensors"}}
        ),
    }
    files = []
    for name, value in contents.items():
        payload = value.encode()
        (model_dir / name).write_bytes(payload)
        files.append(
            {
                "path": name,
                "size": len(payload),
                "kind": "weights" if name.endswith(".safetensors") else "metadata",
                "upstream_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "repo_id": "google/gemma-4-E4B-it",
                        "revision": revision,
                        "license": "apache-2.0",
                        "files": files,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="unverified shards"):
        verified_assets(model_dir, manifest)
