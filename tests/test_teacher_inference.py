"""Teacher inference adapter tests use fakes and tiny local fixtures only."""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import types
import wave
from pathlib import Path
from typing import Any

import pytest

from train.teacher_inference import (
    OFFICIAL_JEV_OMNI_SHA256,
    JevOmniTeacherForward,
    TeacherDecisionHead,
    _video_frames,
    build_teacher_processor_inputs,
    expected_head_sha256_from_report,
    find_teacher_decoder,
    inspect_safe_teacher_head,
    load_safe_teacher_head_module,
    official_teacher_prompt,
    prepared_teacher_input,
    run_teacher_preflight,
    teacher_forward_readiness_report,
    validate_teacher_options,
    validate_teacher_source_tree,
)


class FakeTensor:
    shape = (1, 3)
    dtype = "torch.int64"


class RecordingProcessor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}


def test_prompt_matches_pinned_upstream_text() -> None:
    assert official_teacher_prompt("state", "pick?", ["red", "blue"]) == (
        "state\n\n---\n\nQUESTION: pick?\n\nOPTIONS:\n1. red\n2. blue\n\n"
        "Reply with only the number of the correct option (1-2).\n"
        "Output a single number and nothing else."
    )


def test_teacher_options_preserve_upstream_range_and_reject_ambiguous_values() -> None:
    assert len(validate_teacher_options([str(index) for index in range(256)])) == 256
    with pytest.raises(ValueError, match="2-256"):
        validate_teacher_options(["only"])
    with pytest.raises(ValueError, match="2-256"):
        validate_teacher_options([str(index) for index in range(257)])
    with pytest.raises(ValueError, match="distinct"):
        validate_teacher_options(["same", "same"])


def test_processor_call_uses_official_shape_and_disables_thinking() -> None:
    processor = RecordingProcessor()
    with prepared_teacher_input(
        state="state", question="question", options=("left", "right")
    ) as prepared:
        result = build_teacher_processor_inputs(processor, prepared)
    assert result.summary == {
        "modality": "text",
        "option_count": 2,
        "keys": {
            "attention_mask": {"kind": "tensor", "shape": [1, 3], "dtype": "torch.int64"},
            "input_ids": {"kind": "tensor", "shape": [1, 3], "dtype": "torch.int64"},
        },
        "thinking_disabled": True,
        "tokenize": True,
        "contains_payload_values": False,
    }
    assert processor.calls[0]["kwargs"] == {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
    assert processor.calls[0]["messages"][0]["role"] == "user"


def test_audio_input_consumes_prepared_teacher_audio_context(tmp_path: Path, monkeypatch) -> None:
    numpy = pytest.importorskip("numpy")
    source = tmp_path / "source.wav"
    source.write_bytes(b"placeholder")
    prepared = tmp_path / "prepared.wav"
    with wave.open(str(prepared), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x00\x00\xff\x7f")
    calls = []

    @contextlib.contextmanager
    def fake_prepared_audio(path: Path, **_kwargs: Any):
        calls.append(path)
        yield prepared

    monkeypatch.setattr("train.teacher_inference.prepared_teacher_audio", fake_prepared_audio)
    with prepared_teacher_input(
        state="s", question="q", options=("a", "b"), modality="audio", media=source
    ) as payload:
        content = payload.messages[0]["content"]
        assert content[0]["type"] == "audio"
        assert isinstance(content[0]["audio"], numpy.ndarray)
        assert content[0]["audio"].dtype == numpy.float32
        assert content[0]["audio"].tolist() == [0.0, pytest.approx(32767 / 32768)]
        assert content[1]["type"] == "text"
    assert calls == [source.resolve()]


def test_pinned_video_sampling_rejects_duplicate_indices_before_decode(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeCapture:
        grabbed = 0
        read_called = False

        def __init__(self, _path: str) -> None:
            pass

        def get(self, _prop: int) -> int:
            return 17

        def grab(self) -> bool:
            type(self).grabbed += 1
            return True

        def read(self) -> tuple[bool, object]:
            type(self).read_called = True
            return True, object()

        def release(self) -> None:
            pass

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.CAP_PROP_FRAME_COUNT = 7
    fake_cv2.COLOR_BGR2RGB = 4
    fake_cv2.VideoCapture = FakeCapture
    fake_cv2.cvtColor = lambda frame, _code: frame
    fake_pil = types.ModuleType("PIL")
    fake_pil.__path__ = []
    fake_image = types.ModuleType("PIL.Image")
    fake_image.fromarray = lambda frame: frame
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "PIL.Image", fake_image)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")

    with pytest.raises(
        ValueError, match="pinned sampling protocol yields 9 unique frames, requested 16"
    ):
        _video_frames(video, count=16)

    assert FakeCapture.grabbed == 0
    assert FakeCapture.read_called is False


def _source_tree(root: Path) -> Path:
    root.mkdir()
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
        "decision_config.json",
        "processor_config.json",
    ):
        (root / name).write_text("{}")
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Gemma4UnifiedForConditionalGeneration"],
                "model_type": "gemma4_unified",
            }
        )
    )
    (root / "decision_config.json").write_text(
        json.dumps({"hidden_size": 3840, "output_classes": 256, "dtype": "float32"})
    )
    (root / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gemma4UnifiedProcessor",
                "feature_extractor": {"sampling_rate": 16000},
                "video_processor": {"num_frames": 32},
            }
        )
    )
    (root / "jev_omni.py").write_text("fixture")
    return root


def test_source_tree_requires_pinned_upstream_hash(tmp_path: Path) -> None:
    source = _source_tree(tmp_path / "source")
    with pytest.raises(ValueError, match="SHA256"):
        validate_teacher_source_tree(source)
    report = validate_teacher_source_tree(
        source,
        expected_jev_omni_sha256=__import__("hashlib").sha256(b"fixture").hexdigest(),
    )
    assert report["upstream_custom_code_executed"] is False
    assert report["decision_config"]["output_classes"] == 256


def test_safe_head_inspection_uses_safetensors_and_expected_spec(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    head = tmp_path / "head.safetensors"
    tensors = {
        "linear.bias": torch.zeros(2, dtype=torch.float32),
        "linear.weight": torch.ones((2, 3), dtype=torch.float32),
        "mu": torch.zeros((1, 3), dtype=torch.float32),
        "sd": torch.ones((1, 3), dtype=torch.float32),
    }
    safetensors.save_file(tensors, str(head))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in tensors.items()
                },
            }
        )
    )
    report = inspect_safe_teacher_head(
        head, spec, expected_sha256=hashlib.sha256(head.read_bytes()).hexdigest()
    )
    assert report["torch_load_used"] is False
    assert report["tensor_count"] == 4
    assert set(report["tensors"]) == set(tensors)


def test_safe_head_inspection_requires_hash_anchor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    head = tmp_path / "head.safetensors"
    safetensors.save_file({"x": torch.ones(1, dtype=torch.float32)}, str(head))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {"schema_version": 1, "tensors": {"x": {"shape": [1], "dtype": "torch.float32"}}}
        )
    )
    with pytest.raises(ValueError, match="SHA256"):
        inspect_safe_teacher_head(head, spec, expected_sha256="0" * 64)


def test_expected_head_sha256_defaults_to_conversion_report(tmp_path: Path) -> None:
    report = tmp_path / "conversion.json"
    report.write_text(
        json.dumps({"status": "converted", "exit_code": 0, "output_sha256": "a" * 64})
    )
    assert expected_head_sha256_from_report(report) == "a" * 64
    report.write_text(json.dumps({"status": "blocked", "exit_code": 1, "output_sha256": "a" * 64}))
    with pytest.raises(ValueError, match="not a passing"):
        expected_head_sha256_from_report(report)


def test_preflight_report_blocks_without_leaking_payloads(tmp_path: Path, monkeypatch) -> None:
    source = _source_tree(tmp_path / "source")
    monkeypatch.setattr(
        "train.teacher_inference.OFFICIAL_JEV_OMNI_SHA256",
        OFFICIAL_JEV_OMNI_SHA256,
    )
    report = run_teacher_preflight(
        source_dir=source,
        head_path=tmp_path / "missing.safetensors",
        spec_path=tmp_path / "missing-spec.json",
        conversion_report=tmp_path / "missing-conversion.json",
        synthetic_modality="text",
    )
    assert report["status"] == "blocked"
    assert report["gpu_used"] is False
    assert report["model_backbone_loaded"] is False
    assert "Which option" not in json.dumps(report)


class FakeBackbone:
    def __init__(self, hidden: object) -> None:
        self.layers = [object()]
        self.hidden = hidden
        self.hook = None
        self.remove_count = 0

    def register_forward_hook(self, hook):
        self.hook = hook
        owner = self

        class Handle:
            def remove(self) -> None:
                owner.remove_count += 1
                owner.hook = None

        return Handle()

    def __call__(self, **_kwargs: object):
        output = types.SimpleNamespace(last_hidden_state=self.hidden)
        if self.hook is not None:
            self.hook(self, (), output)
        return output


class FakeForwardModel:
    def __init__(self, hidden: object) -> None:
        self.model = FakeBackbone(hidden)
        self.calls: list[dict[str, object]] = []

    def forward(self, input_ids: object, *, use_cache: bool = True, logits_to_keep: int = 0):
        self.calls.append(
            {
                "input_ids": input_ids,
                "use_cache": use_cache,
                "logits_to_keep": logits_to_keep,
            }
        )
        return self.model(input_ids=input_ids)

    __call__ = forward


def test_forward_adapter_captures_decoder_hidden_and_clears_hook_state() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[99.0, 99.0], [2.0, 3.0]]])
    model = FakeForwardModel(hidden)
    decoder_path, decoder = find_teacher_decoder(model)
    assert decoder_path == "model"
    head = TeacherDecisionHead(hidden_size=2)
    with torch.no_grad():
        head.mu.zero_()
        head.sd.fill_(1.0)
        head.linear.weight.zero_()
        head.linear.bias.zero_()
        head.linear.weight[0] = torch.tensor([1.0, 0.0])
        head.linear.weight[1] = torch.tensor([0.0, 1.0])
        head.linear.weight[2] = torch.tensor([-5.0, -5.0])
    processor = RecordingProcessor()
    adapter = JevOmniTeacherForward(
        model=model, head=head, processor=processor, decoder=decoder, device="cpu"
    )
    prepared = types.SimpleNamespace(modality="text", option_count=2)
    result = adapter.predict_prepared(
        prepared, {"input_ids": torch.tensor([[1, 2, 3]])}, ["first", "second"]
    )
    assert result.prediction == "second"
    assert result.prediction_index == 1
    assert result.probabilities["second"] > result.probabilities["first"]
    assert len(model.calls) == 1
    assert torch.equal(model.calls[0]["input_ids"], torch.tensor([[1, 2, 3]]))
    assert model.calls[0]["use_cache"] is False
    assert model.calls[0]["logits_to_keep"] == 1
    assert adapter._capture == {}
    adapter.close()
    assert decoder.remove_count == 1
    assert decoder.hook is None


def test_forward_adapter_rejects_stale_or_missing_decoder_capture() -> None:
    torch = pytest.importorskip("torch")

    class NoHookModel:
        def forward(self, input_ids: object, *, use_cache: bool = True):
            return types.SimpleNamespace()

        __call__ = forward

    class NoHookDecoder:
        layers = [object()]

        def register_forward_hook(self, _hook):
            return types.SimpleNamespace(remove=lambda: None)

    head = TeacherDecisionHead(hidden_size=2)
    adapter = JevOmniTeacherForward(
        model=NoHookModel(),
        head=head,
        processor=RecordingProcessor(),
        decoder=NoHookDecoder(),
        device="cpu",
    )
    adapter._capture["hidden"] = torch.ones((1, 2))
    with pytest.raises(RuntimeError, match="decoder hidden capture was empty"):
        adapter.predict_prepared(
            types.SimpleNamespace(modality="text", option_count=2),
            {"input_ids": torch.tensor([[1]])},
            ["a", "b"],
        )
    assert adapter._capture == {}


def test_load_safe_teacher_head_module_uses_converted_head_semantics(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    head_path = tmp_path / "head.safetensors"
    tensors = {
        "linear.bias": torch.tensor([0.0, 0.0, 9.0], dtype=torch.float32),
        "linear.weight": torch.eye(3, dtype=torch.float32),
        "mu": torch.zeros((1, 3), dtype=torch.float32),
        "sd": torch.ones((1, 3), dtype=torch.float32),
    }
    safetensors.save_file(tensors, str(head_path))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in tensors.items()
                },
            }
        )
    )
    head = load_safe_teacher_head_module(
        head_path,
        spec,
        expected_sha256=hashlib.sha256(head_path.read_bytes()).hexdigest(),
        device="cpu",
    )
    logits = head(torch.ones((1, 3), dtype=torch.float32), torch.tensor([2]))
    assert logits.shape == (1, 3)
    assert logits[0, 2].item() == pytest.approx(-1e30)


def test_forward_readiness_blocks_before_from_pretrained_without_cuda(
    tmp_path: Path, monkeypatch
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    source = _source_tree(tmp_path / "source")
    expected_source_sha = hashlib.sha256(b"fixture").hexdigest()
    monkeypatch.setattr("train.teacher_inference.OFFICIAL_JEV_OMNI_SHA256", expected_source_sha)
    head_path = tmp_path / "head.safetensors"
    tensors = {
        "linear.bias": torch.zeros(2, dtype=torch.float32),
        "linear.weight": torch.ones((2, 3), dtype=torch.float32),
        "mu": torch.zeros((1, 3), dtype=torch.float32),
        "sd": torch.ones((1, 3), dtype=torch.float32),
    }
    safetensors.save_file(tensors, str(head_path))
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tensors": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in tensors.items()
                },
            }
        )
    )
    conversion = tmp_path / "conversion.json"
    conversion.write_text(
        json.dumps(
            {
                "status": "converted",
                "exit_code": 0,
                "output_sha256": hashlib.sha256(head_path.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report = teacher_forward_readiness_report(
        source_dir=source,
        head_path=head_path,
        spec_path=spec,
        conversion_report=conversion,
    )

    assert report["status"] == "blocked"
    assert report["loader_stage"] == "pre_from_pretrained"
    assert report["model_backbone_loaded"] is False
    assert report["model_forward_run"] is False
    assert report["trust_remote_code"] is False
    assert report["local_files_only"] is True
    assert report["use_safetensors_for_model"] is True
    assert "requires CUDA before from_pretrained" in report["error"]
