from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from data.schema import Question
from scripts.s0_public_baselines import sha256_file
from scripts.s0_valen_public import (
    compare_repeatability,
    valen_probabilities,
    verify_qwen_base_assets,
)
from train.valen_inference import (
    Processor,
    Tokenizer,
    ValenInferenceError,
    assert_lora_key_set,
    assert_tensor_manifest_matches,
    bilinear_head_logits_lists,
    compile_qwen_question,
    softmax,
    valen_candidates,
)


class RecordingTokenizer(Tokenizer):
    def __init__(self) -> None:
        self.all_special_tokens: list[str] = ["<BAD>"]
        self.encoded: list[str] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        self.encoded.append(text)
        return [ord(ch) for ch in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        result: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


class FakeTensor:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values
        self.shape = (len(values), len(values[0]) if values else 0)


class FakeTorch:
    long = "long"

    @staticmethod
    def tensor(values: list[list[int]], *, dtype: str) -> FakeTensor:
        return FakeTensor(values)

    @staticmethod
    def ones(shape: tuple[int, int], *, dtype: str) -> FakeTensor:
        return FakeTensor([[1] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def ones_like(value: FakeTensor) -> FakeTensor:
        return FakeTensor([[1] * value.shape[1] for _ in range(value.shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int], *, dtype: str) -> FakeTensor:
        return FakeTensor([[0] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def cat(values: list[FakeTensor], *, dim: int) -> FakeTensor:
        assert dim == 1
        return FakeTensor([[item for tensor in values for item in tensor.values[0]]])


class FakeProcessor(Processor):
    def __init__(self) -> None:
        self.tokenizer = RecordingTokenizer()
        self.kwargs: dict[str, Any] | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        assert messages == [{"role": "user", "content": "state"}]
        return {
            "input_ids": FakeTensor([[11, 22]]),
            "attention_mask": FakeTensor([[1, 1]]),
            "video_metadata": ["must be stripped"],
        }


def _choice_question() -> Question:
    return Question.model_validate(
        {
            "qid": "q1",
            "type": "choice",
            "instruction": "Pick one",
            "options": ["Alpha", "Beta"],
        }
    )


def test_compiler_choice_positions_and_template_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    processor = FakeProcessor()
    compiled = compile_qwen_question(
        processor=processor, state="state", question=_choice_question()
    )

    assert processor.kwargs is not None
    assert processor.kwargs["add_generation_prompt"] is False
    assert processor.kwargs["enable_thinking"] is False
    assert processor.kwargs["processor_kwargs"]["return_metadata"] is True
    assert len(compiled.branches) == 1
    branch = compiled.branches[0]
    assert branch.keys == ["A", "B"]
    assert branch.descriptions == ["Alpha", "Beta"]
    assert "video_metadata" not in branch.inputs
    assert branch.inputs["attention_mask"].shape == branch.inputs["input_ids"].shape
    assert branch.candidate_positions[0] < branch.candidate_positions[1] < branch.decision_position
    assert processor.tokenizer.encoded[0] == "<|im_start|>user\n"
    assert "Task: choice\nQuestion: Pick one\nCandidates:\n" in processor.tokenizer.encoded
    assert "A: Alpha" in processor.tokenizer.encoded
    assert "B: Beta" in processor.tokenizer.encoded
    assert processor.tokenizer.encoded[-1] == "Decision:"


def test_score_compiler_uses_one_branch_per_level_without_level_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    processor = FakeProcessor()
    question = Question.model_validate(
        {
            "qid": "score",
            "type": "score",
            "instruction": "Rate",
            "score_levels": 3,
            "score_descriptions": ["Bad", "Okay", "Good"],
        }
    )

    compiled = compile_qwen_question(processor=processor, state="state", question=question)

    assert len(compiled.branches) == 3
    assert [branch.keys for branch in compiled.branches] == [["0"], ["1"], ["2"]]
    assert "Bad" in processor.tokenizer.encoded
    assert "0: Bad" not in processor.tokenizer.encoded
    assert "Okay" in processor.tokenizer.encoded
    assert "1: Okay" not in processor.tokenizer.encoded


def test_noul_candidates_are_true_false_fixed_descriptions() -> None:
    question = Question.model_validate({"qid": "n", "type": "noul", "instruction": "Is it true?"})

    keys, descriptions = valen_candidates(question)

    assert keys == ["true", "false"]
    assert descriptions == [
        "True / 是：满足问题中的条件。",
        "False / 否：不满足问题中的条件。",
    ]


def test_bilinear_head_uses_candidate_and_decision_last_hidden_tokens() -> None:
    logits = bilinear_head_logits_lists(
        candidate_hidden=[[1.0, 0.0], [0.0, 2.0]],
        decision_hidden=[3.0, 4.0],
        candidate_weight=[[1.0, 0.0], [0.0, 1.0]],
        decision_weight=[[2.0, 0.0], [0.0, 3.0]],
    )

    assert logits == pytest.approx([6 / math.sqrt(2), 24 / math.sqrt(2)])


def test_valen_probabilities_map_noul_true_false_to_public_yes_no() -> None:
    from data.jevbench import JevBenchTask

    task = JevBenchTask(
        id="n",
        split_name="original",
        family="f",
        group=None,
        state="state",
        question={
            "type": "noul",
            "instructions": "Is it true?",
            "criteria": {"true": "t", "false": "f"},
        },
        labels=("no", "yes"),
        expected="yes",
        source_sha256="0" * 64,
        project_tokens=("yes", "no"),
        source_to_project={"no": "no", "yes": "yes"},
        project_to_source={"yes": "yes", "no": "no"},
    )
    question = Question.model_validate({"qid": "n", "type": "noul", "instruction": "Is it true?"})

    result = valen_probabilities(task, question, [2.0, 0.0])

    assert result["u_supported"] is False
    assert result["p_U"] is None
    assert result["prediction_source"] == "yes"
    assert result["source_probabilities"]["yes"] > result["source_probabilities"]["no"]


def test_tensor_manifest_mismatch_is_rejected() -> None:
    expected = {"head.candidate.weight": ((256, 2048), "torch.float32")}
    observed = {"head.candidate.weight": ((128, 2048), "torch.float32")}

    with pytest.raises(ValenInferenceError, match="metadata mismatch"):
        assert_tensor_manifest_matches(expected, observed)

    with pytest.raises(ValenInferenceError, match="keys mismatch"):
        assert_tensor_manifest_matches(expected, {})


def _lora_keys(count: int) -> set[str]:
    return {
        f"backbone.language_model.base_model.model.layers.{i}.x.lora_A.default.weight"
        for i in range(count)
    }


def test_lora_key_set_must_match_exact_372_keys() -> None:
    keys = _lora_keys(372)
    assert_lora_key_set(keys, set(keys))

    with pytest.raises(ValenInferenceError, match="LoRA key mismatch"):
        assert_lora_key_set(keys, set(list(keys)[:-1]))

    with pytest.raises(ValenInferenceError, match="key count changed"):
        assert_lora_key_set(_lora_keys(371), _lora_keys(371))


def test_softmax_rejects_nonfinite_logits() -> None:
    with pytest.raises(ValenInferenceError, match="non-finite"):
        softmax([0.0, float("nan")])
    with pytest.raises(ValenInferenceError, match="non-finite"):
        softmax([0.0, float("inf")])


def test_repeatability_compares_raw_logits() -> None:
    left = [
        {
            "id": "x",
            "raw_logits": [1.0, 2.0],
            "source_probabilities": {"a": 1.0},
            "prediction_source": "a",
        }
    ]
    right = [
        {
            "id": "x",
            "raw_logits": [1.0, 2.1],
            "source_probabilities": {"a": 1.0},
            "prediction_source": "a",
        }
    ]

    report = compare_repeatability(left, right)

    assert report["passed"] is False
    assert report["first_mismatches"][0]["kind"] == "raw_logits"


def _git_blob(path: Path) -> str:
    import hashlib

    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def test_qwen_base_assets_verify_exact_manifest_path_and_architecture(tmp_path: Path) -> None:
    revision = "15852e8c16360a2fea060d615a32b45270f8a8fc"
    root = tmp_path / "models"
    model_dir = root / "Qwen__Qwen3.5-2B" / revision
    model_dir.mkdir(parents=True)
    files = {
        "config.json": json.dumps(
            {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}
        )
        + "\n",
        "tokenizer.json": "{}\n",
        "tokenizer_config.json": "{}\n",
        "model.safetensors": "fake pinned weight\n",
    }
    entries = []
    for name, text in files.items():
        path = model_dir / name
        path.write_text(text)
        is_weight = name.endswith(".safetensors")
        entries.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "required": True,
                "kind": "weights" if is_weight else "metadata",
                "lfs_sha256": sha256_file(path) if is_weight else None,
                "git_blob_sha1": None if is_weight else _git_blob(path),
            }
        )
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {
                        "id": "qwen3.5-2b-for-valen",
                        "repo_id": "Qwen/Qwen3.5-2B",
                        "revision": revision,
                        "license": "apache-2.0",
                        "files": entries,
                    }
                ],
            }
        )
    )

    evidence = verify_qwen_base_assets(
        asset_manifest=manifest, base_model_dir=model_dir, model_root=root
    )

    assert verify_qwen_base_assets(asset_manifest=manifest, base_model_dir=model_dir) == evidence
    assert evidence["revision"] == revision
    assert evidence["architecture"]["auto_map_absent"] is True
    with pytest.raises(ValenInferenceError, match="base model dir"):
        verify_qwen_base_assets(
            asset_manifest=manifest, base_model_dir=tmp_path / "wrong" / revision, model_root=root
        )


@pytest.mark.parametrize(
    "message", ["internal processor failure", "unexpected keyword enable_thinking"]
)
def test_template_type_error_fails_without_retry(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    class RaisingProcessor(FakeProcessor):
        calls = 0

        def apply_chat_template(
            self, messages: list[dict[str, Any]], **kwargs: Any
        ) -> dict[str, Any]:
            self.calls += 1
            assert kwargs["enable_thinking"] is False
            raise TypeError(message)

    processor = RaisingProcessor()
    with pytest.raises(TypeError, match=message):
        compile_qwen_question(processor=processor, state="state", question=_choice_question())
    assert processor.calls == 1
