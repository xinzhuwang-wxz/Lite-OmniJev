"""Safe Valen Qwen inference helpers from pinned Apache-2.0 audited source.

This module re-implements the small inference surface needed for public JevBench
baselines. It does not import or execute upstream Valen modules and never reads
raw pickle checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from data.schema import Question, answer_tokens

VALEN_SOURCE_REPO = "Valen-Team/Valen-Preview-0923"
VALEN_SOURCE_REVISION = "30096aaabb2689614c83a79f64f38c865c2ce1c1"
VALEN_SOURCE_CODE_REPO = "Liuziyu77/Valen"
VALEN_SOURCE_CODE_REVISION = "42a6a10e127e9248a6d7824299551b78af9fcb97"
VALEN_BASE_REPO = "Qwen/Qwen3.5-2B"
VALEN_BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
VALEN_WEIGHTS_SHA256 = "27b8f6cd4874a356af8a5d6a3676d2a9a03e4b3301d066d13645f673f72f4468"
VALEN_TENSOR_COUNT = 428
VALEN_GROUP_COUNTS = {"lora": 372, "vision": 48, "merger": 6, "head": 2}
VALEN_LORA_TENSOR_COUNT = 372
MAX_LENGTH = 8192


class ValenInferenceError(RuntimeError):
    """Raised when Valen public inference evidence or inputs are invalid."""


class Tokenizer(Protocol):
    all_special_tokens: list[str]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> Mapping[str, Any]: ...


class Processor(Protocol):
    tokenizer: Tokenizer

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ValenBranch:
    inputs: dict[str, Any]
    candidate_positions: list[int]
    decision_position: int
    keys: list[str]
    descriptions: list[str]


@dataclass(frozen=True)
class CompiledValenQuestion:
    qid: str
    kind: str
    keys: list[str]
    descriptions: list[str]
    branches: list[ValenBranch]
    logical_tokens: int
    compute_tokens: int
    media: list[dict[str, Any]]
    enable_thinking_false_used: bool


@dataclass(frozen=True)
class ValenEvidence:
    spec_sha256: str
    conversion_report_sha256: str
    weights_sha256: str
    tensor_count: int
    groups: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValenInferenceError(f"JSON root must be an object: {path}")
    return data


def expected_tensor_meta(spec: Mapping[str, Any]) -> dict[str, tuple[tuple[int, ...], str]]:
    tensors = spec.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValenInferenceError("Valen spec lacks tensor manifest")
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for name, raw in tensors.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise ValenInferenceError("invalid tensor spec entry")
        shape = raw.get("shape")
        dtype = raw.get("dtype")
        if not isinstance(shape, list) or not isinstance(dtype, str):
            raise ValenInferenceError("invalid tensor spec shape/dtype")
        result[name] = (tuple(int(dim) for dim in shape), dtype)
    return result


def assert_tensor_manifest_matches(
    expected: Mapping[str, tuple[tuple[int, ...], str]],
    observed: Mapping[str, tuple[tuple[int, ...], str]],
) -> None:
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValenInferenceError(
            f"Valen tensor keys mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    mismatched = [
        name for name in sorted(expected) if tuple(observed[name]) != tuple(expected[name])
    ]
    if mismatched:
        raise ValenInferenceError(f"Valen tensor metadata mismatch: {mismatched[:5]}")


def lora_tensor_keys(keys: Sequence[str]) -> set[str]:
    return {key for key in keys if ".lora_A." in key or ".lora_B." in key}


def assert_lora_key_set(expected: set[str], observed: set[str]) -> None:
    if len(expected) != VALEN_LORA_TENSOR_COUNT:
        raise ValenInferenceError(f"Valen spec LoRA key count changed: {len(expected)}")
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValenInferenceError(
            f"Valen LoRA key mismatch: missing={missing[:5]} extra={extra[:5]}"
        )


def safetensors_manifest(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    safe_open = __import__("safetensors", fromlist=["safe_open"]).safe_open
    observed: dict[str, tuple[tuple[int, ...], str]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            observed[name] = (tuple(int(dim) for dim in tensor.shape), str(tensor.dtype))
    return observed


def verify_valen_evidence(
    *, spec_path: Path, conversion_report_path: Path, weights_path: Path
) -> ValenEvidence:
    spec = load_json(spec_path)
    report = load_json(conversion_report_path)
    if spec.get("source_repo") != VALEN_SOURCE_REPO:
        raise ValenInferenceError("unexpected Valen source repo")
    if spec.get("source_revision") != VALEN_SOURCE_REVISION:
        raise ValenInferenceError("unexpected Valen source revision")
    if spec.get("source_code_repo") != VALEN_SOURCE_CODE_REPO:
        raise ValenInferenceError("unexpected Valen source code repo")
    if spec.get("source_code_revision") != VALEN_SOURCE_CODE_REVISION:
        raise ValenInferenceError("unexpected Valen source code revision")
    if spec.get("base_repo") != VALEN_BASE_REPO or spec.get("base_revision") != VALEN_BASE_REVISION:
        raise ValenInferenceError("unexpected Valen base model")
    if report.get("status") != "converted" or report.get("output_sha256") != VALEN_WEIGHTS_SHA256:
        raise ValenInferenceError("Valen conversion report is not the pinned converted artifact")
    if (
        report.get("tensor_count") != VALEN_TENSOR_COUNT
        or spec.get("tensor_count") != VALEN_TENSOR_COUNT
    ):
        raise ValenInferenceError("Valen tensor count changed")
    groups = report.get("groups")
    if not isinstance(groups, Mapping):
        raise ValenInferenceError("Valen conversion report lacks groups")
    for group, count in VALEN_GROUP_COUNTS.items():
        if not isinstance(groups.get(group), Mapping) or groups[group].get("count") != count:
            raise ValenInferenceError(f"Valen tensor group changed: {group}")
    actual_sha = sha256_file(weights_path)
    if actual_sha != VALEN_WEIGHTS_SHA256:
        raise ValenInferenceError("Valen safetensors SHA256 mismatch")
    expected = expected_tensor_meta(spec)
    observed = safetensors_manifest(weights_path)
    assert_tensor_manifest_matches(expected, observed)
    return ValenEvidence(
        spec_sha256=sha256_file(spec_path),
        conversion_report_sha256=sha256_file(conversion_report_path),
        weights_sha256=actual_sha,
        tensor_count=len(observed),
        groups=dict(groups),
    )


def valen_candidates(question: Question) -> tuple[list[str], list[str]]:
    if question.type == "noul":
        return ["true", "false"], [
            "True / 是：满足问题中的条件。",
            "False / 否：不满足问题中的条件。",
        ]
    if question.type == "choice":
        tokens = list(answer_tokens(question))
        descriptions = list(question.options or [])
        return tokens, descriptions
    if question.type == "score":
        tokens = list(answer_tokens(question))
        descriptions = list(question.score_descriptions or tokens)
        return tokens, descriptions
    raise ValenInferenceError(f"unsupported question type: {question.type}")


def _state_messages(state: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": state}]


def _messages_for_state(state: str | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(state, str):
        return _state_messages(state)
    return [dict(message) for message in state]


def _base_from_processor(
    processor: Processor,
    state: str | Sequence[Mapping[str, Any]],
    media_kwargs: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={**dict(media_kwargs or {}), "return_metadata": True},
    )
    try:
        base = dict(
            processor.apply_chat_template(
                _messages_for_state(state), enable_thinking=False, **kwargs
            )
        )
        thinking_false = True
    except TypeError:
        base = dict(processor.apply_chat_template(_messages_for_state(state), **kwargs))
        thinking_false = False
    media: list[dict[str, Any]] = []
    base.pop("video_metadata", None)
    return base, media, thinking_false


def _contains_special(tokenizer: Tokenizer, text: str) -> bool:
    return any(token in text for token in getattr(tokenizer, "all_special_tokens", []))


def _tensor_cat(torch: Any, base_ids: Any, suffix: list[int]) -> Any:
    return torch.cat([base_ids, torch.tensor([suffix], dtype=torch.long)], dim=1)


def _append_suffix(suffix: list[int], tokenizer: Tokenizer, text: str) -> None:
    suffix.extend(tokenizer.encode(text, add_special_tokens=False))


def compile_qwen_question(
    *,
    processor: Processor,
    state: str | Sequence[Mapping[str, Any]],
    question: Question,
    max_length: int = MAX_LENGTH,
    media_kwargs: Mapping[str, Any] | None = None,
) -> CompiledValenQuestion:
    torch = __import__("torch")
    tokenizer = processor.tokenizer
    base, media, thinking_false = _base_from_processor(processor, state, media_kwargs)
    if "input_ids" not in base:
        raise ValenInferenceError("processor did not return input_ids")
    base_length = int(base["input_ids"].shape[1])
    keys, descriptions = valen_candidates(question)
    if len(keys) != len(descriptions):
        raise ValenInferenceError("candidate keys/descriptions mismatch")
    if any(_contains_special(tokenizer, value) for value in [question.instruction, *descriptions]):
        raise ValenInferenceError("reserved tokenizer control token in question/criteria")
    groups = list(zip(keys, descriptions, strict=True))
    branch_groups = [[pair] for pair in groups] if question.type == "score" else [groups]
    branches: list[ValenBranch] = []
    logical_tokens = base_length
    compute_tokens = 0
    for group in branch_groups:
        suffix: list[int] = []

        suffix.extend(tokenizer.encode("<|im_start|>user\n", add_special_tokens=False))
        _append_suffix(
            suffix,
            tokenizer,
            "Task: " + question.type + "\nQuestion: " + question.instruction + "\nCandidates:\n",
        )
        positions: list[int] = []
        group_keys: list[str] = []
        group_descriptions: list[str] = []
        for key, description in group:
            prefix = key + ": " if question.type != "score" else ""
            _append_suffix(suffix, tokenizer, prefix + description)
            positions.append(base_length + len(suffix) - 1)
            group_keys.append(key)
            group_descriptions.append(description)
            _append_suffix(suffix, tokenizer, "\n")
        _append_suffix(suffix, tokenizer, "Decision:")
        length = base_length + len(suffix)
        if length > max_length:
            raise ValenInferenceError(
                f"{question.qid}: {length} tokens exceed max_length={max_length}; no truncation"
            )
        inputs = dict(base)
        inputs["input_ids"] = _tensor_cat(torch, base["input_ids"], suffix)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        if "mm_token_type_ids" in base:
            inputs["mm_token_type_ids"] = torch.cat(
                [base["mm_token_type_ids"], torch.zeros((1, len(suffix)), dtype=torch.long)],
                dim=1,
            )
        branches.append(
            ValenBranch(
                inputs=inputs,
                candidate_positions=positions,
                decision_position=length - 1,
                keys=group_keys,
                descriptions=group_descriptions,
            )
        )
        logical_tokens += len(suffix)
        compute_tokens += length
    return CompiledValenQuestion(
        qid=question.qid,
        kind=question.type,
        keys=keys,
        descriptions=descriptions,
        branches=branches,
        logical_tokens=logical_tokens,
        compute_tokens=compute_tokens,
        media=media,
        enable_thinking_false_used=thinking_false,
    )


def bilinear_head_logits_lists(
    candidate_hidden: Sequence[Sequence[float]],
    decision_hidden: Sequence[float],
    candidate_weight: Sequence[Sequence[float]],
    decision_weight: Sequence[Sequence[float]],
) -> list[float]:
    def project(vector: Sequence[float], weight: Sequence[Sequence[float]]) -> list[float]:
        return [
            math.fsum(float(a) * float(b) for a, b in zip(row, vector, strict=True))
            for row in weight
        ]

    decision = project(decision_hidden, decision_weight)
    scale = math.sqrt(float(len(candidate_weight)))
    return [
        math.fsum(
            a * b for a, b in zip(project(candidate, candidate_weight), decision, strict=True)
        )
        / scale
        for candidate in candidate_hidden
    ]


def bilinear_head_logits(
    hidden: Any,
    candidate_positions: Sequence[int],
    decision_position: int,
    candidate_weight: Any,
    decision_weight: Any,
) -> Any:
    torch = __import__("torch")
    candidate_hidden = hidden[0, list(candidate_positions)].float()
    decision_hidden = hidden[0, decision_position].float()
    candidates = torch.nn.functional.linear(candidate_hidden, candidate_weight.float())
    decision = torch.nn.functional.linear(decision_hidden, decision_weight.float())
    return (candidates * decision).sum(-1) / math.sqrt(float(candidate_weight.shape[0]))


class ValenDecisionHead:
    def __init__(self, tensors: Mapping[str, Any]) -> None:
        self.candidate_weight = tensors["head.candidate.weight"].float()
        self.decision_weight = tensors["head.decision.weight"].float()

    def logits(self, hidden: Any, branch: ValenBranch) -> Any:
        return bilinear_head_logits(
            hidden,
            branch.candidate_positions,
            branch.decision_position,
            self.candidate_weight,
            self.decision_weight,
        )


class ValenRuntime:
    def __init__(self, backbone: Any, head: ValenDecisionHead, torch: Any) -> None:
        self.backbone = backbone
        self.head = head
        self.torch = torch

    def logits(self, compiled: CompiledValenQuestion) -> list[float]:
        outputs: list[float] = []
        device = next(self.backbone.parameters()).device
        for branch in compiled.branches:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in branch.inputs.items()
            }
            with self.torch.inference_mode():
                hidden = self.backbone(
                    **inputs, use_cache=False, return_dict=True
                ).last_hidden_state
                logits = self.head.logits(hidden, branch).float().detach().cpu().tolist()
            outputs.extend(float(value) for value in logits)
        return outputs


def load_valen_runtime(
    *, base_model_dir: Path, weights_path: Path, dtype_name: str = "bfloat16"
) -> Any:
    torch = __import__("torch")
    transformers = __import__("transformers")
    peft = __import__("peft")
    safetensors_torch = __import__("safetensors.torch", fromlist=["load_file"])
    if not torch.cuda.is_available():
        raise ValenInferenceError("cuda_required_before_valen_model_load")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype_name
    ]
    model_cls = transformers.Qwen3_5ForConditionalGeneration
    container, loading_info = model_cls.from_pretrained(
        str(base_model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=dtype,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(
        loading_info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValenInferenceError(f"incomplete base model load: {loading_info}")
    backbone = container.model
    del container
    backbone.requires_grad_(False)
    targets = [
        name
        for name, module in backbone.language_model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    backbone.language_model = peft.get_peft_model(
        backbone.language_model,
        peft.LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules=targets, bias="none"),
    )
    tensors = safetensors_torch.load_file(str(weights_path), device="cpu")
    provided_lora = lora_tensor_keys(tuple(tensors))
    runtime = torch.nn.Module()
    runtime.backbone = backbone
    runtime.head = torch.nn.Module()
    runtime.head.candidate = torch.nn.Linear(2048, 256, bias=False)
    runtime.head.decision = torch.nn.Linear(2048, 256, bias=False)
    created_lora = lora_tensor_keys(tuple(runtime.state_dict()))
    assert_lora_key_set(provided_lora, created_lora)
    state = dict(tensors)
    missing, unexpected = runtime.load_state_dict(state, strict=False)
    if unexpected:
        raise ValenInferenceError(f"unexpected Valen delta keys: {unexpected[:5]}")
    runtime.requires_grad_(False)
    runtime.eval()
    runtime.to("cuda")
    moved_head = {
        "head.candidate.weight": runtime.head.candidate.weight,
        "head.decision.weight": runtime.head.decision.weight,
    }
    return ValenRuntime(runtime.backbone, ValenDecisionHead(moved_head), torch)


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValenInferenceError("cannot softmax empty logits")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValenInferenceError("cannot softmax non-finite logits")
    high = max(values)
    exps = [math.exp(value - high) for value in values]
    total = math.fsum(exps)
    return [value / total for value in exps]
