"""CPU-only S0 baseline compatibility probe.

This script records pinned model provenance and, when local metadata/tokenizer files
are present, probes official tokenizer/processor behavior without loading model
weights, importing remote model code, downloading files, or claiming model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, cast

from data.schema import Prefix, Question, answer_tokens
from scripts.assets import verify_file, write_json
from scripts.s0_prepare import read_manifest
from train.multimodal import file_hash, resolve_media
from train.template import render_payload

ANSWER_COUNTS = (2, 3, 4, 20)
CORE_MODEL_IDS = {
    "gemma4-e4b-it",
    "gemma4-e2b-it",
    "jev-omni-12b",
    "onejev-4b",
    "onejev-27b",
}
VALEN_ADAPTER_ID = "valen-preview-0923"
VALEN_BASE_ID = "qwen3.5-2b-for-valen"

TokenFactory = Callable[[Path], Any]
ProcessorFactory = Callable[[Path], Any]
AssetPathMap = dict[str, Path]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_asset_manifest(path: Path) -> dict[str, Any]:
    data = cast(dict[str, Any], json.loads(path.read_text()))
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), list):
        raise ValueError("unsupported project model asset manifest")
    ids = [model.get("id") for model in data["models"]]
    missing = sorted((CORE_MODEL_IDS | {VALEN_ADAPTER_ID, VALEN_BASE_ID}) - set(ids))
    if missing:
        raise ValueError(f"asset manifest missing required models: {missing}")
    return data


def model_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = manifest["models"]
    result = {str(model["id"]): model for model in models}
    if len(result) != len(models):
        raise ValueError("duplicate model IDs in asset manifest")
    return result


NORMAL_METADATA_NAMES = {
    ".gitattributes",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "NOTICE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "decision_config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "sha256.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "training_metadata.json",
    "verification.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


def _normal_metadata_path(path: str) -> bool:
    name = Path(path).name
    if path.endswith((".safetensors", ".pt", ".bin", ".pth", ".ckpt")):
        return False
    if path.startswith(("assets/", "card-assets/")):
        return False
    return name in NORMAL_METADATA_NAMES


def metadata_files(model: dict[str, Any]) -> list[dict[str, Any]]:
    files = model.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"model lacks file inventory: {model.get('id')}")
    selected = []
    for file in files:
        path = str(file.get("path", ""))
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"unsafe asset path for {model.get('id')}: {path}")
        if file.get("required") is True and _normal_metadata_path(path):
            selected.append(file)
    if not selected:
        raise ValueError(f"model lacks required normal metadata inventory: {model.get('id')}")
    return selected


def _verify_file_contract(file: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": file["path"],
        "size": file.get("bytes"),
        "kind": "metadata",
        "upstream_sha256": file.get("lfs_sha256"),
        "git_blob_sha1": file.get("git_blob_sha1"),
    }


def verify_required_metadata(directory: Path, model: dict[str, Any]) -> dict[str, str]:
    verified: dict[str, str] = {}
    for file in metadata_files(model):
        rel = Path(str(file["path"]))
        path = directory / rel
        verified[str(rel)] = verify_file(path, _verify_file_contract(file))
    return dict(sorted(verified.items()))


def manifest_metadata_digest(model: dict[str, Any]) -> str:
    keep = []
    for file in metadata_files(model):
        keep.append(
            {
                "path": file.get("path"),
                "bytes": file.get("bytes"),
                "lfs_sha256": file.get("lfs_sha256"),
                "git_blob_sha1": file.get("git_blob_sha1"),
            }
        )
    return canonical_json_sha256(sorted(keep, key=lambda item: str(item["path"])))


def model_dir(root: Path, model: dict[str, Any]) -> Path:
    repo_id = str(model["repo_id"])
    revision = str(model["revision"])
    if "/" not in repo_id or any(part in {"", ".", ".."} for part in repo_id.split("/")):
        raise ValueError("unsafe repo_id in manifest")
    return root / repo_id.replace("/", "__") / revision


def resolve_model_dir(
    root: Path | None, model: dict[str, Any], asset_paths: AssetPathMap | None = None
) -> Path | None:
    model_id = str(model["id"])
    if asset_paths and model_id in asset_paths:
        return asset_paths[model_id]
    if root is None:
        return None
    return model_dir(root, model)


def format_model_dir(directory: Path | None) -> str | None:
    return str(directory) if directory is not None else None


def existing_metadata_hashes(directory: Path, files: Iterable[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for file in files:
        rel = Path(str(file["path"]))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe metadata path")
        path = directory / rel
        if path.is_file():
            hashes[str(rel)] = file_hash(path)
    return dict(sorted(hashes.items()))


def build_subjects(
    manifest: dict[str, Any],
    *,
    model_root: Path | None = None,
    asset_paths: AssetPathMap | None = None,
) -> list[dict[str, Any]]:
    models = model_by_id(manifest)
    subjects: list[dict[str, Any]] = []
    for model_id in sorted(CORE_MODEL_IDS):
        model = models[model_id]
        subjects.append(
            {
                "id": model_id,
                "display_name": model.get("display_name"),
                "kind": "single_model",
                "repo_id": model.get("repo_id"),
                "revision": model.get("revision"),
                "license": model.get("license"),
                "role": model.get("role"),
                "manifest_metadata_sha256": manifest_metadata_digest(model),
                "local_model_dir": format_model_dir(
                    resolve_model_dir(model_root, model, asset_paths)
                ),
                "probe_model_id": model_id,
                "adapter": None,
                "base": None,
            }
        )
    adapter = models[VALEN_ADAPTER_ID]
    base = models[VALEN_BASE_ID]
    subjects.append(
        {
            "id": "valen-2b-qwen3.5-base",
            "display_name": "Valen-Preview-0923 / Valen-2B with frozen Qwen3.5-2B base",
            "kind": "adapter_plus_base_metadata_only",
            "repo_id": adapter.get("repo_id"),
            "revision": adapter.get("revision"),
            "license": adapter.get("license"),
            "role": adapter.get("role"),
            "manifest_metadata_sha256": canonical_json_sha256(
                {
                    "adapter": manifest_metadata_digest(adapter),
                    "base": manifest_metadata_digest(base),
                }
            ),
            "local_model_dir": format_model_dir(resolve_model_dir(model_root, base, asset_paths)),
            "probe_model_id": VALEN_BASE_ID,
            "adapter": {
                "repo_id": adapter.get("repo_id"),
                "revision": adapter.get("revision"),
                "license": adapter.get("license"),
                "manifest_metadata_sha256": manifest_metadata_digest(adapter),
                "local_model_dir": format_model_dir(
                    resolve_model_dir(model_root, adapter, asset_paths)
                ),
            },
            "base": {
                "repo_id": base.get("repo_id"),
                "revision": base.get("revision"),
                "license": base.get("license"),
                "manifest_metadata_sha256": manifest_metadata_digest(base),
            },
        }
    )
    return subjects


def default_tokenizer_factory(directory: Path) -> Any:
    transformers = importlib.import_module("transformers")
    return transformers.AutoTokenizer.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
    )


def default_processor_factory(directory: Path) -> Any:
    transformers = importlib.import_module("transformers")
    return transformers.AutoProcessor.from_pretrained(
        directory,
        local_files_only=True,
        trust_remote_code=False,
    )


def tokenizer_class_name(tokenizer: Any) -> str:
    return f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"


def template_source(tokenizer: Any, directory: Path) -> str | None:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and template:
        return template
    path = directory / "chat_template.jinja"
    if path.is_file():
        return path.read_text()
    config = directory / "tokenizer_config.json"
    if config.is_file():
        value = json.loads(config.read_text()).get("chat_template")
        if isinstance(value, str) and value:
            return value
    return None


def encode_literal(tokenizer: Any, token: str) -> list[int]:
    return list(tokenizer.encode(token, add_special_tokens=False))


def decode_literal(tokenizer: Any, token_id: int) -> str | None:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return None
    return str(decode([token_id], skip_special_tokens=False))


def token_is_unknown_or_special(tokenizer: Any, token_id: int) -> bool:
    if token_id == getattr(tokenizer, "unk_token_id", None):
        return True
    special_ids = getattr(tokenizer, "all_special_ids", None)
    return isinstance(special_ids, list) and token_id in special_ids


def _question_for_probe(kind: str, count: int | None = None) -> Question:
    if kind == "choice":
        assert count is not None
        return Question(
            qid=f"choice-{count}",
            type="choice",
            instruction=f"Choose one of {count} options.",
            options=[f"option {index}" for index in range(count)],
        )
    if kind == "score":
        assert count is not None
        return Question(
            qid=f"score-{count}",
            type="score",
            instruction=f"Score from 0 to {count - 1}.",
            score_levels=count,
        )
    return Question(qid="noul", type="noul", instruction="Answer yes or no.")


def _probe_question_tokens(tokenizer: Any, question: Question) -> dict[str, Any]:
    mapping: dict[str, int] = {}
    failures: list[str] = []
    for token in (*answer_tokens(question), "U"):
        ids = encode_literal(tokenizer, token)
        if len(ids) != 1:
            failures.append(f"{token}: token_count={len(ids)}")
            continue
        token_id = ids[0]
        if token_is_unknown_or_special(tokenizer, token_id):
            failures.append(f"{token}: unknown_or_special")
        decoded = decode_literal(tokenizer, token_id)
        if decoded is not None and decoded != token:
            failures.append(f"{token}: decode_roundtrip={decoded!r}")
        if token_id in mapping.values():
            failures.append(f"{token}: duplicate_token_id")
        mapping[token] = token_id
    return {
        "status": "passed" if not failures else "failed",
        "answer_count_including_u": len(answer_tokens(question)) + 1,
        "token_ids": mapping,
        "token_ids_sha256": canonical_json_sha256(mapping),
        "failures": failures,
    }


def probe_answer_mapping(tokenizer: Any) -> dict[str, Any]:
    probes: dict[str, Any] = {"choice": {}, "score": {}, "noul": {}}
    aggregate_failures: list[str] = []
    for count in ANSWER_COUNTS:
        item = _probe_question_tokens(tokenizer, _question_for_probe("choice", count))
        probes["choice"][str(count)] = item
        aggregate_failures.extend(f"choice_{count}:{failure}" for failure in item["failures"])
    for count in (2, 3, 10):
        item = _probe_question_tokens(tokenizer, _question_for_probe("score", count))
        probes["score"][str(count)] = item
        aggregate_failures.extend(f"score_{count}:{failure}" for failure in item["failures"])
    item = _probe_question_tokens(tokenizer, _question_for_probe("noul"))
    probes["noul"] = item
    aggregate_failures.extend(f"noul:{failure}" for failure in item["failures"])
    return {
        "status": "passed" if not aggregate_failures else "failed",
        "probes": probes,
        "failures": aggregate_failures,
    }


def _template_has_enable_thinking_disabled_branch(template: str | None) -> bool:
    if not template:
        return False
    compact = " ".join(template.split())
    disabled_marker_in_template = (
        "<think>\\n\\n</think>" in template or "<think>\n\n</think>" in template
    )
    explicit_false_branch = (
        "enable_thinking is defined" in compact and "enable_thinking is false" in compact
    )
    explicit_true_else_branch = (
        "enable_thinking is defined" in compact
        and "enable_thinking is true" in compact
        and "{%- else %}" in template
        and disabled_marker_in_template
    )
    return explicit_false_branch or explicit_true_else_branch


def _disabled_thinking_marker(off: str, on: str) -> str | None:
    for marker in ("<think>\n\n</think>", "<think>\\n\\n</think>"):
        if marker in off and marker not in on:
            return marker
    return None


def probe_thinking(tokenizer: Any, template: str | None = None) -> dict[str, Any]:
    messages = [{"role": "user", "content": "Reply with exactly one decision token."}]
    try:
        off = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        on = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception as exc:  # noqa: BLE001 - third-party template errors are reported by type only.
        return {"status": "not_proved", "reason": type(exc).__name__}
    if not isinstance(off, str) or not isinstance(on, str) or not off:
        return {"status": "not_proved", "reason": "non_string_or_empty_template"}
    config = getattr(tokenizer, "init_kwargs", {})
    think_token = config.get("think_token") if isinstance(config, dict) else None
    candidates = [
        token for token in (think_token, "<|think|>", "<think>") if isinstance(token, str)
    ]
    matched = [token for token in candidates if token and token not in off and token in on]
    disabled_marker = _disabled_thinking_marker(off, on)
    template_disabled_branch = _template_has_enable_thinking_disabled_branch(template)
    proved_by_template = off != on and template_disabled_branch and disabled_marker is not None
    status = "proved_disabled" if matched or proved_by_template else "not_proved"
    return {
        "status": status,
        "off_prompt_sha256": sha256_bytes(off.encode()),
        "on_prompt_sha256": sha256_bytes(on.encode()),
        "evidence_token_sha256": sha256_bytes(matched[0].encode()) if matched else None,
        "evidence_kind": (
            "token_absent_when_disabled"
            if matched
            else "template_enable_thinking_disabled_branch"
            if proved_by_template
            else None
        ),
        "disabled_marker_sha256": (
            sha256_bytes(disabled_marker.encode())
            if proved_by_template and disabled_marker
            else None
        ),
    }


def tensor_summary(value: Any) -> dict[str, Any]:
    if not hasattr(value, "detach"):
        raise ValueError("processor returned non-tensor output")
    array = value.detach().cpu().numpy()
    return {
        "shape": list(array.shape),
        "dtype": str(value.dtype),
        "sha256": sha256_bytes(array.tobytes()),
    }


def probe_text_case(tokenizer: Any, question: Question, prefix: Prefix) -> dict[str, Any]:
    payload = render_payload(prefix, question)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": payload}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("official text template returned empty prompt")
    return {
        "status": "passed",
        "payload_sha256": sha256_bytes(payload.encode()),
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "prompt_tokens": len(encode_literal(tokenizer, prompt)),
    }


def probe_image_case(processor: Any, case: Any, asset_root: Path) -> dict[str, Any]:
    if not case.prefix.media:
        raise ValueError("image case has no media")
    first = case.prefix.media[0]
    if first.type != "image":
        raise ValueError("public image probe only accepts image media")
    image_module = importlib.import_module("PIL.Image")
    image_path = resolve_media(asset_root, first.uri, first.sha256, max_bytes=16 * 1024 * 1024)
    with image_module.open(image_path) as source:
        width = int(getattr(source, "width", 0))
        height = int(getattr(source, "height", 0))
        if width <= 0 or height <= 0 or width * height > 40_000_000:
            raise ValueError("image exceeds pixel limit")
        source.load()
        image = source.convert("RGB")
    payload = render_payload(Prefix(state=case.prefix.state), case.question)
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": payload}]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("official image template returned empty prompt")
    inputs = processor(text=[prompt], images=[[image]], return_tensors="pt", truncation=False)
    summaries = {key: tensor_summary(value) for key, value in dict(inputs).items()}
    if "input_ids" not in summaries:
        raise ValueError("processor omitted input_ids")
    return {
        "status": "passed",
        "category": case.category,
        "case_id_sha256": sha256_bytes(case.id.encode()),
        "payload_sha256": sha256_bytes(payload.encode()),
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "media_sha256": [first.sha256],
        "tensors": summaries,
    }


def select_public_cases(manifest_path: Path | None, *, allow_partial: bool) -> dict[str, Any]:
    if manifest_path is None:
        return {"status": "not_requested"}
    cases = read_manifest(manifest_path, allow_partial=allow_partial)
    text = next((case for case in cases if case.category == "text"), None)
    image = next((case for case in cases if case.category in {"image", "screenshot"}), None)
    return {
        "status": "selected",
        "manifest_sha256": file_hash(manifest_path),
        "case_count": len(cases),
        "text": text,
        "image": image,
    }


def sanitize_blocker(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__}


def probe_subject(
    subject: dict[str, Any],
    models: dict[str, dict[str, Any]],
    *,
    model_root: Path,
    asset_paths: AssetPathMap | None,
    public_selection: dict[str, Any],
    public_asset_root: Path | None,
    tokenizer_factory: TokenFactory = default_tokenizer_factory,
    processor_factory: ProcessorFactory = default_processor_factory,
) -> dict[str, Any]:
    probe_model = models[str(subject["probe_model_id"])]
    directory = resolve_model_dir(model_root, probe_model, asset_paths)
    if directory is None:
        raise ValueError("model_root or explicit asset path is required")
    result = dict(subject)
    result.update(
        {
            "status": "blocked",
            "gpu_used": False,
            "weights_loaded": False,
            "quality_admitted": False,
            "published_teacher_readout_verified": False,
            "local_model_dir": str(directory),
            "local_metadata_sha256": None,
            "tokenizer": {"status": "not_probed"},
            "processor": {"status": "not_probed"},
            "public_inputs": {"status": public_selection.get("status", "not_requested")},
        }
    )
    if not directory.is_dir():
        result["blocker"] = "local_model_directory_missing"
        return result
    try:
        hashes = verify_required_metadata(directory, probe_model)
    except Exception as exc:  # noqa: BLE001
        result["blocker"] = "local_metadata_invalid"
        result["metadata_error"] = sanitize_blocker(exc)
        return result
    result["local_metadata_sha256"] = canonical_json_sha256(hashes)
    if subject.get("kind") == "adapter_plus_base_metadata_only":
        adapter = models[VALEN_ADAPTER_ID]
        adapter_directory = resolve_model_dir(model_root, adapter, asset_paths)
        if adapter_directory is None:
            result["blocker"] = "adapter_metadata_invalid"
            result["adapter_metadata_error"] = {"type": "ValueError"}
            return result
        try:
            adapter_hashes = verify_required_metadata(adapter_directory, adapter)
        except Exception as exc:  # noqa: BLE001
            result["blocker"] = "adapter_metadata_invalid"
            result["adapter_metadata_error"] = sanitize_blocker(exc)
            return result
        if "config.json" not in adapter_hashes:
            result["blocker"] = "adapter_config_missing"
            return result
        result["adapter_local_metadata_sha256"] = canonical_json_sha256(adapter_hashes)
        result["adapter_config_sha256"] = adapter_hashes["config.json"]
    try:
        tokenizer = tokenizer_factory(directory)
        template = template_source(tokenizer, directory)
        answer_readout = probe_answer_mapping(tokenizer)
        thinking = probe_thinking(tokenizer, template)
        token_report: dict[str, Any] = {
            "status": "passed",
            "class": tokenizer_class_name(tokenizer),
            "template_sha256": sha256_bytes(template.encode()) if template else None,
            "template_status": "present" if template else "missing",
            "answer_readout": answer_readout,
            "thinking": thinking,
        }
        if answer_readout["status"] != "passed":
            token_report["status"] = "failed"
        elif not template or thinking["status"] != "proved_disabled":
            token_report["status"] = "partial"
        result["tokenizer"] = token_report
    except Exception as exc:  # noqa: BLE001
        result["tokenizer"] = {"status": "blocked", "blocker": sanitize_blocker(exc)}
        result["blocker"] = "tokenizer_probe_blocked"
        return result
    try:
        processor = processor_factory(directory)
        result["processor"] = {
            "status": "loaded",
            "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        }
    except Exception as exc:  # noqa: BLE001
        result["processor"] = {"status": "blocked", "blocker": sanitize_blocker(exc)}
        result["status"] = (
            "partial" if result["tokenizer"]["status"] in {"passed", "partial"} else "blocked"
        )
        result["blocker"] = "processor_probe_blocked"
        return result
    public_report: dict[str, Any] = {"status": public_selection.get("status", "not_requested")}
    if public_selection.get("status") == "selected":
        public_report.update(
            {
                "manifest_sha256": public_selection["manifest_sha256"],
                "case_count": public_selection["case_count"],
                "text": {"status": "not_probed"},
                "image": {"status": "not_probed"},
            }
        )
        text_case = public_selection.get("text")
        image_case = public_selection.get("image")
        if text_case is not None:
            try:
                public_report["text"] = probe_text_case(
                    tokenizer, text_case.question, text_case.prefix
                )
            except Exception as exc:  # noqa: BLE001
                public_report["text"] = {"status": "blocked", "blocker": sanitize_blocker(exc)}
        if image_case is not None and public_asset_root is not None:
            try:
                public_report["image"] = probe_image_case(processor, image_case, public_asset_root)
            except Exception as exc:  # noqa: BLE001
                public_report["image"] = {"status": "blocked", "blocker": sanitize_blocker(exc)}
    result["public_inputs"] = public_report
    public_blocked = False
    if public_selection.get("status") == "selected":
        requested = [public_report.get("text"), public_report.get("image")]
        public_blocked = any(
            isinstance(item, dict) and item.get("status") != "passed" for item in requested
        )
    elif public_selection.get("status") == "not_requested":
        public_blocked = True
        result["public_inputs"] = {
            "status": "not_requested",
            "reason": "public text/image probes are required for run acceptance",
        }
    token_status = result["tokenizer"]["status"]
    if token_status == "failed":
        result["status"] = "failed"
    elif (
        token_status == "passed"
        and result["processor"]["status"] == "loaded"
        and not public_blocked
    ):
        result["status"] = "passed"
    else:
        result["status"] = "partial"
        if public_blocked:
            result["blocker"] = "requested_public_probe_incomplete"
    return result


def validate_asset_paths(manifest: dict[str, Any], asset_paths: AssetPathMap | None) -> None:
    if not asset_paths:
        return
    known = set(model_by_id(manifest))
    unknown = sorted(set(asset_paths) - known)
    if unknown:
        raise ValueError(f"asset path override uses unknown model IDs: {unknown}")


def validate_report(
    asset_manifest: Path,
    *,
    model_root: Path | None = None,
    asset_paths: AssetPathMap | None = None,
) -> dict[str, Any]:
    manifest = load_asset_manifest(asset_manifest)
    validate_asset_paths(manifest, asset_paths)
    subjects = build_subjects(manifest, model_root=model_root, asset_paths=asset_paths)
    return {
        "schema": "lite-omnijev.s0-baseline-cpu.v1",
        "mode": "validate",
        "status": "passed",
        "exit_code": 0,
        "asset_manifest": str(asset_manifest),
        "asset_manifest_sha256": file_hash(asset_manifest),
        "asset_paths": {key: str(value) for key, value in sorted((asset_paths or {}).items())},
        "code_revision": git_revision(),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "cost": None,
        "gpu_used": False,
        "weights_loaded": False,
        "quality_admitted": False,
        "subject_count": len(subjects),
        "subjects": subjects,
    }


def run_report(
    asset_manifest: Path,
    model_root: Path,
    *,
    asset_paths: AssetPathMap | None = None,
    public_manifest: Path | None = None,
    public_asset_root: Path | None = None,
    allow_partial_public_manifest: bool = False,
    tokenizer_factory: TokenFactory = default_tokenizer_factory,
    processor_factory: ProcessorFactory = default_processor_factory,
) -> dict[str, Any]:
    started = time.monotonic()
    manifest = load_asset_manifest(asset_manifest)
    validate_asset_paths(manifest, asset_paths)
    models = model_by_id(manifest)
    public_selection = select_public_cases(
        public_manifest, allow_partial=allow_partial_public_manifest
    )
    subjects = []
    for subject in build_subjects(manifest, model_root=model_root, asset_paths=asset_paths):
        subjects.append(
            probe_subject(
                subject,
                models,
                model_root=model_root,
                asset_paths=asset_paths,
                public_selection=public_selection,
                public_asset_root=public_asset_root,
                tokenizer_factory=tokenizer_factory,
                processor_factory=processor_factory,
            )
        )
    passed = sum(1 for subject in subjects if subject["status"] == "passed")
    partial = sum(1 for subject in subjects if subject["status"] == "partial")
    blocked = sum(1 for subject in subjects if subject["status"] == "blocked")
    failed = sum(1 for subject in subjects if subject["status"] == "failed")
    status: Literal["passed", "partial", "failed"]
    if failed:
        status = "failed"
    elif passed == len(subjects):
        status = "passed"
    else:
        status = "partial"
    return {
        "schema": "lite-omnijev.s0-baseline-cpu.v1",
        "mode": "run",
        "status": status,
        "exit_code": 0 if status == "passed" else 1,
        "asset_manifest": str(asset_manifest),
        "asset_manifest_sha256": file_hash(asset_manifest),
        "model_root": str(model_root),
        "asset_paths": {key: str(value) for key, value in sorted((asset_paths or {}).items())},
        "code_revision": git_revision(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
        },
        "cost": None,
        "gpu_used": False,
        "weights_loaded": False,
        "quality_admitted": False,
        "published_teacher_readout_verified": False,
        "summary": {"passed": passed, "partial": partial, "blocked": blocked, "failed": failed},
        "subjects": subjects,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_asset_paths(values: list[str] | None) -> AssetPathMap:
    result: AssetPathMap = {}
    for value in values or []:
        model_id, sep, raw_path = value.partition("=")
        if not sep or not model_id or not raw_path:
            raise ValueError("--asset-path must use MODEL_ID=PATH")
        if model_id in result:
            raise ValueError(f"duplicate --asset-path for {model_id}")
        result[model_id] = Path(raw_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate pinned manifest and subject expansion")
    validate.add_argument(
        "--asset-manifest", type=Path, default=Path("configs/project-model-assets.json")
    )
    validate.add_argument("--model-root", type=Path)
    validate.add_argument(
        "--asset-path",
        action="append",
        help="explicit model asset path override as MODEL_ID=PATH; still hash-checked",
    )
    validate.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="probe local official tokenizer/processor metadata")
    run.add_argument(
        "--asset-manifest", type=Path, default=Path("configs/project-model-assets.json")
    )
    run.add_argument("--model-root", type=Path, required=True)
    run.add_argument(
        "--asset-path",
        action="append",
        help="explicit model asset path override as MODEL_ID=PATH; still hash-checked",
    )
    run.add_argument("--public-manifest", type=Path)
    run.add_argument("--public-asset-root", type=Path)
    run.add_argument("--allow-partial-public-manifest", action="store_true")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output path already exists; choose a new report path")
    report: dict[str, Any] = {
        "schema": "lite-omnijev.s0-baseline-cpu.v1",
        "mode": args.command,
        "status": "failed",
        "exit_code": 1,
        "gpu_used": False,
        "weights_loaded": False,
        "quality_admitted": False,
        "cost": None,
    }
    write_json(args.output, report)
    try:
        asset_paths = parse_asset_paths(args.asset_path)
        if args.command == "validate":
            report = validate_report(
                args.asset_manifest, model_root=args.model_root, asset_paths=asset_paths
            )
        else:
            report = run_report(
                args.asset_manifest,
                args.model_root,
                asset_paths=asset_paths,
                public_manifest=args.public_manifest,
                public_asset_root=args.public_asset_root,
                allow_partial_public_manifest=args.allow_partial_public_manifest,
            )
    except Exception as exc:  # noqa: BLE001
        report.update(status="failed", error=sanitize_blocker(exc))
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "exit_code")}, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
