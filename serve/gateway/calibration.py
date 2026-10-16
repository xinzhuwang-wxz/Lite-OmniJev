"""Explicit local-bootstrap temperature registry; never private admission or quality acceptance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import Field

from data.schema import Media, Question, answer_tokens
from scripts.assets import selected_files
from scripts.check_tokens import probe
from train.calibration import (
    Binding,
    Cohort,
    Primitive,
    Sha256,
    StrictModel,
    TemperatureArtifact,
    apply_temperature,
    load_temperature,
)
from train.checkpoint import file_sha256
from train.readout import Readout, read_probabilities
from train.trainer import verified_assets, verified_init_adapter

REACHABLE_COHORTS: tuple[Cohort, ...] = ("text", "image", "audio", "video", "crossmodal")
PRIMITIVES: tuple[Primitive, ...] = ("choice", "score", "noul")
ALL_COHORTS: tuple[Cohort, ...] = (*REACHABLE_COHORTS, "screenshot")
REQUIRED_GROUPS = frozenset((c, p) for c in ALL_COHORTS for p in PRIMITIVES)


class RegistryEntry(StrictModel):
    path: str
    sha256: Sha256
    binding: Binding


class Group(StrictModel):
    cohort: Cohort
    primitive: Primitive


class RuntimeProvenance(StrictModel):
    path: str
    sha256: Sha256


class RegistryManifest(StrictModel):
    schema_version: Literal[1, 2]
    model_version: Annotated[str, Field(min_length=1, max_length=128)]
    calibration_version: Annotated[str, Field(min_length=1, max_length=128)]
    groups: Annotated[list[RegistryEntry], Field(min_length=1)]
    fitted: list[Group] | None = None
    unfitted: list[Group] | None = None
    runtime_provenance: list[RuntimeProvenance] = []


def infer_cohort(media: list[Media]) -> Cohort:
    kinds = {item.type for item in media}
    if not kinds:
        return "text"
    if len(kinds) > 1:
        return "crossmodal"
    return next(iter(kinds))


def _cmdline(pid: int) -> tuple[str, ...]:
    return tuple(
        part.decode()
        for part in (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
        if part
    )


def _flag(argv: list[str], name: str) -> str:
    if argv.count(name) != 1:
        raise ValueError("unambiguous local serving flag required")
    index = argv.index(name)
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ValueError("missing local serving flag value")
    return argv[index + 1]


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class CalibrationRegistry:
    model_version: str
    version: str
    sha256: str
    model_dir: Path
    upstream_url: str
    upstream_model: str
    pid: int
    process_command: tuple[str, ...]
    token_map: Mapping[str, int]
    groups: Mapping[tuple[Cohort, Primitive], tuple[TemperatureArtifact, str]]
    file_stamps: Mapping[Path, tuple[int, int, int, int]]

    def ensure_current(self) -> None:
        if _cmdline(self.pid) != self.process_command:
            raise ValueError("local serving process identity changed")
        if any(_stamp(path) != stamp for path, stamp in self.file_stamps.items()):
            raise ValueError("bound calibration/deployment files changed")

    def verify_runtime(self, tokenizer: Any, template: Any) -> None:
        self.ensure_current()
        expected = next(iter(self.groups.values()))[0].binding.official_template_sha256
        if (
            not isinstance(template, str)
            or hashlib.sha256(template.encode()).hexdigest() != expected
        ):
            raise ValueError("gateway rendering template differs from calibration")
        if tokenizer is None:
            raise ValueError("gateway tokenizer unavailable")
        for token, identifier in self.token_map.items():
            if tokenizer.encode(token, add_special_tokens=False) != [identifier]:
                raise ValueError("gateway tokenizer differs from calibration")

    def read(
        self,
        media: list[Media],
        question: Question,
        logits: dict[str, float],
        token_ids: dict[str, int],
        *,
        require_calibrated: bool = False,
    ) -> tuple[Readout, dict[str, str]]:
        self.ensure_current()
        group = (infer_cohort(media), question.type)
        expected_tokens = (*answer_tokens(question), "U")
        if token_ids != {token: self.token_map[token] for token in expected_tokens}:
            raise ValueError("runtime answer token mapping differs from deployment")
        if group not in self.groups:
            if require_calibrated:
                raise ValueError("requested calibration group is explicitly unfitted")
            return read_probabilities(logits, answer_tokens(question)), {
                "calibration": "unfitted",
                "calibration_version": self.version,
                "calibration_registry_sha256": self.sha256,
                "calibration_cohort": group[0],
            }
        artifact, digest = self.groups[group]
        result = apply_temperature(
            artifact, artifact.binding, logits, list(answer_tokens(question))
        )
        return result, {
            "calibration": "scalar_temperature",
            "calibration_version": self.version,
            "calibration_registry_sha256": self.sha256,
            "calibration_artifact_sha256": digest,
            "calibration_cohort": group[0],
        }


def load_registry(
    path: Path,
    *,
    serving_receipt: Path,
    asset_manifest: Path,
    model_dir: Path,
    upstream_url: str,
    upstream_model: str,
) -> CalibrationRegistry:
    """Use the existing native bootstrap argv/PID receipt plus live local /proc and actual files.

    Caller owns the local bootstrap and pinned manifest. Root and arbitrary remote servers
    are outside this ordinary local trust boundary. Historical/merged/quantized recipes reject.
    """
    initial_stamps = {p: _stamp(p) for p in (path, serving_receipt, asset_manifest)}
    manifest = RegistryManifest.model_validate_json(path.read_bytes())
    server: Any = json.loads(serving_receipt.read_bytes())
    if server.get("ready") is not True or "server_exit_code" in server:
        raise ValueError("live local bootstrap receipt required")
    argv, pid = server.get("argv"), server.get("server_pid")
    if (
        not isinstance(argv, list)
        or not all(isinstance(value, str) for value in argv)
        or type(pid) is not int
        or pid <= 0
        or argv.count("serve") != 1
    ):
        raise ValueError("local native serving argv/PID required")
    process = _cmdline(pid)
    if process != tuple(argv) and process[1:] != tuple(argv):
        raise ValueError("receipt differs from actual local process")
    serve_index = argv.index("serve")
    if serve_index != 1 or serve_index + 1 >= len(argv) or Path(argv[0]).name != "vllm":
        raise ValueError("standard native vLLM bootstrap required")
    if not Path(argv[serve_index + 1]).is_absolute():
        raise ValueError("absolute observed model path required")
    actual_model = Path(argv[serve_index + 1]).resolve()
    if actual_model != model_dir.resolve():
        raise ValueError("gateway model differs from serving process")
    if any(flag in argv for flag in ("--chat-template", "--hf-config-path", "--quantization")):
        raise ValueError("overridden/derived serving recipe requires separate binding")
    if "--tokenizer" in argv and Path(_flag(argv, "--tokenizer")).resolve() != actual_model:
        raise ValueError("serving tokenizer differs from official model")
    url = urlsplit(upstream_url)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost")
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
        or _flag(argv, "--host") not in ("127.0.0.1", "localhost")
        or url.port != int(_flag(argv, "--port"))
        or _flag(argv, "--logprobs-mode") != "raw_logprobs"
        or "--enable-lora" not in argv
    ):
        raise ValueError("local endpoint/raw-logprobs bootstrap mismatch")
    modules_index = argv.index("--lora-modules") if "--lora-modules" in argv else -1
    if modules_index < 0 or argv.count("--lora-modules") != 1:
        raise ValueError("explicit native decision adapter required")
    modules = []
    for value in argv[modules_index + 1 :]:
        if value.startswith("--"):
            break
        if value.count("=") != 1:
            raise ValueError("standard alias=path LoRA mapping required")
        name, directory = value.split("=", 1)
        if name == upstream_model:
            if not Path(directory).is_absolute():
                raise ValueError("absolute observed adapter path required")
            modules.append(Path(directory).resolve())
    if len(modules) != 1:
        raise ValueError("requested model is not one observed native adapter")
    asset_document = json.loads(asset_manifest.read_bytes())
    matched = [
        m for m in asset_document.get("models", []) if m.get("revision") == actual_model.name
    ]
    selected = selected_files({**asset_document, "models": matched}, metadata_only=False)
    watched = [actual_model / entry["path"] for _, entry in selected]
    initial_stamps.update({p: _stamp(p) for p in watched})
    verified_assets(actual_model, asset_manifest)
    tokenizer = probe(actual_model)
    if tokenizer["status"] != "passed":
        raise ValueError("official tokenizer/template probe failed")
    token_map = tokenizer["answer_token_ids"]
    mapping_sha = hashlib.sha256(json.dumps(token_map, sort_keys=True).encode()).hexdigest()
    adapter = modules[0]
    initial_stamps.update(
        {
            p: _stamp(p)
            for p in [
                adapter / "adapter_model.safetensors",
                adapter / "adapter_config.json",
            ]
        }
    )
    weights_sha = file_sha256(adapter / "adapter_model.safetensors")
    verified_init_adapter(adapter, weights_sha)
    groups = {}
    for entry in manifest.groups:
        binding = entry.binding
        if (
            binding.base_revision != actual_model.name
            or binding.adapter_weights_sha256 != weights_sha
            or binding.official_template_sha256 != tokenizer["template_sha256"]
            or binding.answer_token_map_sha256 != mapping_sha
        ):
            raise ValueError("calibration does not match actual deployment files")
        group = binding.cohort, binding.primitive
        if group in groups:
            raise ValueError("duplicate calibration group")
        relative = Path(entry.path)
        artifact_path = (path.parent / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not artifact_path.is_relative_to(path.parent.resolve())
        ):
            raise ValueError("calibration artifact escapes registry root")
        initial_stamps[artifact_path] = _stamp(artifact_path)
        if file_sha256(artifact_path) != entry.sha256:
            raise ValueError("calibration artifact checksum mismatch")
        groups[group] = load_temperature(artifact_path, binding), entry.sha256
    if manifest.schema_version == 1:
        if manifest.fitted is not None or manifest.unfitted is not None:
            raise ValueError("legacy registry cannot declare partial partition")
        if set(groups) != REQUIRED_GROUPS:
            raise ValueError("missing configured modality/primitive calibration groups")
    else:
        if manifest.fitted is None or manifest.unfitted is None:
            raise ValueError("explicit fitted/unfitted partition required")
        fitted = [(g.cohort, g.primitive) for g in manifest.fitted]
        unfitted = [(g.cohort, g.primitive) for g in manifest.unfitted]
        if (
            len(set(fitted)) != len(fitted)
            or len(set(unfitted)) != len(unfitted)
            or set(fitted) & set(unfitted)
            or set(fitted) | set(unfitted) != REQUIRED_GROUPS
            or set(fitted) != set(groups)
        ):
            raise ValueError("fitted/unfitted partition differs from exact eighteen groups")
    # Artifact Binding predates runtime identity. Bind its raw receipt bytes instead;
    # never infer engine/precision equivalence from matching weights or token IDs.
    proven = set()
    for provenance in manifest.runtime_provenance:
        relative = Path(provenance.path)
        raw_path = (path.parent / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not raw_path.is_relative_to(path.parent.resolve())
        ):
            raise ValueError("runtime provenance escapes registry root")
        initial_stamps[raw_path] = _stamp(raw_path)
        if file_sha256(raw_path) != provenance.sha256:
            raise ValueError("runtime provenance checksum mismatch")
        recipe = json.loads(raw_path.read_bytes()).get("runtime_recipe")
        dtype = _flag(argv, "--dtype") if "--dtype" in argv else None
        if recipe != {
            "engine": "vllm",
            "dtype": dtype,
            "adapter_mode": "native_lora",
        } or dtype not in ("bfloat16", "float32", "float16"):
            raise ValueError(
                "calibration runtime recipe is absent or differs from observed "
                "serving engine/precision/adapter mode"
            )
        if provenance.sha256 in proven:
            raise ValueError("duplicate runtime provenance")
        proven.add(provenance.sha256)
    if proven != {artifact.binding.raw_logits_receipt_sha256 for artifact, _ in groups.values()}:
        raise ValueError(
            "hash-bound calibration runtime provenance required for every fitted group"
        )
    registry = CalibrationRegistry(
        manifest.model_version,
        manifest.calibration_version,
        file_sha256(path),
        actual_model,
        upstream_url.rstrip("/"),
        upstream_model,
        pid,
        process,
        MappingProxyType(token_map),
        MappingProxyType(groups),
        MappingProxyType(initial_stamps),
    )
    registry.ensure_current()
    return registry
