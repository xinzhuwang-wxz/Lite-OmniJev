"""Fixed trusted private model job. Root access is not external isolation proof.

The supervisor exports aggregates only; worker streams and per-item records stay private.
No arbitrary command, Python, environment, or backend selector is accepted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import random
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Never

from pydantic import Field, model_validator

from data.schema import Contract, HardTarget, Sha256
from scripts import s0_development_baselines as baseline
from scripts.s0_public_baselines import configure_torch_runtime, sha256_file, sha256_json
from train.multimodal import configure_image_processor

FROZEN_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
MAX_RECEIPT = 65536
CODE_FILES = (
    *baseline.REQUIRED_CODE_FILES,
    "scripts/private_stage_eval.py",
    "train/trainer.py",
    "scripts/s1_dev_eval.py",
)


class Rejected(ValueError):
    """Only static categories may cross the supervisor boundary."""

    def __init__(self, category: str, exit_code: int | None = None) -> None:
        super().__init__(category)
        self.exit_code = exit_code


class Artifact(Contract):
    path: str
    sha256: Sha256


class Admission(Contract):
    schema_version: Literal["private-stage-data-admission-v1"]
    status: Literal["admitted"]
    data_sha256: Sha256
    selected_manifest_sha256: Sha256
    evidence_bundle_sha256: Sha256
    pool: Literal["calibration", "holdout"]
    cohort: str
    expected_rows: int = Field(gt=0)


class Spec(Contract):
    mode: Literal["engineering", "S4", "S5", "S6"]
    cohort: Literal["text", "image", "screenshot", "audio", "video", "crossmodal"]
    data: Artifact
    pool: Literal["dev", "calibration", "holdout"]
    media_root: str
    model_dir: str
    model_id: Literal["gemma4-e4b-it"] = "gemma4-e4b-it"
    base_revision: Literal["ee0ef6023621cff504d758262d4e04895a5af4a2"]
    model_files: dict[str, Sha256] = Field(min_length=1)
    adapter_dir: str | None = None
    adapter_files: dict[str, Sha256] | None = None
    admission: Artifact | None = None
    expected_rows: int = Field(gt=0)
    subset_limit: int | None = Field(default=None, ge=1, le=50)
    seed: Literal[42] = 42
    image_budget: Literal[1120] = 1120
    video_frames: Literal[16] = 16
    output_root: str
    timeout_seconds: int = Field(ge=1, le=86400)

    @model_validator(mode="after")
    def scope(self) -> Spec:
        if self.mode == "engineering":
            if self.pool != "dev" or self.admission is not None:
                raise ValueError("engineering_scope")
        elif self.pool == "dev" or self.subset_limit is not None or self.admission is None:
            raise ValueError("stage_scope")
        if (self.adapter_dir is None) != (self.adapter_files is None):
            raise ValueError("adapter_binding")
        return self


class Metrics(Contract):
    count: int = Field(gt=0)
    accuracy: float = Field(ge=0, le=1)
    ece15: float = Field(ge=0, le=1)
    brier: float = Field(ge=0)
    coverage_at_error: float = Field(ge=0, le=1)
    max_error: float = Field(ge=0, le=1)


class Runtime(Contract):
    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    transformers: str = Field(min_length=1)
    device: str = Field(min_length=1)


class FailureReceipt(Contract):
    schema_version: Literal["private-stage-eval-failure-v1"] = "private-stage-eval-failure-v1"
    execution_status: Literal["rejected"] = "rejected"
    quality_status: Literal["not_evaluated"] = "not_evaluated"
    category: Literal[
        "input_rejected",
        "binding_mismatch",
        "model_binding_incomplete",
        "model_rejected",
        "admission_mismatch",
        "row_count_mismatch",
        "cohort_mismatch",
        "sample_rejected",
        "engineering_limit_required",
        "private_root_required",
        "timeout",
        "interrupted",
        "backend_failed",
        "invalid_receipt",
        "job_rejected",
    ]
    spec_sha256: Sha256 | None
    mode: Literal["engineering", "S4", "S5", "S6"] | None
    wall_seconds: float = Field(ge=0)
    exit_code: int | None
    cost_usd: None = None
    cost_boundary: Literal["not_measured_no_price_assumption"] = "not_measured_no_price_assumption"


class Receipt(Contract):
    schema_version: Literal["private-stage-eval-v1"] = "private-stage-eval-v1"
    mode: Literal["engineering", "S4", "S5", "S6"]
    admission_status: Literal["not_required_engineering", "admitted"]
    execution_status: Literal["complete"] = "complete"
    quality_status: Literal["not_evaluated"] = "not_evaluated"
    boundary: Literal["trusted_private_subprocess_root_visible"] = (
        "trusted_private_subprocess_root_visible"
    )
    spec_sha256: Sha256
    data_sha256: Sha256
    code_sha256: Sha256
    base_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_manifest_sha256: Sha256
    adapter_manifest_sha256: Sha256 | None
    tokenizer_template_sha256: Sha256
    answer_map_sha256: Sha256
    config_sha256: Sha256
    admission_sha256: Sha256 | None
    records_sha256: Sha256
    seed: Literal[42] = 42
    source_rows: int = Field(gt=0)
    records: int = Field(gt=0)
    metrics: Metrics
    wall_seconds: float = Field(ge=0)
    peak_memory_bytes: int = Field(ge=0)
    runtime: Runtime
    unique_families: int = Field(gt=0)
    dependency_claim: Literal["no_iid_claim"] = "no_iid_claim"
    cost_usd: None = None
    cost_boundary: Literal["not_measured_no_price_assumption"] = "not_measured_no_price_assumption"
    exit_code: Literal[0] = 0


def absolute(value: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise Rejected("input_rejected")
    return path


def read_json(path: Path, limit: int = MAX_RECEIPT) -> bytes:
    absolute(str(path))
    with path.open("rb") as handle:
        payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise Rejected("input_rejected")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Rejected("input_rejected")
            result[key] = value
        return result

    def reject_constant(value: str) -> Never:
        raise Rejected("input_rejected")

    json.loads(payload, object_pairs_hook=unique, parse_constant=reject_constant)
    return payload


def binding(artifact: Artifact) -> Path:
    path = absolute(artifact.path)
    if sha256_file(path) != artifact.sha256:
        raise Rejected("binding_mismatch")
    return path


def code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    return sha256_json(
        {
            str(path.relative_to(root)): sha256_file(path)
            for directory in ("scripts", "train", "eval", "data")
            for path in sorted((root / directory).rglob("*.py"))
        }
    )


def bind_files(directory: str, files: dict[str, str], *, adapter: bool = False) -> str:
    root = absolute(directory)
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise Rejected("input_rejected")
        binding(Artifact(path=str(root / relative), sha256=expected))
    required = (
        {"adapter_config.json", "adapter_model.safetensors"}
        if adapter
        else {"config.json", "tokenizer_config.json", "tokenizer.json"}
    )
    if not required <= files.keys() or not any(name.endswith(".safetensors") for name in files):
        raise Rejected("model_binding_incomplete")
    # Every local loadable config/tokenizer/template/weight must be bound, including shard indexes.
    actual = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and (p.suffix in {".json", ".safetensors", ".jinja", ".model"})
    }
    if not actual <= files.keys():
        raise Rejected("model_binding_incomplete")
    if not adapter:
        config = json.loads(read_json(root / "config.json"))
        if config.get("model_type") != "gemma4":
            raise Rejected("model_rejected")
        token_config = json.loads(read_json(root / "tokenizer_config.json"))
        if not token_config.get("chat_template") and not any(
            "chat_template" in name for name in files
        ):
            raise Rejected("model_binding_incomplete")
        for name in files:
            if name.endswith(".safetensors.index.json"):
                index = json.loads(read_json(root / name, 8 * 1024 * 1024))
                shards = set(index["weight_map"].values())
                if not shards or not shards <= files.keys():
                    raise Rejected("model_binding_incomplete")
    return sha256_json(files)


def validate(spec: Spec) -> tuple[list[Any], dict[str, Any]]:
    data = binding(spec.data)
    model_hash = bind_files(spec.model_dir, spec.model_files)
    adapter_hash = (
        bind_files(spec.adapter_dir, spec.adapter_files, adapter=True)
        if spec.adapter_dir and spec.adapter_files
        else None
    )
    admission_hash = None
    if spec.admission:
        path = binding(spec.admission)
        admitted = Admission.model_validate_json(read_json(path))
        if (admitted.data_sha256, admitted.pool, admitted.cohort, admitted.expected_rows) != (
            spec.data.sha256,
            spec.pool,
            spec.cohort,
            spec.expected_rows,
        ):
            raise Rejected("admission_mismatch")
        admission_hash = spec.admission.sha256
    samples = baseline.load_samples(data)
    if len(samples) != spec.expected_rows:
        raise Rejected("row_count_mismatch")
    ids: set[str] = set()
    families: set[str] = set()
    for sample in samples:
        media = [item.type for item in sample.prefix.media]
        expected = (
            []
            if spec.cohort == "text"
            else ["image" if spec.cohort == "screenshot" else spec.cohort]
        )
        if (
            spec.cohort == "crossmodal"
            and (
                sorted(media) != ["audio", "image"]
                or sorted(sample.meta.modality) != ["audio", "image"]
            )
        ) or (spec.cohort != "crossmodal" and media != expected):
            raise Rejected("cohort_mismatch")
        if (
            sample.meta.pool != spec.pool
            or len(sample.questions) != 1
            or not isinstance(sample.questions[0].target, HardTarget)
        ):
            raise Rejected("sample_rejected")
        if any(kind not in sample.meta.modality for kind in media) or (
            not media and sample.meta.modality != ["text"]
        ):
            raise Rejected("cohort_mismatch")
        baseline.target_index(sample.questions[0])
        if sample.id in ids:
            raise Rejected("sample_rejected")
        ids.add(sample.id)
        families.add(sample.meta.family_id)
    if spec.mode == "engineering" and spec.subset_limit is None and len(samples) > 50:
        raise Rejected("engineering_limit_required")
    if spec.subset_limit:
        samples = random.Random(42).sample(samples, min(len(samples), spec.subset_limit))
    return samples, {
        "model_manifest_sha256": model_hash,
        "adapter_manifest_sha256": adapter_hash,
        "admission_sha256": admission_hash,
        "unique_families": len(families),
    }


class BudgetProcessor:
    """Pass the budget per image call; processor merge defaults can override attributes."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)

    def __call__(self, **kwargs: Any) -> Any:
        if "images" in kwargs:
            kwargs["images_kwargs"] = {"max_soft_tokens": 1120}
        return self.processor(**kwargs)


def safe_environment() -> dict[str, str]:
    allowed = (
        "PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "TMPDIR",
        "LANG",
        "LC_ALL",
    )
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONNOUSERSITE="1",
    )
    return environment


def real_backend(
    spec: Spec,
) -> tuple[Any, Any, Callable[[], None], dict[str, str], Callable[[], int]]:
    torch = baseline.ensure_cuda_available()
    configure_torch_runtime(torch)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()
    transformers = importlib.import_module("transformers")
    processor = baseline.default_processor_factory(Path(spec.model_dir))
    configure_image_processor(processor, 1120)
    processor = BudgetProcessor(processor)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        spec.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if getattr(model.config, "model_type", None) != "gemma4":
        raise Rejected("model_rejected")
    if spec.adapter_dir:
        from scripts.s1_dev_eval import attach_adapter

        model = attach_adapter(model, Path(spec.adapter_dir))
    model.requires_grad_(False)
    model.eval().to("cuda")

    def forward(prepared: Any) -> dict[int, float]:
        inputs = baseline.move_inputs_to_cuda(prepared.inputs, torch, torch.bfloat16)
        with torch.inference_mode():
            logits = (
                model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                .logits[0, -1]
                .float()
            )
            if not bool(torch.isfinite(logits).all().item()):
                raise Rejected("backend_failed")
            values = torch.nn.functional.log_softmax(logits, dim=-1)
            return {
                index: float(values[index].item()) for index in prepared.answer_token_ids.values()
            }

    return (
        processor,
        forward,
        torch.cuda.empty_cache,
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": torch.cuda.get_device_name(0),
        },
        lambda: int(torch.cuda.max_memory_allocated()),
    )


def publish(path: Path, value: Contract) -> None:
    payload = (value.model_dump_json() + "\n").encode()
    temp = path.parent / ("." + uuid.uuid4().hex)
    try:
        with temp.open("xb") as handle:
            os.chmod(temp, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def worker(spec_path: Path, backend: Callable[..., Any] = real_backend) -> Receipt:
    started = time.monotonic()
    if spec_path.parent.stat().st_mode & 0o077:
        raise Rejected("private_root_required")
    raw = read_json(spec_path)
    spec = Spec.model_validate_json(raw)
    samples, evidence = validate(spec)
    initial_code = code_hash()
    processor, forward, cleanup, runtime, peak = backend(spec)
    try:
        records = baseline.run_pass(
            samples,
            processor=processor,
            media_root=absolute(spec.media_root),
            pass_index=1,
            forward=forward,
            records_sidecar=spec_path.parent / "records.jsonl",
        )
        if len(records) != len(samples):
            raise Rejected("partial_records")
        validate(spec)
        if initial_code != code_hash():
            raise Rejected("binding_mismatch")
        receipt = Receipt(
            mode=spec.mode,
            admission_status="not_required_engineering"
            if spec.mode == "engineering"
            else "admitted",
            spec_sha256=sha256_file(spec_path),
            data_sha256=spec.data.sha256,
            base_revision=spec.base_revision,
            code_sha256=initial_code,
            **evidence,
            tokenizer_template_sha256=sha256_json(
                {k: v for k, v in spec.model_files.items() if "token" in k or "template" in k}
            ),
            answer_map_sha256=sha256_json([r["answer_token_ids"] for r in records]),
            config_sha256=sha256_json(
                {
                    "seed": 42,
                    "image_budget": 1120,
                    "video_frames": 16,
                    "dtype": "bfloat16",
                    "attention": "sdpa",
                }
            ),
            records_sha256=sha256_file(spec_path.parent / "records.jsonl"),
            source_rows=spec.expected_rows,
            records=len(records),
            metrics=Metrics.model_validate(baseline.score_records(records)),
            wall_seconds=time.monotonic() - started,
            peak_memory_bytes=peak(),
            runtime=Runtime.model_validate(runtime),
        )
        publish(spec_path.parent / "receipt.json", receipt)
        return receipt
    finally:
        cleanup()


def terminate(process: subprocess.Popen[bytes]) -> int:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait()


def _execute(spec_path: Path, report: Path) -> Receipt:
    raw = read_json(spec_path)
    spec = Spec.model_validate_json(raw)
    root = absolute(spec.output_root)
    absolute(str(report))
    if report.exists():
        raise Rejected("report_exists")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077:
        raise Rejected("private_root_required")
    run = root / uuid.uuid4().hex
    run.mkdir(mode=0o700)
    private_spec = run / "spec.json"
    private_spec.write_bytes(raw)
    private_spec.chmod(0o600)
    expected_code = code_hash()
    with (run / "private.log").open("xb") as log:
        os.chmod(log.name, 0o600)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.private_stage_eval",
                "_worker",
                "--spec",
                str(private_spec),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            env=safe_environment(),
            cwd=Path(__file__).resolve().parents[1],
        )
        try:
            exit_code = process.wait(timeout=spec.timeout_seconds)
        except subprocess.TimeoutExpired:
            code = terminate(process)
            raise Rejected("timeout", code) from None
        except BaseException:
            code = terminate(process)
            raise Rejected("interrupted", code) from None
    if exit_code != 0:
        raise Rejected("backend_failed", exit_code)
    try:
        receipt = Receipt.model_validate_json(read_json(run / "receipt.json"))
        expected_records = min(spec.expected_rows, spec.subset_limit or spec.expected_rows)
        if (
            receipt.spec_sha256 != sha256_file(private_spec)
            or receipt.code_sha256 != expected_code
            or code_hash() != expected_code
            or receipt.data_sha256 != spec.data.sha256
            or receipt.mode != spec.mode
            or receipt.base_revision != spec.base_revision
            or receipt.records != expected_records
            or receipt.source_rows != spec.expected_rows
            or receipt.metrics.count != expected_records
            or receipt.admission_status
            != ("not_required_engineering" if spec.mode == "engineering" else "admitted")
            or receipt.admission_sha256 != (spec.admission.sha256 if spec.admission else None)
            or receipt.tokenizer_template_sha256
            != sha256_json(
                {k: v for k, v in spec.model_files.items() if "token" in k or "template" in k}
            )
            or receipt.config_sha256
            != sha256_json(
                {
                    "seed": 42,
                    "image_budget": 1120,
                    "video_frames": 16,
                    "dtype": "bfloat16",
                    "attention": "sdpa",
                }
            )
        ):
            raise ValueError
        if receipt.model_manifest_sha256 != sha256_json(
            spec.model_files
        ) or receipt.adapter_manifest_sha256 != (
            sha256_json(spec.adapter_files) if spec.adapter_files else None
        ):
            raise ValueError
        if receipt.records_sha256 != sha256_file(run / "records.jsonl"):
            raise ValueError
    except Exception:
        raise Rejected("invalid_receipt", exit_code) from None
    publish(report, receipt)
    return receipt


def execute(spec_path: Path, report: Path) -> Receipt:
    started = time.monotonic()
    absolute(str(report))
    if report.exists():
        raise Rejected("report_exists")
    spec_hash = None
    mode = None
    try:
        raw = read_json(spec_path)
        spec_hash = sha256_file(spec_path)
        mode = Spec.model_validate_json(raw).mode
        return _execute(spec_path, report)
    except BaseException as error:
        category = str(error) if isinstance(error, Rejected) else "job_rejected"
        try:
            failure = FailureReceipt.model_validate(
                {
                    "category": category,
                    "spec_sha256": spec_hash,
                    "mode": mode,
                    "wall_seconds": time.monotonic() - started,
                    "exit_code": error.exit_code if isinstance(error, Rejected) else None,
                }
            )
        except ValueError:
            failure = FailureReceipt(
                category="job_rejected",
                spec_sha256=spec_hash,
                mode=mode,
                wall_seconds=time.monotonic() - started,
                exit_code=None,
            )
        publish(report, failure)
        if isinstance(error, Rejected):
            raise
        raise Rejected("job_rejected") from None


class PrivateParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise Rejected("invalid_arguments")


def main(argv: list[str] | None = None) -> int:
    try:
        parser = PrivateParser(description=__doc__)
        parser.add_argument("action", choices=("run", "_worker"))
        parser.add_argument("--spec", type=Path, required=True)
        parser.add_argument("--report", type=Path)
        args = parser.parse_args(argv)
        if args.action == "_worker":
            os.umask(0o077)
            with (args.spec.parent / "worker.log").open("xb") as private_log:
                os.chmod(private_log.name, 0o600)
                os.dup2(private_log.fileno(), 1)
                os.dup2(private_log.fileno(), 2)
                worker(args.spec)
        else:
            if args.report is None:
                raise Rejected("invalid_arguments")
            receipt = execute(args.spec, args.report)
            print(receipt.model_dump_json())
        return 0
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code == 0:
            return 0
        category = str(error) if isinstance(error, Rejected) else "job_rejected"
        print(json.dumps({"execution_status": "rejected", "category": category}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
