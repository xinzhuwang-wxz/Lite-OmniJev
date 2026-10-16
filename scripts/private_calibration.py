"""Fixed fresh-UID calibration worker; only fitted artifacts and aggregates are exported."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from data.schema import Contract, HardTarget, Sample, answer_tokens
from scripts import private_stage_eval as stage
from scripts import s0_development_baselines as baseline
from scripts.check_tokens import probe
from scripts.s0_public_baselines import configure_torch_runtime, sha256_file, sha256_json
from train.calibration import (
    CalibrationInput,
    Identity,
    LogitRow,
    TemperatureArtifact,
    fit_temperature,
    save_temperature,
)
from train.multimodal import configure_image_processor

GROUP_COUNTS = {
    "text/choice": 3000,
    "image/choice": 3000,
    "screenshot/choice": 3000,
    "audio/choice": 1600,
    "audio/noul": 1400,
    "video/noul": 3000,
    "crossmodal/noul": 3000,
}
ALL_GROUPS = {
    f"{c}/{p}"
    for c in ("text", "image", "screenshot", "audio", "video", "crossmodal")
    for p in ("choice", "score", "noul")
}


class Progress(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    phase: Literal[
        "snapshot",
        "runtime",
        "plan",
        "media_snapshot",
        "inference",
        "fitting",
        "completed",
        "failed",
    ]
    cohort: Literal["text", "image", "screenshot", "audio", "video", "crossmodal"] | None = None
    processed: int = Field(ge=0)
    total: int = Field(gt=0)
    elapsed_seconds: FiniteFloat = Field(ge=0)
    nonfinite_count: int = Field(ge=0)
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def bounded_count(self) -> Progress:
        if self.processed > self.total:
            raise ValueError("progress_count_exceeds_total")
        return self


def publish_progress(path: Path, progress: Progress) -> None:
    """Exclusive temporary file plus atomic replace; no private payload fields."""
    require(not path.is_symlink(), "unsafe_progress_path")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        write_private(temporary, progress.model_dump())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def worker_progress(
    state: dict[str, Any],
    phase: str,
    processed: int,
    started: float,
    cohort: str | None = None,
    nonfinite_count: int = 0,
) -> None:
    progress = Progress.model_validate(
        {
            "phase": phase,
            "cohort": cohort,
            "processed": processed,
            "total": sum(s["expected_rows"] for s in state["spec"]["datasets"]),
            "elapsed_seconds": time.monotonic() - state.get("job_started_monotonic", started),
            "nonfinite_count": nonfinite_count,
            "spec_sha256": state["source_spec_sha256"],
            "code_sha256": state["code_sha256"],
        }
    )
    publish_progress(Path(state["output"]) / "progress.json", progress)


def forward_private(
    forward: Any, prepared: Any, state: dict[str, Any], processed: int, started: float, cohort: str
) -> dict[str, float]:
    try:
        return forward(prepared)  # type: ignore[no-any-return]
    except ValueError as error:
        if str(error) == "nonfinite_logits":
            worker_progress(state, "failed", processed, started, cohort, nonfinite_count=1)
        raise


class JobSpec(Contract):
    schema_version: Literal["private-calibration-job-v1"] = "private-calibration-job-v1"
    datasets: list[stage.Spec] = Field(min_length=7, max_length=7)
    producer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_python: stage.Artifact
    inventory: stage.Artifact
    inventory_receipt: stage.Artifact
    root_traverse_acl: bool = False
    max_snapshot_bytes: int = Field(default=64 * 1024**3, gt=0, le=128 * 1024**3)

    @model_validator(mode="after")
    def fixed_scope(self) -> JobSpec:
        first = self.datasets[0]
        totals: Counter[str] = Counter()
        for item in self.datasets:
            if item.mode != "S5" or item.pool != "calibration" or item.subset_limit is not None:
                raise ValueError("calibration_only_full_artifacts")
            for field in (
                "model_dir",
                "model_files",
                "adapter_dir",
                "adapter_files",
                "base_revision",
            ):
                if getattr(item, field) != getattr(first, field):
                    raise ValueError("one_frozen_model_required")
            totals[item.cohort] += item.expected_rows
        if dict(totals) != {
            c: 3000 for c in ("text", "image", "screenshot", "audio", "video", "crossmodal")
        }:
            raise ValueError("six_complete_calibration_cohorts_required")
        if first.adapter_dir is None or first.adapter_files is None:
            raise ValueError("frozen_adapter_required")
        return self


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_private(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def lexical_media(root: str, uri: str) -> str:
    parsed = urlsplit(uri)
    require(
        parsed.scheme in ("", "file") and not (parsed.netloc or parsed.query or parsed.fragment),
        "unsafe_media",
    )
    value = unquote(parsed.path)
    require(
        "\\" not in value and "\x00" not in value and ".." not in Path(value).parts, "unsafe_media"
    )
    path = Path(value) if Path(value).is_absolute() else Path(root) / value
    require(path.is_absolute() and path.is_relative_to(root), "media_outside_declared_root")
    return str(path)


def private_samples(state: dict[str, Any]) -> tuple[JobSpec, list[tuple[str, Sample, str]]]:
    spec = JobSpec.model_validate(state["spec"])
    samples: list[tuple[str, Sample, str]] = []
    counts: Counter[str] = Counter()
    for source, snapshot in zip(spec.datasets, state["datasets"], strict=True):
        path = Path(snapshot)
        require(sha256_file(path) == source.data.sha256, "snapshot_data_changed")
        rows = [
            Sample.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        require(
            len(rows) == source.expected_rows and len({s.id for s in rows}) == len(rows),
            "row_count_or_duplicates",
        )
        for sample in rows:
            require(sample.meta.pool == "calibration" and len(sample.questions) == 1, "wrong_pool")
            question = sample.questions[0]
            require(
                isinstance(question.target, HardTarget)
                and question.target.value in [*answer_tokens(question), "U"],
                "existing_hard_gold_required",
            )
            media = sorted(m.type for m in sample.prefix.media)
            expected = (
                []
                if source.cohort == "text"
                else ["audio", "image"]
                if source.cohort == "crossmodal"
                else ["image" if source.cohort == "screenshot" else source.cohort]
            )
            require(media == expected, "cohort_media_mismatch")
            key = f"{source.cohort}/{question.type}"
            require(key in GROUP_COUNTS, "unsupported_group")
            counts[key] += 1
            samples.append((source.cohort, sample, source.media_root))
    require(dict(counts) == GROUP_COUNTS, "real_group_counts_mismatch")
    return spec, samples


def plan_worker(state_path: Path) -> None:
    state = json.loads(state_path.read_text())
    spec, _ = private_samples(state)
    media: dict[str, dict[str, str]] = {}
    for source, snapshot in zip(spec.datasets, state["datasets"], strict=True):
        for line in Path(snapshot).read_text().splitlines():
            if not line.strip():
                continue
            sample = Sample.model_validate_json(line)
            for item in sample.prefix.media:
                path = lexical_media(source.media_root, item.uri)
                entry = {"sha256": item.sha256, "type": item.type}
                require(path not in media or media[path] == entry, "conflicting_media")
                media[path] = entry
    write_private(
        Path(state["output"]) / "media-plan.json", {"media": media, "groups": GROUP_COUNTS}
    )


def raw_backend(model_dir: Path, adapter_dir: Path) -> tuple[Any, Any, Any]:
    """Read actual pre-softmax legal logits; never relabel historical log-probabilities."""
    import torch
    import transformers
    from peft import PeftModel

    configure_torch_runtime(torch)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    processor = baseline.default_processor_factory(model_dir)
    configure_image_processor(processor, 1120)
    processor = stage.BudgetProcessor(processor)
    model: Any = transformers.AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(
        model, str(adapter_dir), local_files_only=True, is_trainable=False
    )
    model.requires_grad_(False)
    model.eval().to("cuda")

    def forward(prepared: Any) -> dict[str, float]:
        inputs = baseline.move_inputs_to_cuda(prepared.inputs, torch, torch.bfloat16)
        with torch.inference_mode():
            logits = (
                model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
                .logits[0, -1]
                .float()
            )
        require(bool(torch.isfinite(logits).all()), "nonfinite_logits")
        return {
            token: float(logits[index].item()) for token, index in prepared.answer_token_ids.items()
        }

    return processor, forward, model


def fit_groups(
    groups: dict[str, list[LogitRow]], identity: dict[str, Any], output: Path
) -> list[dict[str, Any]]:
    require(
        {key: len(rows) for key, rows in groups.items()} == GROUP_COUNTS,
        "real_group_counts_mismatch",
    )
    entries = []
    for key, rows in sorted(groups.items()):
        cohort, primitive = key.split("/")
        data = CalibrationInput(
            schema_version=1,
            pool="calibration",
            identity=Identity.model_validate(
                {**identity, "cohort": cohort, "primitive": primitive}
            ),
            rows=rows,
        )
        payload = data.model_dump_json().encode()
        private_path = output / (key.replace("/", "-") + "-input.json")
        with private_path.open("xb") as stream:
            os.chmod(private_path, 0o600)
            stream.write(payload)
        artifact = fit_temperature(data, hashlib.sha256(payload).hexdigest())
        filename = key.replace("/", "-") + "-temperature.json"
        save_temperature(artifact, output / filename)
        entries.append(
            {
                "group": key,
                "path": filename,
                "sha256": sha256_file(output / filename),
                "rows": artifact.rows,
                "temperature": artifact.temperature,
                "nll_before": artifact.nll_before,
                "nll_after": artifact.nll_after,
                "binding": artifact.binding.model_dump(),
            }
        )
    return entries


def fit_worker(state_path: Path) -> None:
    begun = time.monotonic()
    state = json.loads(state_path.read_text())
    spec, samples = private_samples(state)
    first = spec.datasets[0]
    model_dir, adapter_dir = Path(state["model"]), Path(state["adapter"])
    stage.bind_files(str(model_dir), first.model_files)
    assert first.adapter_files is not None
    stage.bind_files(str(adapter_dir), first.adapter_files, adapter=True)
    token_probe = probe(model_dir)
    require(token_probe["status"] == "passed", "official_token_probe_failed")
    mapping = token_probe["answer_token_ids"]
    require(len(mapping) == 33, "global33_answer_map_required")
    processor, forward, model = raw_backend(model_dir, adapter_dir)
    media_map = json.loads(Path(state["media_map"]).read_text())
    groups: dict[str, list[LogitRow]] = {key: [] for key in GROUP_COUNTS}
    try:
        worker_progress(state, "inference", 0, begun)
        for index, (cohort, original, media_root) in enumerate(samples):
            items = []
            for item in original.prefix.media:
                original_path = lexical_media(media_root, item.uri)
                relative = media_map[original_path]
                items.append(item.model_copy(update={"uri": relative}))
            sample = original.model_copy(
                update={"prefix": original.prefix.model_copy(update={"media": items})}
            )
            prepared = baseline.prepare_sample(sample, processor, Path(state["media_root"]))
            question = sample.questions[0]
            options = list(answer_tokens(question))
            require(
                prepared.answer_token_ids == {token: mapping[token] for token in [*options, "U"]},
                "prepared_token_map_mismatch",
            )
            target = question.target
            assert isinstance(target, HardTarget)
            groups[f"{cohort}/{question.type}"].append(
                LogitRow(
                    pool="calibration",
                    logits_kind="raw_logits",
                    complete_legal_logits=True,
                    options=options,
                    logits=forward_private(forward, prepared, state, index, begun, cohort),
                    target=str(target.value),
                )
            )
            if (
                (index + 1) % 100 == 0
                or index + 1 == len(samples)
                or samples[index + 1][0] != cohort
            ):
                worker_progress(state, "inference", index + 1, begun, cohort)
        worker_progress(state, "fitting", len(samples), begun)
        output = Path(state["output"])
        raw_receipt = {
            "schema_version": "calibration-raw-logits-receipt-v1",
            "spec_sha256": state["source_spec_sha256"],
            "code_sha256": state["code_sha256"],
            "model_files": first.model_files,
            "adapter_files": first.adapter_files,
            "official_template_sha256": token_probe["template_sha256"],
            "answer_token_map_sha256": hashlib.sha256(
                json.dumps(mapping, sort_keys=True).encode()
            ).hexdigest(),
            "datasets": [
                {"sha256": item.data.sha256, "cohort": item.cohort, "rows": item.expected_rows}
                for item in spec.datasets
            ],
            "groups": {
                key: {
                    "rows": len(rows),
                    "raw_rows_sha256": sha256_json([r.model_dump() for r in rows]),
                }
                for key, rows in groups.items()
            },
            "logits_kind": "raw_logits",
            "runtime_recipe": {
                "engine": "transformers",
                "dtype": "bfloat16",
                "adapter_mode": "native_lora",
            },
            "complete_legal_logits": True,
        }
        raw_receipt_path = output / "raw-receipt.json"
        write_private(raw_receipt_path, raw_receipt)
        identity = {
            "base_revision": first.base_revision,
            "adapter_weights_sha256": first.adapter_files["adapter_model.safetensors"],
            "official_template_sha256": token_probe["template_sha256"],
            "answer_token_map_sha256": raw_receipt["answer_token_map_sha256"],
            "raw_logits_receipt_sha256": sha256_file(raw_receipt_path),
            "seed": 42,
            "code_revision": spec.producer_revision,
        }
        entries = fit_groups(groups, identity, output)
        write_private(
            output / "receipt.json",
            {
                "status": "fitted",
                "exit_code": 0,
                "spec_sha256": state["source_spec_sha256"],
                "code_sha256": state["code_sha256"],
                "artifacts": entries,
                "missing_groups": sorted(ALL_GROUPS - GROUP_COUNTS.keys()),
                "raw_logits_receipt_sha256": identity["raw_logits_receipt_sha256"],
                "stage_acceptance": "not_evaluated",
                "wall_seconds": time.monotonic() - begun,
            },
        )
    finally:
        del model


class PosixACL:
    """Minimal libacl binding; no dependency or shell command is installed."""

    def __init__(self) -> None:
        library = ctypes.util.find_library("acl")
        require(library is not None, "libacl_unavailable")
        self.lib = ctypes.CDLL(library, use_errno=True)
        self.lib.acl_get_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.acl_get_file.restype = ctypes.c_void_p
        self.lib.acl_to_text.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ssize_t)]
        self.lib.acl_to_text.restype = ctypes.c_void_p
        self.lib.acl_from_text.argtypes = [ctypes.c_char_p]
        self.lib.acl_from_text.restype = ctypes.c_void_p
        self.lib.acl_set_file.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.acl_set_file.restype = ctypes.c_int
        self.lib.acl_free.argtypes = [ctypes.c_void_p]
        self.lib.acl_free.restype = ctypes.c_int

    def read(self, path: Path) -> str:
        acl = self.lib.acl_get_file(os.fsencode(path), 0x8000)
        require(bool(acl), "acl_read_failed")
        try:
            length = ctypes.c_ssize_t()
            text = self.lib.acl_to_text(acl, ctypes.byref(length))
            require(bool(text), "acl_text_failed")
            try:
                return ctypes.string_at(text, length.value).decode()
            finally:
                self.lib.acl_free(text)
        finally:
            self.lib.acl_free(acl)

    def write(self, path: Path, text: str) -> None:
        acl = self.lib.acl_from_text(text.encode())
        require(bool(acl), "acl_parse_failed")
        try:
            require(self.lib.acl_set_file(os.fsencode(path), 0x8000, acl) == 0, "acl_set_failed")
        finally:
            self.lib.acl_free(acl)


def restrict_child() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(38, 1, 0, 0, 0) == 0, "no_new_privileges_unavailable")


def verify_child_identity() -> None:
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    require(
        os.geteuid() != 0
        and status["NoNewPrivs"].strip() == "1"
        and all(int(status[key].strip(), 16) == 0 for key in ("CapEff", "CapPrm", "CapAmb")),
        "unsafe_worker_identity",
    )


def durable_json(path: Path, value: Any) -> None:
    write_private(path, value)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def revoke_uid_processes(uid: int) -> None:
    # Fresh permanently leased UID; never targets any pre-existing user process.
    for _ in range(3):
        found = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status = (entry / "status").read_text()
                identities = next(line for line in status.splitlines() if line.startswith("Uid:"))
                if uid in map(int, identities.split()[1:]):
                    found = True
                    os.kill(int(entry.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                continue
        if not found:
            return
        time.sleep(0.05)


@contextmanager
def guarded_acl(uid: int, backup: Path) -> Iterator[dict[str, Any]]:
    """Serialize globally; independent guardian restores on supervisor SIGKILL/EOF."""
    import fcntl

    from scripts.private_gold_job import bounded_read

    lock_path = Path("/var/tmp/omnijev-root-traverse-acl.lock")
    journal = Path("/var/tmp/omnijev-root-traverse-acl-recovery.json")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    require(
        info.st_uid == 0 and stat.S_ISREG(info.st_mode) and info.st_mode & 0o777 == 0o600,
        "unsafe_acl_lock",
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        api = PosixACL()
        if journal.exists() or journal.is_symlink():
            info = journal.lstat()
            require(
                info.st_uid == 0 and stat.S_ISREG(info.st_mode) and info.st_mode & 0o777 == 0o600,
                "unsafe_recovery_journal",
            )
            previous = json.loads(bounded_read(journal, 16384))
            require(
                isinstance(previous["uid"], int) and previous["uid"] >= 10000, "unsafe_recovery_uid"
            )
            before = previous["acl_before"]
            allowed = set(before.splitlines()) | {f"user:{previous['uid']}:--x", "mask::--x"}
            require(
                set(before.splitlines()) == {"user::rwx", "group::---", "other::---"}
                and set(api.read(Path("/root")).splitlines())
                in (set(before.splitlines()), allowed),
                "unexpected_recovery_acl",
            )
            revoke_uid_processes(previous["uid"])
            api.write(Path("/root"), before)
            require(api.read(Path("/root")) == before, "recovery_restore_failed")
            journal.unlink()
        before = api.read(Path("/root"))
        require(
            set(before.splitlines()) == {"user::rwx", "group::---", "other::---"},
            "unexpected_root_acl",
        )
        durable_json(journal, {"acl_before": before, "uid": uid, "parent_pid": os.getpid()})
        reader, writer = os.pipe()
        guardian = os.fork()
        if guardian == 0:
            os.close(writer)
            os.setsid()
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                while os.read(reader, 1):
                    pass
                revoke_uid_processes(uid)
                api.write(Path("/root"), before)
                require(api.read(Path("/root")) == before, "guardian_restore_failed")
                journal.unlink()
                directory = os.open(journal.parent, os.O_RDONLY | os.O_DIRECTORY)
                os.fsync(directory)
                os.close(directory)
                os._exit(0)
            except BaseException:
                os._exit(2)
        os.close(reader)
        try:
            with traverse_acl(uid, backup, api=api) as evidence:
                evidence["guardian_pid"] = guardian
                evidence["global_serialization"] = True
                yield evidence
        finally:
            os.close(writer)
            _, status = os.waitpid(guardian, 0)
            require(os.waitstatus_to_exitcode(status) == 0, "acl_guardian_failed")
    finally:
        os.close(fd)


@contextmanager
def traverse_acl(uid: int, backup: Path, *, api: Any = None) -> Iterator[dict[str, Any]]:
    """Only /root, only one fresh UID, only execute; preserve an exact recovery record."""
    if api is None:
        with guarded_acl(uid, backup) as evidence:
            yield evidence
        return
    path = Path("/root")
    before = api.read(path)
    # Refuse extended ACLs: changing their mask could broaden existing named permissions.
    require(
        set(before.splitlines()) == {"user::rwx", "group::---", "other::---"}, "unexpected_root_acl"
    )
    durable_json(backup, {"path": "/root", "uid": uid, "acl_before": before})
    evidence = {
        "path": "/root",
        "uid": uid,
        "permission": "--x",
        "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
    }
    old_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum: int, frame: Any) -> None:
        raise RuntimeError("job_terminated")

    signal.signal(signal.SIGTERM, terminate)
    try:
        api.write(path, before + f"user:{uid}:--x\nmask::--x\n")
        evidence["granted_sha256"] = hashlib.sha256(api.read(path).encode()).hexdigest()
        yield evidence
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        api.write(path, before)
        restored = api.read(path)
        require(restored == before, "root_acl_restore_failed")
        evidence["after_sha256"] = hashlib.sha256(restored.encode()).hexdigest()
        evidence["restored"] = True


def public_readonly(path: Path) -> None:
    """Only already public-readable immutable assets may use the traversal grant."""
    from scripts.private_gold_job import safe_path

    safe_path(path)
    info = path.stat()
    require(
        stat.S_ISREG(info.st_mode)
        and info.st_uid == 0
        and bool(info.st_mode & 0o004)
        and not info.st_mode & 0o022,
        "public_readonly_asset_required",
    )
    for parent in path.parents:
        if parent == Path("/root"):
            continue
        info = parent.stat()
        require(
            info.st_uid == 0 and bool(info.st_mode & 0o001) and not info.st_mode & 0o022,
            "unsafe_asset_ancestor",
        )


def snapshot_media(
    media: dict[str, dict[str, str]],
    roots: list[Path],
    inputs: Path,
    *,
    public_in_place: bool,
    remaining_bytes: int,
    copy_bound: Callable[[Path, str, Path], None],
) -> tuple[dict[str, str], dict[str, int]]:
    """Copy only declared private assets, after a unique-byte capacity check."""
    from scripts.private_gold_job import safe_path

    planned: dict[str, tuple[Path, str, Path | None]] = {}
    unique: dict[Path, int] = {}
    for original, entry in media.items():
        path = Path(original)
        safe_path(path)
        require(any(path.is_relative_to(root) for root in roots), "media_plan_escape")
        digest = entry["sha256"]
        require(
            len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
            "invalid_media_digest",
        )
        info = path.stat()
        require(stat.S_ISREG(info.st_mode), "regular_media_required")
        # Hash every declared path even when its digest duplicates another asset.
        require(sha256_file(path) == digest, "media_changed")
        public = False
        if public_in_place:
            try:
                public_readonly(path)
                public = True
            except ValueError:
                pass
        target = None if public else inputs / "media" / (digest + path.suffix.lower())
        if target is not None:
            require(target not in unique or unique[target] == info.st_size, "media_size_conflict")
            unique[target] = info.st_size
        planned[original] = (path, digest, target)
    needed = sum(unique.values())
    capacity = os.statvfs(inputs)
    available = capacity.f_bavail * capacity.f_frsize
    require(needed <= remaining_bytes, "snapshot_budget_exceeded")
    require(needed + 256 * 1024**2 <= available, "private_media_disk_budget_exceeded")
    copied: set[Path] = set()
    mapping = {}
    for original, (path, digest, target) in planned.items():
        if target is None:
            mapping[original] = str(path).lstrip("/")
        else:
            if target not in copied:
                copy_bound(path, digest, target)
                copied.add(target)
            mapping[original] = (
                str(target).lstrip("/") if public_in_place else str(target.relative_to(inputs))
            )
    return mapping, {
        "declared_assets": len(media),
        "private_snapshot_unique_assets": len(unique),
        "private_snapshot_bytes": needed,
        "available_bytes_before_copy": available,
        "public_assets_in_place": sum(target is None for _, _, target in planned.values()),
    }


def _execute(
    source: Path,
    expected: str,
    work_root: Path,
    report: Path,
    worker_uid: int,
    probe_uid: int,
    timeout: int,
) -> dict[str, Any]:
    """Reuse the existing UID broker; data parsing happens only in fixed child phases."""
    from scripts import private_gold_job as broker

    begun = time.monotonic()
    require(sys.platform == "linux" and os.geteuid() == 0, "boundary_unavailable")
    require(
        1 <= timeout <= 86400
        and worker_uid >= 10000
        and probe_uid >= 10000
        and worker_uid != probe_uid
        and broker.unused_uid(worker_uid)
        and broker.unused_uid(probe_uid),
        "invalid_arguments",
    )
    for path in (source, work_root, report.parent):
        broker.safe_path(path)
    require(
        work_root.is_relative_to("/var/tmp") and work_root != Path("/var/tmp"),
        "private_work_root_required",
    )
    for path in (work_root, report.parent):
        info = path.stat()
        require(
            info.st_uid == 0 and stat.S_ISDIR(info.st_mode) and not info.st_mode & 0o022,
            "root_owned_directory_required",
        )
    require(not report.exists() and not report.is_symlink(), "report_exists")
    raw_spec = broker.bounded_read(source, 4 * 1024**2)
    require(hashlib.sha256(raw_spec).hexdigest() == expected, "spec_sha_mismatch")
    spec = JobSpec.model_validate_json(raw_spec)
    runtime = Path(spec.runtime_python.path)
    broker.safe_path(runtime.parent)
    broker.safe_path(runtime.resolve())
    require(
        (
            runtime.is_relative_to("/var/tmp")
            or (spec.root_traverse_acl and runtime.is_relative_to("/root/autodl-tmp"))
        )
        and sha256_file(runtime) == spec.runtime_python.sha256
        and sha256_file(runtime) == sha256_file(Path(sys.executable).resolve()),
        "verified_reachable_runtime_required",
    )
    inventory = stage.binding(spec.inventory)
    audit = json.loads(broker.bounded_read(stage.binding(spec.inventory_receipt), 1024**2))
    require(
        audit.get("outcome") == "isolated"
        and audit.get("ordinary_identity_boundary") is True
        and audit.get("source_manifest_sha256") == spec.inventory.sha256,
        "current_independent_inventory_required",
    )
    from scripts.gold_inventory import Manifest

    inventory_manifest = Manifest.model_validate_json(broker.bounded_read(inventory, 4 * 1024**2))
    registered = {
        (row.path, row.sha256)
        for row in inventory_manifest.samples
        if row.expected_pool == "calibration" and row.exposure == "current"
    }
    sealed_sources = [
        Path(row.path)
        for row in inventory_manifest.samples
        if row.expected_pool in ("calibration", "holdout")
    ]
    for item in spec.datasets:
        stage.verify_catalog(item)
        require(
            (item.data.path, item.data.sha256) in registered, "calibration_not_in_current_inventory"
        )
    first = spec.datasets[0]
    stage.bind_files(first.model_dir, first.model_files)
    assert first.adapter_dir is not None and first.adapter_files is not None
    stage.bind_files(first.adapter_dir, first.adapter_files, adapter=True)
    broker.reserve_ids(work_root, worker_uid, probe_uid)
    code_sha = stage.code_hash()
    progress_path = report.with_suffix(".public-progress.json")
    require(not progress_path.exists() and not progress_path.is_symlink(), "progress_exists")
    total = sum(item.expected_rows for item in spec.datasets)

    def phase_progress(phase: str, processed: int = 0) -> None:
        publish_progress(
            progress_path,
            Progress.model_validate(
                {
                    "phase": phase,
                    "cohort": None,
                    "processed": processed,
                    "total": total,
                    "elapsed_seconds": time.monotonic() - begun,
                    "nonfinite_count": 0,
                    "spec_sha256": expected,
                    "code_sha256": code_sha,
                }
            ),
        )

    phase_progress("snapshot")
    run = work_root / uuid.uuid4().hex
    exported = report.with_suffix(".artifacts")
    require(not exported.exists(), "artifact_export_exists")
    credential_sentinel = work_root / (run.name + "-credential-deny.json")
    write_private(credential_sentinel, {"purpose": "deny_probe_only"})
    acl_evidence: dict[str, Any] = {}
    acl_context = (
        traverse_acl(worker_uid, work_root / (run.name + "-root-acl.json"))
        if spec.root_traverse_acl
        else nullcontext(acl_evidence)
    )
    with acl_context as acl_evidence, broker.private_run(run, worker_uid):
        inputs, output = run / "inputs", run / "output"
        for path, owner, mode in [(inputs, 0, 0o750), (output, worker_uid, 0o700)]:
            path.mkdir(mode=mode)
            os.chown(path, owner, worker_uid)
        run.chmod(0o750)
        consumed = 0

        def copy_bound(original: Path, digest: str, target: Path) -> None:
            nonlocal consumed
            broker.safe_path(original)
            info = original.stat()
            require(stat.S_ISREG(info.st_mode), "regular_file_required")
            consumed += info.st_size
            require(consumed <= spec.max_snapshot_bytes, "snapshot_budget_exceeded")
            require(sha256_file(original) == digest, "source_file_changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            if (
                info.st_uid == 0
                and info.st_mode & 0o004
                and not info.st_mode & 0o022
                and info.st_dev == inputs.stat().st_dev
            ):
                os.link(original, target)  # Never chmod/chown a hardlink to an original file.
            else:
                shutil.copyfile(original, target)
                os.chown(target, 0, worker_uid)
                target.chmod(0o440)
            require(sha256_file(target) == digest, "snapshot_file_changed")

        source_root = Path(__file__).resolve().parents[1]
        code_root = inputs / "code"
        for directory in ("scripts", "train", "eval", "data"):
            for path in sorted((source_root / directory).rglob("*.py")):
                copy_bound(path, sha256_file(path), code_root / path.relative_to(source_root))
        data_paths = []
        for index, item in enumerate(spec.datasets):
            target = inputs / "data" / f"{index}.jsonl"
            copy_bound(Path(item.data.path), item.data.sha256, target)
            data_paths.append(str(target))
        for kind, directory, files in [
            ("model", first.model_dir, first.model_files),
            ("adapter", first.adapter_dir, first.adapter_files),
        ]:
            for name, digest in files.items():
                if spec.root_traverse_acl:
                    public_readonly(Path(directory) / name)
                else:
                    copy_bound(Path(directory) / name, digest, inputs / kind / name)
        state = {
            "job_started_monotonic": begun,
            "spec": spec.model_dump(),
            "source_spec_sha256": expected,
            "code_sha256": code_sha,
            "datasets": data_paths,
            "model": first.model_dir if spec.root_traverse_acl else str(inputs / "model"),
            "adapter": first.adapter_dir if spec.root_traverse_acl else str(inputs / "adapter"),
            "output": str(output),
            "media_root": "/" if spec.root_traverse_acl else str(inputs),
            "media_map": str(inputs / "media-map.json"),
        }
        state_path = inputs / "state.json"
        write_private(state_path, state)
        os.chown(state_path, 0, worker_uid)
        state_path.chmod(0o440)

        def readonly_directories() -> None:
            for path in [inputs, *[p for p in inputs.rglob("*") if p.is_dir()]]:
                os.chown(path, 0, worker_uid)
                path.chmod(0o550)

        readonly_directories()
        require(
            broker.denied_probe(probe_uid, probe_uid, [state_path, output], [])
            and broker.denied_probe(
                worker_uid,
                worker_uid,
                sealed_sources,
                [state_path, report],
            ),
            "ordinary_identity_probe_failed",
        )
        env = stage.safe_environment()
        env.update(
            PYTHONPATH=str(code_root),
            HOME=str(output),
            TMPDIR=str(output),
            PYTHONDONTWRITEBYTECODE="1",
        )

        def child(phase: str, seconds: int) -> None:
            phase_progress(
                {"_runtime_probe": "runtime", "_plan": "plan", "_fit": "inference"}[phase]
            )
            with (output / (phase + ".log")).open("xb") as log:
                os.chmod(log.name, 0o600)
                os.chown(log.name, worker_uid, worker_uid)
                process = subprocess.Popen(
                    [
                        str(runtime),
                        "-m",
                        "scripts.private_calibration",
                        phase,
                        "--state",
                        str(state_path),
                    ],
                    cwd=output,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    preexec_fn=restrict_child,
                    user=worker_uid,
                    group=worker_uid,
                    extra_groups=[],
                    start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + seconds
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("private_worker_timeout")
                        try:
                            code = process.wait(timeout=min(5, remaining))
                        except subprocess.TimeoutExpired:
                            code = None
                        private_progress = output / "progress.json"
                        if private_progress.exists():
                            progress = Progress.model_validate_json(
                                broker.bounded_read(private_progress, 4096)
                            )
                            require(
                                progress.spec_sha256 == expected
                                and progress.code_sha256 == code_sha
                                and progress.total == total,
                                "private_progress_binding_mismatch",
                            )
                            publish_progress(progress_path, progress)
                        if code is not None:
                            break
                except BaseException:
                    stage.terminate(process)
                    raise
                require(code == 0, "private_calibration_worker_failed")

        if spec.root_traverse_acl:
            require(
                broker.denied_probe(
                    worker_uid,
                    worker_uid,
                    [
                        Path("/root"),
                        credential_sentinel,
                        *([Path("/root/.ssh")] if Path("/root/.ssh").exists() else []),
                    ],
                    [Path(first.model_dir) / n for n in first.model_files]
                    + [Path(__file__).resolve()],
                ),
                "runtime_acl_negative_probe_failed",
            )
        child("_runtime_probe", 60)
        child("_plan", 120)
        plan = json.loads(broker.bounded_read(output / "media-plan.json", 64 * 1024**2))
        require(
            plan["groups"] == GROUP_COUNTS and isinstance(plan["media"], dict),
            "invalid_private_media_plan",
        )
        phase_progress("media_snapshot")
        media_map, media_snapshot = snapshot_media(
            plan["media"],
            [Path(item.media_root) for item in spec.datasets],
            inputs,
            public_in_place=spec.root_traverse_acl,
            remaining_bytes=spec.max_snapshot_bytes - consumed,
            copy_bound=copy_bound,
        )
        write_private(inputs / "media-map.json", media_map)
        os.chown(inputs / "media-map.json", 0, worker_uid)
        (inputs / "media-map.json").chmod(0o440)
        readonly_directories()
        child("_fit", timeout)
        receipt = json.loads(broker.bounded_read(output / "receipt.json", 1024**2))
        require(
            receipt["status"] == "fitted"
            and receipt["exit_code"] == 0
            and receipt["spec_sha256"] == expected
            and receipt["code_sha256"] == code_sha
            and receipt["missing_groups"] == sorted(ALL_GROUPS - GROUP_COUNTS.keys()),
            "invalid_worker_receipt",
        )
        require(
            {e["group"] for e in receipt["artifacts"]} == set(GROUP_COUNTS)
            and len(receipt["artifacts"]) == 7,
            "invalid_group_exports",
        )
        require(stage.code_hash() == code_sha, "code_changed_during_job")
        stage.bind_files(first.model_dir, first.model_files)
        stage.bind_files(first.adapter_dir, first.adapter_files, adapter=True)
        raw_receipt = broker.bounded_read(output / "raw-receipt.json", 1024**2)
        require(
            hashlib.sha256(raw_receipt).hexdigest() == receipt["raw_logits_receipt_sha256"],
            "raw_receipt_changed",
        )
        official = probe(Path(first.model_dir))
        require(official["status"] == "passed", "official_token_probe_failed")
        raw_meta = json.loads(raw_receipt)
        require(
            raw_meta["code_sha256"] == code_sha
            and raw_meta["spec_sha256"] == expected
            and raw_meta["model_files"] == first.model_files
            and raw_meta["adapter_files"] == first.adapter_files
            and raw_meta["logits_kind"] == "raw_logits"
            and raw_meta["runtime_recipe"]
            == {"engine": "transformers", "dtype": "bfloat16", "adapter_mode": "native_lora"}
            and raw_meta["complete_legal_logits"] is True
            and raw_meta["official_template_sha256"] == official["template_sha256"]
            and raw_meta["answer_token_map_sha256"]
            == hashlib.sha256(
                json.dumps(official["answer_token_ids"], sort_keys=True).encode()
            ).hexdigest(),
            "invalid_raw_provenance",
        )
        exported.mkdir(mode=0o700)
        for entry in receipt["artifacts"]:
            name = entry["path"]
            require(
                name == entry["group"].replace("/", "-") + "-temperature.json",
                "invalid_artifact_path",
            )
            payload = broker.bounded_read(output / name, 1024**2)
            artifact = TemperatureArtifact.model_validate_json(payload)
            require(
                hashlib.sha256(payload).hexdigest() == entry["sha256"]
                and artifact.binding.model_dump() == entry["binding"]
                and artifact.rows == GROUP_COUNTS[entry["group"]]
                and artifact.binding.adapter_weights_sha256
                == first.adapter_files["adapter_model.safetensors"]
                and artifact.binding.base_revision == first.base_revision
                and artifact.binding.code_revision == spec.producer_revision
                and artifact.binding.seed == 42
                and f"{artifact.binding.cohort}/{artifact.binding.primitive}" == entry["group"]
                and artifact.binding.official_template_sha256
                == raw_meta["official_template_sha256"]
                and artifact.binding.answer_token_map_sha256 == raw_meta["answer_token_map_sha256"]
                and artifact.binding.raw_logits_receipt_sha256
                == receipt["raw_logits_receipt_sha256"],
                "invalid_temperature_export",
            )
            with (exported / name).open("xb") as stream:
                os.chmod(stream.name, 0o600)
                stream.write(payload)
        (exported / "raw-receipt.json").write_bytes(raw_receipt)
        (exported / "raw-receipt.json").chmod(0o600)
    # The original run, rows, inputs, and worker logs remain root-only after lease revocation.
    result = {
        **receipt,
        "media_snapshot": media_snapshot,
        "root_traverse_acl": acl_evidence,
        "boundary": "fresh_ordinary_uid_fixed_calibration_job",
        "ordinary_identity_boundary": True,
        "runtime_python_sha256": spec.runtime_python.sha256,
        "inventory_sha256": spec.inventory.sha256,
        "export_directory": str(exported),
        "duration_seconds": time.monotonic() - begun,
        "no_network_namespace_claim": True,
        "cost_usd": None,
    }
    write_private(report, result)
    phase_progress("completed", total)
    return result


def execute(
    source: Path,
    expected: str,
    work_root: Path,
    report: Path,
    worker_uid: int,
    probe_uid: int,
    timeout: int,
) -> dict[str, Any]:
    progress_path = report.with_suffix(".public-progress.json")
    existed = progress_path.exists() or progress_path.is_symlink()
    try:
        return _execute(source, expected, work_root, report, worker_uid, probe_uid, timeout)
    except BaseException:
        if not existed and progress_path.exists():
            from scripts.private_gold_job import bounded_read

            progress = Progress.model_validate_json(bounded_read(progress_path, 4096))
            require(progress.spec_sha256 == expected, "failure_progress_binding_mismatch")
            publish_progress(progress_path, progress.model_copy(update={"phase": "failed"}))
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("phase", choices=["_plan", "_fit", "_runtime_probe"])
    parser.add_argument("--state", type=Path)
    try:
        args = parser.parse_args(argv)
        verify_child_identity()
        if args.phase == "_runtime_probe":
            import peft
            import torch
            import transformers

            require(
                bool(torch.__version__ and transformers.__version__ and peft.__version__),
                "runtime_unavailable",
            )
        else:
            require(args.state is not None, "private_state_required")
            (plan_worker if args.phase == "_plan" else fit_worker)(args.state)
        return 0
    except BaseException:
        # All worker details remain inside the private log; never echo validation rows.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
