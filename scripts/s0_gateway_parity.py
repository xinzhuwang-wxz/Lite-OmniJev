"""S0 gateway parity harness for the post-HF/vLLM release gate.

The gateway exposes conditional answer probabilities and p(U), not full raw
vocabulary logprobs. This script therefore requires a separate passed HF/vLLM
S0 report and only checks that gateway semantics match those frozen non-holdout
conditional/U readouts for the same canonical 100-case cohort.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from data.schema import Question, answer_tokens
from scripts.assets import write_json
from scripts.s0_parity import row_id_hash, row_to_json, rows_from_json, sha256_json
from scripts.s0_prepare import read_manifest
from serve.gateway.app import DecisionRequest, InputQuestion, core_question
from train.multimodal import PreparationCase

SCHEMA = "lite-omnijev.s0_gateway_parity.report.v1"
SEED = 42
THRESHOLD = 0.001
EXPECTED_COHORT_SIZE = 100
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCE_FILES = (
    "scripts/s0_gateway_parity.py",
    "scripts/s0_parity.py",
    "scripts/s0_prepare.py",
    "serve/gateway/app.py",
    "serve/vllm_chat.py",
    "data/schema.py",
    "train/multimodal.py",
    "train/readout.py",
)


class GatewayParityError(RuntimeError):
    """Raised for inadmissible gateway parity evidence."""


class ReportExistsError(GatewayParityError):
    """Raised before work when a report path would be overwritten."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_existing(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ReportExistsError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def git_revision() -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip() or None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": run("status", "--short"),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("httpx", "pydantic", "fastapi", "starlette", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment_summary() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "packages": package_versions(),
    }


def code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise GatewayParityError(f"required source file missing: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def source_identity() -> dict[str, Any]:
    return {"repo_root": str(REPO_ROOT), "git": git_revision(), "code_hashes": code_hashes()}


def manifest_sha256(path: Path) -> str:
    return sha256_file(path)


def case_fingerprint(case: PreparationCase) -> str:
    return sha256_json(
        {
            "id": case.id,
            "category": case.category,
            "video_frames": case.video_frames,
            "prefix": case.prefix.model_dump(),
            "question": case.question.model_dump(),
            "source_family": case.meta.family_id,
            "source_dataset": case.meta.source_dataset,
            "source_revision": case.meta.source_revision,
            "source_sha256": case.meta.source_sha256,
            "pool": case.meta.pool,
        }
    )


def manifest_binding(path: Path, cases: Sequence[PreparationCase]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": manifest_sha256(path),
        "cases": len(cases),
        "case_fingerprints_sha256": sha256_json([case_fingerprint(case) for case in cases]),
        "source_families_sha256": sha256_json(sorted({case.meta.family_id for case in cases})),
    }


def _strict_float(value: Any, *, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise GatewayParityError(f"{field}_not_finite")
    return float(value)


def _strict_probability(value: Any, *, field: str) -> float:
    result = _strict_float(value, field=field)
    if result < 0.0 or result > 1.0:
        raise GatewayParityError(f"{field}_outside_probability_range")
    return result


def input_question_from_question(question: Question) -> dict[str, Any]:
    if question.type == "choice":
        tokens = answer_tokens(question)
        options = list(question.options or [])
        return {
            "type": "choice",
            "instructions": question.instruction,
            "criteria": dict(zip(tokens, options, strict=True)),
        }
    if question.type == "score":
        return {
            "type": "score",
            "instructions": question.instruction,
            "criteria": list(question.score_descriptions or []),
        }
    return {"type": "noul", "instructions": question.instruction}


def expected_labels(question: Question) -> list[str]:
    return list(answer_tokens(question))


def decision_request_from_case(case: PreparationCase) -> dict[str, Any]:
    return {
        "state": case.prefix.state,
        "media": [item.model_dump() for item in case.prefix.media],
        "video_frames": case.video_frames,
        "model": "lite-omnijev-e4b-dev",
        "questions": {case.question.qid: input_question_from_question(case.question)},
    }


def validate_health(body: Mapping[str, Any], *, expected_model_version: str) -> dict[str, Any]:
    if body.get("status") != "ok":
        raise GatewayParityError("gateway_health_status_not_ok")
    if body.get("mode") != "canonical-chat":
        raise GatewayParityError("gateway_health_not_canonical_chat")
    if body.get("inference_configured") is not True:
        raise GatewayParityError("gateway_inference_not_configured")
    if body.get("model_version") != expected_model_version:
        raise GatewayParityError("gateway_model_version_mismatch")
    return {
        "status": body["status"],
        "mode": body["mode"],
        "inference_configured": True,
        "model_version": body["model_version"],
        "model_quality_verified": body.get("model_quality_verified"),
    }


def validate_gateway_record(
    record: Mapping[str, Any], *, reference_report: Mapping[str, Any], reference_sha256: str
) -> dict[str, Any]:
    provenance = reference_report.get("provenance")
    server = reference_report.get("server", {})
    if record.get("operator_attested") is not True:
        raise GatewayParityError("gateway_record_operator_attestation_required")
    if record.get("s0_hf_vllm_report_sha256") != reference_sha256:
        raise GatewayParityError("gateway_record_reference_report_sha_mismatch")
    if record.get("provenance") != provenance:
        raise GatewayParityError("gateway_record_provenance_mismatch")
    if record.get("upstream_server_record_sha256") != server.get("record_sha256"):
        raise GatewayParityError("gateway_record_upstream_server_mismatch")
    model_version = record.get("model_version")
    if not isinstance(model_version, str) or not model_version:
        raise GatewayParityError("gateway_record_model_version_required")
    launch = record.get("gateway_launch_command")
    if not isinstance(launch, str) or not launch.strip():
        raise GatewayParityError("gateway_launch_command_required")
    return {
        "operator_attested": True,
        "remote_attestation": False,
        "model_version": model_version,
        "gateway_launch_command_sha256": sha256_bytes(launch.encode()),
        "upstream_server_record_sha256": record["upstream_server_record_sha256"],
        "s0_hf_vllm_report_sha256": reference_sha256,
        "record_sha256": sha256_json(record),
    }


def _require_passed_report(label: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GatewayParityError(f"{label}_missing")
    if value.get("passed") is not True or value.get("status") != "passed":
        raise GatewayParityError(f"{label}_not_passed")
    if value.get("samples") != EXPECTED_COHORT_SIZE:
        raise GatewayParityError(f"{label}_requires_100_samples")
    if value.get("threshold_exclusive") != THRESHOLD:
        raise GatewayParityError(f"{label}_threshold_mismatch")
    maxima = value.get("max_differences")
    if not isinstance(maxima, Mapping):
        raise GatewayParityError(f"{label}_max_differences_missing")
    for key in ("raw_probabilities", "joint", "conditional", "p_U"):
        if _strict_float(maxima.get(key), field=f"{label}_{key}") >= THRESHOLD:
            raise GatewayParityError(f"{label}_{key}_exceeds_threshold")
    return value


def load_reference_report(path: Path) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("action") != "compare-vllm" or report.get("status") != "passed":
        raise GatewayParityError("s0_hf_vllm_report_must_be_passed_compare_vllm")
    reference_validation = _require_passed_report(
        "reference_validation", report.get("reference_validation")
    )
    if reference_validation.get("label") != "reference_rows_match_current_canonical_http_inputs":
        raise GatewayParityError("reference_validation_label_mismatch")
    repeatability = _require_passed_report("vllm_repeatability", report.get("vllm_repeatability"))
    if repeatability.get("label") != "vllm_repeatability":
        raise GatewayParityError("vllm_repeatability_label_mismatch")
    parity = _require_passed_report("parity", report.get("parity"))
    if parity.get("label") != "hf_reference_vs_vllm":
        raise GatewayParityError("parity_label_mismatch")
    rows_raw = report.get("rows")
    if not isinstance(rows_raw, list) or len(rows_raw) != EXPECTED_COHORT_SIZE:
        raise GatewayParityError("s0_hf_vllm_report_requires_100_rows")
    try:
        rows = rows_from_json(rows_raw)
    except Exception as exc:
        raise GatewayParityError("s0_hf_vllm_rows_fail_strict_validation") from exc
    canonical_rows = [row_to_json(row) for row in rows]
    if rows_raw != canonical_rows:
        raise GatewayParityError("s0_hf_vllm_rows_not_strict_canonical")
    return report, sha256_bytes(raw), canonical_rows


def validate_manifest_binding(
    *, manifest: Path, cases: Sequence[PreparationCase], reference_report: Mapping[str, Any]
) -> dict[str, Any]:
    binding = manifest_binding(manifest, cases)
    provenance = reference_report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise GatewayParityError("reference_provenance_missing")
    manifest_record = provenance.get("manifest")
    if not isinstance(manifest_record, Mapping):
        raise GatewayParityError("reference_manifest_provenance_missing")
    if manifest_record.get("sha256") != binding["sha256"]:
        raise GatewayParityError("manifest_sha256_mismatch")
    if provenance.get("profile") != "canonical_http":
        raise GatewayParityError("reference_profile_not_canonical_http")
    if provenance.get("cohort_size") != EXPECTED_COHORT_SIZE:
        raise GatewayParityError("reference_cohort_size_mismatch")
    if provenance.get("threshold_exclusive") != THRESHOLD:
        raise GatewayParityError("reference_threshold_mismatch")
    return binding


def reference_row_by_case(
    cases: Sequence[PreparationCase], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    if len(cases) != EXPECTED_COHORT_SIZE or len(rows) != EXPECTED_COHORT_SIZE:
        raise GatewayParityError("gateway_parity_requires_100_cases_and_reference_rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for case, row in zip(cases, rows, strict=True):
        expected = row_id_hash(case.id)
        if row.get("id_sha256") != expected:
            raise GatewayParityError("reference_row_identity_mismatch")
        if len(row.get("token_ids", [])) != len(expected_labels(case.question)) + 1:
            raise GatewayParityError("reference_row_token_count_mismatch")
        by_id[case.id] = row
    return by_id


def gateway_result_row(
    *,
    case: PreparationCase,
    response: Mapping[str, Any],
    expected_model_version: str,
) -> dict[str, Any]:
    if response.get("model") != expected_model_version:
        raise GatewayParityError("gateway_response_model_version_mismatch")
    answers = response.get("answers")
    if not isinstance(answers, Mapping) or set(answers) != {case.question.qid}:
        raise GatewayParityError("gateway_response_answer_set_mismatch")
    answer = answers[case.question.qid]
    if not isinstance(answer, Mapping) or answer.get("type") != case.question.type:
        raise GatewayParityError("gateway_response_answer_type_mismatch")
    labels = expected_labels(case.question)
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, Mapping) or list(probabilities) != labels:
        raise GatewayParityError("gateway_response_probability_labels_mismatch")
    conditional = [
        _strict_probability(probabilities[label], field="gateway_probability") for label in labels
    ]
    total = math.fsum(conditional)
    if abs(total - 1.0) > 1e-6:
        raise GatewayParityError("gateway_conditional_probabilities_not_normalized")
    confidence = _strict_probability(answer.get("confidence"), field="gateway_confidence")
    if answer.get("confidence_semantics") != "normalized_peak_non_U":
        raise GatewayParityError("gateway_confidence_semantics_mismatch")
    if abs(confidence - max(conditional)) > 1e-9:
        raise GatewayParityError("gateway_confidence_not_conditional_peak")
    p_u = _strict_probability(answer.get("p_insufficient"), field="gateway_p_insufficient")
    usage = response.get("usage")
    backend_usage = response.get("backend_usage")
    if not isinstance(usage, Mapping) or not isinstance(backend_usage, Mapping):
        raise GatewayParityError("gateway_usage_missing")
    for field in ("input_tokens", "output_tokens"):
        if type(usage.get(field)) is not int or usage[field] < 0:
            raise GatewayParityError("gateway_usage_invalid")
        if type(backend_usage.get(field)) is not int or backend_usage[field] < 0:
            raise GatewayParityError("gateway_backend_usage_invalid")
    if usage["input_tokens"] != backend_usage["input_tokens"]:
        raise GatewayParityError("gateway_usage_input_mismatch")
    if usage["output_tokens"] != 0 or backend_usage["output_tokens"] != 1:
        raise GatewayParityError("gateway_usage_output_mismatch")
    semantic: dict[str, Any] = {"labels": labels}
    if case.question.type == "choice":
        argmax_label = labels[conditional.index(max(conditional))]
        if answer.get("choice") != argmax_label:
            raise GatewayParityError("gateway_choice_not_argmax")
        semantic["choice"] = answer["choice"]
    elif case.question.type == "score":
        score = _strict_float(answer.get("score"), field="gateway_score")
        expected_score = math.fsum(
            index * probability for index, probability in enumerate(conditional)
        )
        if not 0.0 <= score <= float(len(labels) - 1):
            raise GatewayParityError("gateway_score_outside_range")
        if abs(score - expected_score) > 1e-9:
            raise GatewayParityError("gateway_score_not_expected_value")
        expected_legend = {
            str(index): text for index, text in enumerate(case.question.score_descriptions or [])
        }
        if answer.get("legend") != expected_legend:
            raise GatewayParityError("gateway_score_legend_mismatch")
        semantic["score"] = score
    else:
        noul = _strict_probability(answer.get("noul"), field="gateway_noul")
        if abs(noul - conditional[0]) > 1e-9:
            raise GatewayParityError("gateway_noul_not_yes_probability")
        semantic["noul"] = noul
    return {
        "case_id_sha256": row_id_hash(case.id),
        "category": case.category,
        "primitive": case.question.type,
        "conditional": conditional,
        "p_insufficient": p_u,
        "semantic": semantic,
        "usage": dict(usage),
        "backend_usage": dict(backend_usage),
        "validation_status": response.get("validation_status"),
    }


def validate_loopback_url(value: str) -> str:
    url = httpx.URL(value)
    if url.scheme != "http" or url.host not in {"127.0.0.1", "localhost", "::1"}:
        raise GatewayParityError("gateway_url_must_be_http_loopback")
    if url.userinfo or url.raw_path not in (b"", b"/") or url.query:
        raise GatewayParityError("gateway_url_must_be_loopback_origin_only")
    return value.rstrip("/")


async def run_gateway_pass(
    *,
    cases: Sequence[PreparationCase],
    base_url: str,
    expected_model_version: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=base_url, timeout=30, transport=transport) as client:
        health_response = await client.get("/health")
        health_response.raise_for_status()
        health = validate_health(
            health_response.json(), expected_model_version=expected_model_version
        )
        for case in cases:
            response = await client.post("/v1/systemone", json=decision_request_from_case(case))
            response.raise_for_status()
            rows.append(
                gateway_result_row(
                    case=case,
                    response=response.json(),
                    expected_model_version=expected_model_version,
                )
            )
    return rows, health


def compare_gateway_to_reference(
    *,
    cases: Sequence[PreparationCase],
    reference_rows: Mapping[str, Mapping[str, Any]],
    gateway_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(gateway_rows) != len(cases):
        raise GatewayParityError("gateway_row_count_mismatch")
    maxima = {"conditional": 0.0, "p_insufficient": 0.0}
    details: list[dict[str, Any]] = []
    for case, actual in zip(cases, gateway_rows, strict=True):
        expected = reference_rows[case.id]
        if actual.get("case_id_sha256") != expected.get("id_sha256"):
            raise GatewayParityError("gateway_reference_case_identity_mismatch")
        expected_conditional = expected.get("conditional_without_u")
        if not isinstance(expected_conditional, list):
            raise GatewayParityError("reference_conditional_missing")
        actual_conditional = list(actual["conditional"])
        if len(actual_conditional) != len(expected_conditional):
            raise GatewayParityError("gateway_conditional_length_mismatch")
        expected_values = [
            _strict_probability(value, field="reference_conditional")
            for value in expected_conditional
        ]
        actual_values = [
            _strict_probability(value, field="gateway_conditional") for value in actual_conditional
        ]
        conditional_delta = max(
            abs(left - right) for left, right in zip(expected_values, actual_values, strict=True)
        )
        expected_p_u = _strict_probability(expected.get("p_U"), field="reference_p_insufficient")
        actual_p_u = _strict_probability(actual["p_insufficient"], field="gateway_p_insufficient")
        p_u_delta = abs(expected_p_u - actual_p_u)
        maxima["conditional"] = max(maxima["conditional"], conditional_delta)
        maxima["p_insufficient"] = max(maxima["p_insufficient"], p_u_delta)
        details.append(
            {
                "case_id_sha256": actual["case_id_sha256"],
                "category": actual["category"],
                "primitive": actual["primitive"],
                "conditional_max_delta": conditional_delta,
                "p_insufficient_delta": p_u_delta,
                "semantic": actual["semantic"],
                "usage": actual["usage"],
                "backend_usage": actual["backend_usage"],
            }
        )
    passed = all(value < THRESHOLD for value in maxima.values())
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "threshold_exclusive": THRESHOLD,
        "max_differences": maxima,
        "samples": len(gateway_rows),
        "details_sha256": sha256_json(details),
        "details": details,
        "raw_or_k_plus_one_validated_by_gateway": False,
    }


def compare_repeatability(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    identical = json.loads(json.dumps(first, sort_keys=True)) == json.loads(
        json.dumps(second, sort_keys=True)
    )
    return {"status": "passed" if identical else "failed", "passed": identical}


def report_base(action: str, started_at: str, started: float) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "action": action,
        "started_at": started_at,
        "finished_at": now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "environment": environment_summary(),
        "source": source_identity(),
        "seed": SEED,
        "cost": {"new_paid_resources": 0, "gpu_started_by_this_cli": False},
        "release_or_stage_acceptance": False,
    }


def write_report_once(path: Path, report: Mapping[str, Any]) -> None:
    refuse_existing(path)
    write_json(path, report)


def sanitized_error(error: BaseException) -> dict[str, str]:
    if isinstance(error, GatewayParityError):
        code = str(error) or "gateway_parity_error"
    elif isinstance(error, httpx.TimeoutException):
        code = "http_timeout"
    elif isinstance(error, httpx.HTTPStatusError):
        code = "http_status_error"
    elif isinstance(error, httpx.HTTPError):
        code = "http_error"
    elif isinstance(error, json.JSONDecodeError):
        code = "invalid_json"
    else:
        code = "unexpected_error"
    return {"type": type(error).__name__, "code": code}


def failure_report(
    path: Path, action: str, started_at: str, started: float, error: BaseException
) -> int:
    report = report_base(action, started_at, started)
    report.update(
        status="blocked",
        exit_code=2,
        error=sanitized_error(error),
        raw_or_k_plus_one_validated_by_gateway=False,
    )
    write_report_once(path, report)
    return 2


async def compare_gateway(args: argparse.Namespace) -> int:
    refuse_existing(args.output)
    started = time.monotonic()
    started_at = now_iso()
    try:
        reference_report, reference_sha, canonical_rows = load_reference_report(args.hf_vllm_report)
        gateway_record = json.loads(args.gateway_record.read_text())
        gateway = validate_gateway_record(
            gateway_record, reference_report=reference_report, reference_sha256=reference_sha
        )
        cases = read_manifest(args.manifest, allow_partial=False)
        manifest = validate_manifest_binding(
            manifest=args.manifest, cases=cases, reference_report=reference_report
        )
        gateway_url = validate_loopback_url(args.gateway_url)
        ref_rows = reference_row_by_case(cases, canonical_rows)
        first, health = await run_gateway_pass(
            cases=cases,
            base_url=gateway_url,
            expected_model_version=gateway["model_version"],
        )
        second, second_health = await run_gateway_pass(
            cases=cases,
            base_url=gateway_url,
            expected_model_version=gateway["model_version"],
        )
        repeat = compare_repeatability(first, second)
        comparison = compare_gateway_to_reference(
            cases=cases, reference_rows=ref_rows, gateway_rows=first
        )
        passed = bool(repeat["passed"] and comparison["passed"])
        report = report_base("compare-gateway", started_at, started)
        report.update(
            status="passed" if passed else "failed",
            exit_code=0 if passed else 2,
            gateway=gateway,
            manifest=manifest,
            health=health,
            second_health=second_health,
            hf_vllm_report={
                "sha256": reference_sha,
                "status": reference_report.get("status"),
                "parity": reference_report.get("parity"),
            },
            repeatability=repeat,
            comparison=comparison,
            raw_or_k_plus_one_validated_by_gateway=False,
            scope=(
                "Gateway parity checks conditional probabilities and p_insufficient only; "
                "raw/K+1 remain delegated to the referenced HF/vLLM report."
            ),
        )
        write_json(args.output, report)
        return int(report["exit_code"])
    except Exception as exc:
        return failure_report(args.output, "compare-gateway", started_at, started, exc)


def _synthetic_case(index: int, question_type: str) -> PreparationCase:
    from data.schema import Metadata, Prefix

    if question_type == "choice":
        question = Question(
            qid=f"q{index}",
            type="choice",
            instruction="synthetic choice instruction",
            options=["left", "right"],
        )
    elif question_type == "score":
        question = Question(
            qid=f"q{index}",
            type="score",
            instruction="synthetic score instruction",
            score_levels=3,
            score_descriptions=["zero", "one", "two"],
        )
    else:
        question = Question(qid=f"q{index}", type="noul", instruction="synthetic noul instruction")
    return PreparationCase(
        id=f"synthetic-{index}",
        category="text",
        prefix=Prefix(state=f"synthetic state {index}"),
        question=question,
        meta=Metadata(
            modality=["text"],
            source_dataset="synthetic",
            source_revision="none",
            source_sha256="0" * 64,
            license="project synthetic",
            usage="research",
            pool="dev",
            family_id=f"synthetic-{index}",
            data_version="test",
        ),
    )


def validate_case_request_mapping(case: PreparationCase) -> dict[str, Any]:
    payload = decision_request_from_case(case)
    request = DecisionRequest.model_validate(payload)
    if request.state != case.prefix.state:
        raise GatewayParityError("gateway_payload_state_mismatch")
    if [media.model_dump() for media in request.media] != [
        media.model_dump() for media in case.prefix.media
    ]:
        raise GatewayParityError("gateway_payload_media_mismatch")
    if request.video_frames != case.video_frames:
        raise GatewayParityError("gateway_payload_video_frames_mismatch")
    if set(request.questions) != {case.question.qid}:
        raise GatewayParityError("gateway_payload_question_set_mismatch")
    item = request.questions[case.question.qid]
    if not isinstance(item, InputQuestion):
        raise GatewayParityError("gateway_payload_question_type_mismatch")
    question, labels = core_question(case.question.qid, item)
    if question.model_dump() != case.question.model_dump():
        raise GatewayParityError("gateway_core_question_mismatch")
    expected = expected_labels(case.question)
    if labels != expected:
        raise GatewayParityError("gateway_core_labels_mismatch")
    return {
        "id_sha256": row_id_hash(case.id),
        "case_fingerprint": case_fingerprint(case),
        "category": case.category,
        "primitive": case.question.type,
        "labels": labels,
        "answer_count_including_u": len(labels) + 1,
        "payload_sha256": sha256_json(payload),
    }


def validate_manifest_request_mapping(
    *,
    cases: Sequence[PreparationCase],
    canonical_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = [validate_case_request_mapping(case) for case in cases]
    if canonical_report is not None:
        if (
            canonical_report.get("status") != "passed"
            or canonical_report.get("complete_100_cases") is not True
        ):
            raise GatewayParityError("canonical_summary_report_not_passed_100")
        summaries = canonical_report.get("cases")
        if not isinstance(summaries, list) or len(summaries) != len(cases):
            raise GatewayParityError("canonical_summary_case_count_mismatch")
        by_id = {
            summary.get("id"): summary for summary in summaries if isinstance(summary, Mapping)
        }
        for case, row in zip(cases, rows, strict=True):
            summary = by_id.get(case.id)
            if not isinstance(summary, Mapping):
                raise GatewayParityError("canonical_summary_case_missing")
            if (
                summary.get("category") != case.category
                or summary.get("primitive") != case.question.type
            ):
                raise GatewayParityError("canonical_summary_case_contract_mismatch")
            if summary.get("answer_count_including_u") != row["answer_count_including_u"]:
                raise GatewayParityError("canonical_summary_answer_count_mismatch")
            provenance = summary.get("provenance")
            if provenance != case.meta.model_dump():
                raise GatewayParityError("canonical_summary_provenance_mismatch")
    return {
        "status": "passed",
        "passed": True,
        "samples": len(rows),
        "request_mapping_sha256": sha256_json(rows),
        "primitive_counts": {
            primitive: sum(row["primitive"] == primitive for row in rows)
            for primitive in ("choice", "score", "noul")
        },
        "category_counts": {
            category: sum(row["category"] == category for row in rows)
            for category in ("text", "image", "screenshot", "audio", "video")
        },
        "details_sha256": sha256_json(rows),
    }


def synthetic_cases() -> list[PreparationCase]:
    return [_synthetic_case(0, "choice"), _synthetic_case(1, "score"), _synthetic_case(2, "noul")]


def synthetic_reference_rows(cases: Sequence[PreparationCase]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        labels = expected_labels(case.question)
        if len(labels) == 2:
            conditional = [0.7, 0.3]
        else:
            conditional = [0.2, 0.3, 0.5]
        rows[case.id] = {
            "id_sha256": row_id_hash(case.id),
            "conditional_without_u": conditional,
            "p_U": 0.125,
        }
    return rows


def synthetic_response(case: PreparationCase) -> dict[str, Any]:
    labels = expected_labels(case.question)
    conditional = synthetic_reference_rows([case])[case.id]["conditional_without_u"]
    probabilities = dict(zip(labels, conditional, strict=True))
    answer: dict[str, Any] = {
        "type": case.question.type,
        "probabilities": probabilities,
        "confidence": max(conditional),
        "p_insufficient": 0.125,
        "calibration": "unfitted",
        "confidence_semantics": "normalized_peak_non_U",
    }
    if case.question.type == "choice":
        answer["choice"] = labels[conditional.index(max(conditional))]
    elif case.question.type == "score":
        answer["score"] = sum(index * value for index, value in enumerate(conditional))
        answer["legend"] = {
            str(index): text for index, text in enumerate(case.question.score_descriptions or [])
        }
    else:
        answer["noul"] = probabilities["yes"]
    return {
        "model": "synthetic-gateway-v1",
        "answers": {case.question.qid: answer},
        "usage": {"input_tokens": 10, "output_tokens": 0},
        "backend_usage": {"input_tokens": 10, "output_tokens": 1},
        "validation_status": "GPU parity and calibration pending",
    }


def manifest_control(args: argparse.Namespace) -> int:
    refuse_existing(args.output)
    started = time.monotonic()
    started_at = now_iso()
    try:
        cases = read_manifest(args.manifest, allow_partial=False)
        canonical_report = (
            json.loads(args.canonical_report.read_text()) if args.canonical_report else None
        )
        mapping = validate_manifest_request_mapping(cases=cases, canonical_report=canonical_report)
        report = report_base("manifest-control", started_at, started)
        report.update(
            status="passed",
            exit_code=0,
            manifest=manifest_binding(args.manifest, cases),
            mapping=mapping,
            synthetic_non_gpu_acceptance=False,
            raw_or_k_plus_one_validated_by_gateway=False,
            scope=(
                "CPU-only 100-case manifest to gateway payload and core_question mapping; "
                "does not contact a model backend or prove S0 acceptance."
            ),
        )
        write_json(args.output, report)
        return 0
    except Exception as exc:
        return failure_report(args.output, "manifest-control", started_at, started, exc)


async def synthetic_control(args: argparse.Namespace) -> int:
    refuse_existing(args.output)
    started = time.monotonic()
    started_at = now_iso()
    cases = synthetic_cases()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "inference_configured": True,
                    "model_version": "synthetic-gateway-v1",
                    "mode": "canonical-chat",
                    "model_quality_verified": False,
                },
            )
        body = json.loads(request.content.decode("utf-8"))
        qid = next(iter(body["questions"]))
        index = int(qid.removeprefix("q"))
        return httpx.Response(200, json=synthetic_response(cases[index]))

    rows_a, health = await run_gateway_pass(
        cases=cases,
        base_url="http://synthetic.local",
        expected_model_version="synthetic-gateway-v1",
        transport=httpx.MockTransport(handler),
    )
    rows_b, _ = await run_gateway_pass(
        cases=cases,
        base_url="http://synthetic.local",
        expected_model_version="synthetic-gateway-v1",
        transport=httpx.MockTransport(handler),
    )
    comparison = compare_gateway_to_reference(
        cases=cases, reference_rows=synthetic_reference_rows(cases), gateway_rows=rows_a
    )
    repeat = compare_repeatability(rows_a, rows_b)
    report = report_base("synthetic-control", started_at, started)
    report.update(
        status="passed" if comparison["passed"] and repeat["passed"] else "failed",
        exit_code=0 if comparison["passed"] and repeat["passed"] else 2,
        health=health,
        comparison=comparison,
        repeatability=repeat,
        synthetic_non_gpu_acceptance=False,
        raw_or_k_plus_one_validated_by_gateway=False,
        scope=(
            "Synthetic CPU transport only; validates gateway parity control flow, "
            "not real HF/vLLM/gateway acceptance."
        ),
    )
    write_json(args.output, report)
    return int(report["exit_code"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    compare = sub.add_parser("compare-gateway")
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--hf-vllm-report", type=Path, required=True)
    compare.add_argument("--gateway-record", type=Path, required=True)
    compare.add_argument("--gateway-url", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=lambda args: asyncio.run(compare_gateway(args)))
    manifest = sub.add_parser("manifest-control")
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--canonical-report", type=Path)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(func=manifest_control)
    synthetic = sub.add_parser("synthetic-control")
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.set_defaults(func=lambda args: asyncio.run(synthetic_control(args)))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ReportExistsError as exc:
        print(f"s0_gateway_parity: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
