from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from data.schema import Metadata, Prefix, Question
from scripts import s0_gateway_parity as gp
from train.multimodal import PreparationCase


def _meta(index: int) -> Metadata:
    return Metadata(
        modality=["text"],
        source_dataset="synthetic",
        source_revision="none",
        source_sha256="0" * 64,
        license="project synthetic",
        usage="research",
        pool="dev",
        family_id=f"family-{index}",
        data_version="test",
    )


def _case(index: int, question: Question) -> PreparationCase:
    return PreparationCase(
        id=f"case-{index}",
        category="text",
        prefix=Prefix(state=f"state {index}"),
        question=question,
        meta=_meta(index),
    )


def test_input_question_mapping_preserves_choice_and_score_order() -> None:
    choice = Question(
        qid="choice",
        type="choice",
        instruction="choose",
        options=["first", "second", "third"],
    )
    score = Question(
        qid="score",
        type="score",
        instruction="score",
        score_levels=3,
        score_descriptions=["zero", "one", "two"],
    )
    noul = Question(qid="noul", type="noul", instruction="yes or no")

    assert gp.input_question_from_question(choice)["criteria"] == {
        "A": "first",
        "B": "second",
        "C": "third",
    }
    assert gp.input_question_from_question(score)["criteria"] == ["zero", "one", "two"]
    assert "criteria" not in gp.input_question_from_question(noul)
    assert gp.expected_labels(choice) == ["A", "B", "C"]
    assert gp.expected_labels(score) == ["0", "1", "2"]
    assert gp.expected_labels(noul) == ["yes", "no"]


def test_health_and_gateway_record_reject_wrong_version_or_unattested() -> None:
    with pytest.raises(gp.GatewayParityError, match="canonical_chat|canonical"):
        gp.validate_health(
            {
                "status": "ok",
                "mode": "legacy-text-completions",
                "inference_configured": True,
                "model_version": "v1",
            },
            expected_model_version="v1",
        )
    health = gp.validate_health(
        {
            "status": "ok",
            "mode": "canonical-chat",
            "inference_configured": True,
            "model_version": "v1",
        },
        expected_model_version="v1",
    )
    assert health["mode"] == "canonical-chat"

    reference = {"provenance": {"fixed": True}, "server": {"record_sha256": "abc"}}
    with pytest.raises(gp.GatewayParityError, match="operator"):
        gp.validate_gateway_record({}, reference_report=reference, reference_sha256="ref")
    record = {
        "operator_attested": True,
        "s0_hf_vllm_report_sha256": "ref",
        "provenance": {"fixed": True},
        "upstream_server_record_sha256": "abc",
        "model_version": "v1",
        "gateway_launch_command": "uvicorn serve.gateway.app:app",
    }
    validated = gp.validate_gateway_record(
        record, reference_report=reference, reference_sha256="ref"
    )
    assert validated["remote_attestation"] is False
    assert validated["model_version"] == "v1"


def test_mock_gateway_pass_compares_conditional_and_u_without_raw_claims() -> None:
    cases = [
        _case(
            0,
            Question(qid="choice", type="choice", instruction="choose", options=["left", "right"]),
        ),
        _case(
            1,
            Question(
                qid="score",
                type="score",
                instruction="score",
                score_levels=3,
                score_descriptions=["zero", "one", "two"],
            ),
        ),
        _case(2, Question(qid="noul", type="noul", instruction="yes?")),
    ]
    reference = {
        cases[0].id: {
            "id_sha256": gp.row_id_hash(cases[0].id),
            "conditional_without_u": [0.7, 0.3],
            "p_U": 0.125,
        },
        cases[1].id: {
            "id_sha256": gp.row_id_hash(cases[1].id),
            "conditional_without_u": [0.2, 0.3, 0.5],
            "p_U": 0.125,
        },
        cases[2].id: {
            "id_sha256": gp.row_id_hash(cases[2].id),
            "conditional_without_u": [0.7, 0.3],
            "p_U": 0.125,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "mode": "canonical-chat",
                    "inference_configured": True,
                    "model_version": "gateway-v1",
                },
            )
        body = json.loads(request.content.decode())
        qid = next(iter(body["questions"]))
        index = {case.question.qid: i for i, case in enumerate(cases)}[qid]
        case = cases[index]
        conditional = reference[case.id]["conditional_without_u"]
        labels = gp.expected_labels(case.question)
        probabilities = dict(zip(labels, conditional, strict=True))
        answer = {
            "type": case.question.type,
            "probabilities": probabilities,
            "confidence": (max(conditional) - 1 / len(labels)) / (1 - 1 / len(labels)),
            "p_insufficient": 0.125,
            "calibration": "unfitted",
            "confidence_semantics": "normalized_peak_non_U",
        }
        if case.question.type == "choice":
            answer["choice"] = labels[conditional.index(max(conditional))]
        elif case.question.type == "score":
            answer["score"] = sum(i * p for i, p in enumerate(conditional))
            answer["legend"] = {
                str(i): text for i, text in enumerate(case.question.score_descriptions or [])
            }
        else:
            answer["noul"] = probabilities["yes"]
        return httpx.Response(
            200,
            json={
                "model": "gateway-v1",
                "answers": {qid: answer},
                "usage": {"input_tokens": 5, "output_tokens": 0},
                "backend_usage": {"input_tokens": 5, "output_tokens": 1},
                "validation_status": "GPU parity and calibration pending",
            },
        )

    rows_a, health = asyncio.run(
        gp.run_gateway_pass(
            cases=cases,
            base_url="http://gateway.local",
            expected_model_version="gateway-v1",
            transport=httpx.MockTransport(handler),
        )
    )
    rows_b, _ = asyncio.run(
        gp.run_gateway_pass(
            cases=cases,
            base_url="http://gateway.local",
            expected_model_version="gateway-v1",
            transport=httpx.MockTransport(handler),
        )
    )
    comparison = gp.compare_gateway_to_reference(
        cases=cases, reference_rows=reference, gateway_rows=rows_a
    )
    repeat = gp.compare_repeatability(rows_a, rows_b)

    assert health["inference_configured"] is True
    assert comparison["passed"] is True
    assert comparison["raw_or_k_plus_one_validated_by_gateway"] is False
    assert repeat["passed"] is True
    assert {row["primitive"] for row in rows_a} == {"choice", "score", "noul"}


def test_gateway_compare_rejects_bad_nan_and_version() -> None:
    case = _case(0, Question(qid="choice", type="choice", instruction="choose", options=["a", "b"]))
    response = {
        "model": "wrong-version",
        "answers": {
            "choice": {
                "type": "choice",
                "probabilities": {"A": 0.5, "B": 0.5},
                "p_insufficient": 0.0,
            }
        },
        "usage": {"input_tokens": 1, "output_tokens": 0},
        "backend_usage": {"input_tokens": 1, "output_tokens": 1},
    }
    with pytest.raises(gp.GatewayParityError, match="model_version"):
        gp.gateway_result_row(case=case, response=response, expected_model_version="gateway-v1")
    response["model"] = "gateway-v1"
    response["answers"]["choice"]["probabilities"]["A"] = float("nan")
    with pytest.raises(gp.GatewayParityError, match="not_finite"):
        gp.gateway_result_row(case=case, response=response, expected_model_version="gateway-v1")

    gateway_row = {
        "case_id_sha256": gp.row_id_hash(case.id),
        "category": "text",
        "primitive": "choice",
        "conditional": [0.5, 0.5],
        "p_insufficient": 0.0,
        "semantic": {"labels": ["A", "B"], "choice": "A"},
        "usage": {"input_tokens": 1, "output_tokens": 0},
        "backend_usage": {"input_tokens": 1, "output_tokens": 1},
    }
    reference = {
        case.id: {
            "id_sha256": gp.row_id_hash(case.id),
            "conditional_without_u": [float("nan"), 0.5],
            "p_U": 0.0,
        }
    }
    with pytest.raises(gp.GatewayParityError, match="reference_conditional_not_finite"):
        gp.compare_gateway_to_reference(
            cases=[case], reference_rows=reference, gateway_rows=[gateway_row]
        )


def test_synthetic_control_writes_non_acceptance_report_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    assert gp.main(["synthetic-control", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["synthetic_non_gpu_acceptance"] is False
    assert report["raw_or_k_plus_one_validated_by_gateway"] is False
    assert report["release_or_stage_acceptance"] is False
    assert gp.main(["synthetic-control", "--output", str(output)]) == 2


def _full_cases() -> list[PreparationCase]:
    return gp.read_manifest(
        Path("tests/fixtures/s0-prepare-public-001/manifest.json"), allow_partial=False
    )


def _raw_row_for_case(case: PreparationCase) -> dict[str, object]:
    from scripts.s0_parity import make_row, row_to_json

    token_count = len(gp.expected_labels(case.question)) + 1
    token_ids = list(range(1000, 1000 + token_count))
    row = make_row(
        id_sha256=gp.row_id_hash(case.id),
        input_sha256=gp.sha256_json({"input": case.id}),
        media_sha256=[gp.sha256_json({"media": case.id})],
        token_ids=token_ids,
        raw_logprobs=[-10.0 - index for index in range(token_count)],
    )
    return row_to_json(row)


def _reference_report(cases: list[PreparationCase], manifest_path: Path) -> dict[str, object]:
    maxima = {"raw_probabilities": 0.0, "joint": 0.0, "conditional": 0.0, "p_U": 0.0}
    passed = {
        "label": "hf_reference_vs_vllm",
        "status": "passed",
        "passed": True,
        "samples": 100,
        "threshold_exclusive": 0.001,
        "max_differences": maxima,
    }
    return {
        "action": "compare-vllm",
        "status": "passed",
        "provenance": {
            "profile": "canonical_http",
            "cohort_size": 100,
            "threshold_exclusive": 0.001,
            "manifest": {"sha256": gp.sha256_file(manifest_path)},
        },
        "server": {"record_sha256": "server-sha"},
        "reference_validation": {
            **passed,
            "label": "reference_rows_match_current_canonical_http_inputs",
        },
        "vllm_repeatability": {**passed, "label": "vllm_repeatability"},
        "parity": passed,
        "rows": [_raw_row_for_case(case) for case in cases],
    }


def test_reference_report_requires_strict_rows_and_passed_subreports(tmp_path: Path) -> None:
    manifest = Path("tests/fixtures/s0-prepare-public-001/manifest.json")
    cases = _full_cases()
    report = _reference_report(cases, manifest)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(report))
    loaded, sha, rows = gp.load_reference_report(path)
    assert loaded["status"] == "passed"
    assert len(sha) == 64
    assert len(rows) == 100

    tampered = json.loads(json.dumps(report))
    tampered["rows"][0]["raw_probabilities"][0] = 0.25
    path.write_text(json.dumps(tampered))
    with pytest.raises(gp.GatewayParityError, match="strict|derived|canonical|validation"):
        gp.load_reference_report(path)

    not_repeated = json.loads(json.dumps(report))
    not_repeated["vllm_repeatability"]["passed"] = False
    path.write_text(json.dumps(not_repeated))
    with pytest.raises(gp.GatewayParityError, match="vllm_repeatability"):
        gp.load_reference_report(path)


def test_manifest_binding_rejects_changed_manifest_and_token_count() -> None:
    manifest = Path("tests/fixtures/s0-prepare-public-001/manifest.json")
    cases = _full_cases()
    report = _reference_report(cases, manifest)
    binding = gp.validate_manifest_binding(manifest=manifest, cases=cases, reference_report=report)
    assert binding["sha256"] == report["provenance"]["manifest"]["sha256"]

    wrong = json.loads(json.dumps(report))
    wrong["provenance"]["manifest"]["sha256"] = "0" * 64
    with pytest.raises(gp.GatewayParityError, match="manifest_sha256"):
        gp.validate_manifest_binding(manifest=manifest, cases=cases, reference_report=wrong)

    rows = report["rows"]
    rows[0]["token_ids"].append(9999)
    with pytest.raises(gp.GatewayParityError, match="token_count"):
        gp.reference_row_by_case(cases, rows)


def test_gateway_result_row_rejects_semantic_and_usage_mismatches() -> None:
    choice = _case(
        0, Question(qid="choice", type="choice", instruction="choose", options=["a", "b"])
    )
    response = {
        "model": "gateway-v1",
        "answers": {
            "choice": {
                "type": "choice",
                "probabilities": {"A": 0.8, "B": 0.2},
                "confidence": 0.6,
                "confidence_semantics": "normalized_peak_non_U",
                "p_insufficient": 0.1,
                "choice": "B",
            }
        },
        "usage": {"input_tokens": 3, "output_tokens": 0},
        "backend_usage": {"input_tokens": 3, "output_tokens": 1},
    }
    with pytest.raises(gp.GatewayParityError, match="argmax"):
        gp.gateway_result_row(case=choice, response=response, expected_model_version="gateway-v1")

    score = _case(
        1,
        Question(
            qid="score",
            type="score",
            instruction="score",
            score_levels=3,
            score_descriptions=["zero", "one", "two"],
        ),
    )
    response = {
        "model": "gateway-v1",
        "answers": {
            "score": {
                "type": "score",
                "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5},
                "confidence": 0.25,
                "confidence_semantics": "normalized_peak_non_U",
                "p_insufficient": 0.1,
                "score": 2.0,
                "legend": {"0": "zero", "1": "one", "2": "two"},
            }
        },
        "usage": {"input_tokens": 3, "output_tokens": 0},
        "backend_usage": {"input_tokens": 3, "output_tokens": 1},
    }
    with pytest.raises(gp.GatewayParityError, match="expected_value"):
        gp.gateway_result_row(case=score, response=response, expected_model_version="gateway-v1")

    response["answers"]["score"]["score"] = 1.3
    response["backend_usage"]["output_tokens"] = 2
    with pytest.raises(gp.GatewayParityError, match="usage_output"):
        gp.gateway_result_row(case=score, response=response, expected_model_version="gateway-v1")


def test_manifest_control_real_100_payload_mapping_against_canonical_summary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest-control.json"
    assert (
        gp.main(
            [
                "manifest-control",
                "--manifest",
                "tests/fixtures/s0-prepare-public-001/manifest.json",
                "--canonical-report",
                "reports/s0-canonical-http-cpu-003.json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["mapping"]["samples"] == 100
    assert report["mapping"]["primitive_counts"] == {"choice": 20, "score": 20, "noul": 60}
    assert report["synthetic_non_gpu_acceptance"] is False


def test_failure_report_sanitizes_external_error_text(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"
    secret = "https://token.example/private?secret=abc"
    error = httpx.HTTPStatusError(
        f"bad response {secret}",
        request=httpx.Request("GET", secret),
        response=httpx.Response(500, text="secret body"),
    )
    assert gp.failure_report(output, "compare-gateway", gp.now_iso(), 0.0, error) == 2
    report = json.loads(output.read_text())
    assert report["error"] == {"type": "HTTPStatusError", "code": "http_status_error"}
    assert secret not in output.read_text()
    assert "secret body" not in output.read_text()


@pytest.mark.parametrize(
    "probabilities,confidence,valid",
    [
        ([0.8, 0.2], 0.6, True),
        ([0.8, 0.2], 0.8, False),
        ([0.5, 0.5], 0.0, True),
        ([0.5, 0.5], 0.5, False),
    ],
)
def test_gateway_confidence_uses_chance_adjusted_peak(probabilities, confidence, valid):
    case = _case(0, Question(qid="choice", type="choice", instruction="choose", options=["a", "b"]))
    response = gp.synthetic_response(case)
    answer = response["answers"]["choice"]
    answer["probabilities"] = dict(zip(["A", "B"], probabilities, strict=True))
    answer["confidence"] = confidence
    if valid:
        result = gp.gateway_result_row(
            case=case, response=response, expected_model_version=response["model"]
        )
        assert result["conditional"] == probabilities
    else:
        with pytest.raises(gp.GatewayParityError, match="confidence"):
            gp.gateway_result_row(
                case=case, response=response, expected_model_version=response["model"]
            )
