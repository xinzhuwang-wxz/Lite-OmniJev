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
            "confidence": max(conditional),
            "p_insufficient": 0.125,
            "calibration": "unfitted",
            "confidence_semantics": "normalized_peak_non_U",
        }
        if case.question.type == "choice":
            answer["choice"] = labels[conditional.index(max(conditional))]
        elif case.question.type == "score":
            answer["score"] = sum(i * p for i, p in enumerate(conditional))
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
