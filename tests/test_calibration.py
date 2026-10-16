"""Temperature fitting is CPU preparation, not a model-quality or stage acceptance test."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from train.calibration import (
    Binding,
    CalibrationInput,
    TemperatureArtifact,
    apply_temperature,
    fit_temperature,
    load_temperature,
    save_temperature,
)
from train.readout import read_probabilities


@pytest.fixture
def payload():
    return {
        "schema_version": 1,
        "pool": "calibration",
        "identity": {
            "base_revision": "a" * 40,
            "adapter_weights_sha256": "b" * 64,
            "official_template_sha256": "c" * 64,
            "answer_token_map_sha256": "d" * 64,
            "raw_logits_receipt_sha256": "9" * 64,
            "cohort": "video",
            "primitive": "choice",
            "seed": 42,
            "code_revision": "e" * 40,
        },
        "rows": [
            {
                "pool": "calibration",
                "logits_kind": "raw_logits",
                "complete_legal_logits": True,
                "options": ["A", "B"],
                "logits": {"A": 4.0, "B": 0.0, "U": -4.0},
                "target": target,
            }
            for target in ("A", "A", "A", "B")
        ],
    }


def fitted(payload):
    data = CalibrationInput.model_validate(payload)
    return fit_temperature(data, "f" * 64)


def test_fit_improves_joint_nll_and_replays(payload, tmp_path):
    artifact = fitted(payload)
    assert artifact.nll_after < artifact.nll_before
    assert 1 < artifact.temperature <= 20
    assert artifact == fitted(payload)
    assert artifact.stage_acceptance == "not_evaluated"
    path = tmp_path / "temperature.json"
    save_temperature(artifact, path)
    loaded = load_temperature(path, artifact.binding)
    assert loaded == artifact
    row = payload["rows"][0]
    result = apply_temperature(loaded, artifact.binding, row["logits"], row["options"])
    assert result == read_probabilities(
        row["logits"], row["options"], temperature=artifact.temperature
    )
    assert sum(result.probabilities.values()) == pytest.approx(1)
    assert sum(result.joint_probabilities.values()) == pytest.approx(1)
    assert result.p_insufficient == result.joint_probabilities["U"]
    assert result.probabilities["A"] == pytest.approx(
        result.joint_probabilities["A"] / (1 - result.p_insufficient)
    )
    with pytest.raises(FileExistsError):
        save_temperature(artifact, path)


def test_temperature_one_identity_and_u_target(payload):
    for row in payload["rows"]:
        row["logits"] = {"A": 0.0, "B": 0.0, "U": 0.0}
        row["target"] = "U"
    artifact = fitted(payload)
    assert artifact.temperature == 1.0
    row = payload["rows"][0]
    assert apply_temperature(artifact, artifact.binding, row["logits"], row["options"]) == (
        read_probabilities(row["logits"], row["options"])
    )
    assert artifact.nll_after == pytest.approx(1.0986122886681098)


@pytest.mark.parametrize(
    "key",
    [
        "base_revision",
        "adapter_weights_sha256",
        "official_template_sha256",
        "answer_token_map_sha256",
        "raw_logits_receipt_sha256",
        "cohort",
        "primitive",
        "calibration_data_sha256",
        "seed",
        "code_revision",
    ],
)
def test_each_binding_field_mismatch_rejected(payload, tmp_path, key):
    artifact = fitted(payload)
    path = tmp_path / "temperature.json"
    save_temperature(artifact, path)
    values = artifact.binding.model_dump()
    values[key] = {
        "cohort": "text",
        "primitive": "score",
        "seed": 43,
    }.get(key, "1" * len(values[key]) if isinstance(values[key], str) else None)
    expected = Binding.model_validate(values)
    with pytest.raises(ValueError, match="binding mismatch"):
        load_temperature(path, expected)
    with pytest.raises(ValueError, match="binding mismatch"):
        apply_temperature(artifact, expected, {"A": 1.0, "B": 2.0, "U": 0.0}, ["A", "B"])


@pytest.mark.parametrize("pool", ["train", "dev", "holdout", "unknown"])
@pytest.mark.parametrize("where", ["dataset", "row"])
def test_fit_rejects_non_calibration_pools(payload, pool, where):
    if where == "dataset":
        payload["pool"] = pool
    else:
        payload["rows"][0]["pool"] = pool
    with pytest.raises(ValidationError):
        CalibrationInput.model_validate(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_u",
        "extra_token",
        "single_option",
        "duplicate_option",
        "missing_option",
        "nan",
        "inf",
        "bool_logit",
        "logprobs",
        "sampled",
        "incomplete",
        "integer_declaration",
        "target_unknown",
        "unknown_field",
        "missing_pool",
        "wrong_cohort",
        "invalid_hash",
    ],
)
def test_reject_invalid_fit_input(payload, mutation):
    row = payload["rows"][0]
    if mutation == "missing_u":
        del row["logits"]["U"]
    elif mutation == "extra_token":
        row["logits"]["C"] = 0.0
    elif mutation == "single_option":
        row["options"] = ["A"]
    elif mutation == "duplicate_option":
        row["options"] = ["A", "A"]
    elif mutation == "missing_option":
        del row["logits"]["B"]
    elif mutation in ("nan", "inf", "bool_logit"):
        row["logits"]["A"] = {"nan": float("nan"), "inf": float("inf"), "bool_logit": True}[
            mutation
        ]
    elif mutation in ("logprobs", "sampled"):
        row["logits_kind"] = mutation
    elif mutation == "integer_declaration":
        row["complete_legal_logits"] = 1
    elif mutation == "incomplete":
        row["complete_legal_logits"] = False
    elif mutation == "target_unknown":
        row["target"] = "C"
    elif mutation == "unknown_field":
        row["top_k"] = 3
    elif mutation == "missing_pool":
        del row["pool"]
    elif mutation == "wrong_cohort":
        payload["identity"]["cohort"] = "unknown"
    else:
        payload["identity"]["adapter_weights_sha256"] = "unbound"
    with pytest.raises(ValidationError):
        CalibrationInput.model_validate(payload)


@pytest.mark.parametrize("primitive", ["choice", "score", "noul"])
def test_primitives_and_option_limits(payload, primitive):
    payload["identity"]["primitive"] = primitive
    if primitive == "score":
        for row in payload["rows"]:
            row.update(options=["0", "1"], logits={"0": 4.0, "1": 0.0, "U": -4.0})
            row["target"] = "0"
    if primitive == "noul":
        for row in payload["rows"]:
            row.update(options=["yes", "no"], logits={"yes": 4.0, "no": 0.0, "U": -4.0})
            row["target"] = "yes"
    assert fitted(payload).binding.primitive == primitive
    if primitive == "score":
        payload["rows"][0].update(
            options=list("ABCDEFGHIJK"), logits={key: 0.0 for key in "ABCDEFGHIJKU"}
        )
    elif primitive == "noul":
        payload["rows"][0]["options"] = ["no", "yes"]
    else:
        payload["rows"][0]["options"] = list("ABCDEFGHIJKLMNOPQRSTV")
    with pytest.raises(ValidationError):
        CalibrationInput.model_validate(payload)


@pytest.mark.parametrize("temperature", [0.0, 0.049, 20.01, float("inf"), float("nan")])
def test_invalid_artifact_temperature_rejected(payload, temperature):
    values = fitted(payload).model_dump()
    values["temperature"] = temperature
    with pytest.raises(ValidationError):
        TemperatureArtifact.model_validate(values)


def test_cli_fit_verify_and_fail_closed(payload, tmp_path):
    input_path = tmp_path / "synthetic-calibration.json"
    output = tmp_path / "temperature.json"
    binding = tmp_path / "expected-binding.json"
    input_path.write_text(json.dumps(payload))
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "scripts.fit_calibration", *map(str, args)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    result = cli("fit", "--input", input_path, "--output", output)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["stage_acceptance"] == "not_evaluated"
    artifact = TemperatureArtifact.model_validate_json(output.read_text())
    assert (
        artifact.binding.calibration_data_sha256
        == hashlib.sha256(input_path.read_bytes()).hexdigest()
    )
    binding.write_text(artifact.binding.model_dump_json())
    assert cli("verify", "--artifact", output, "--expected-binding", binding).returncode == 0
    wrong_binding = artifact.binding.model_dump()
    wrong_binding["adapter_weights_sha256"] = "0" * 64
    binding.write_text(json.dumps(wrong_binding))
    assert cli("verify", "--artifact", output, "--expected-binding", binding).returncode == 1
    original = output.read_bytes()
    assert cli("fit", "--input", input_path, "--output", output).returncode == 1
    assert output.read_bytes() == original
    wrong = copy.deepcopy(payload)
    wrong["pool"] = "holdout"
    input_path.write_text(json.dumps(wrong))
    rejected_output = tmp_path / "rejected.json"
    result = cli("fit", "--input", input_path, "--output", rejected_output)
    assert result.returncode == 1
    assert not rejected_output.exists()
    assert "holdout" not in result.stdout


def test_application_rejects_wrong_primitive_tokens(payload):
    payload["identity"]["primitive"] = "noul"
    for row in payload["rows"]:
        row.update(options=["yes", "no"], logits={"yes": 1.0, "no": 0.0, "U": -1.0})
        row["target"] = "yes"
    artifact = fitted(payload)
    with pytest.raises(ValueError, match="canonical primitive"):
        apply_temperature(artifact, artifact.binding, {"A": 1.0, "B": 0.0, "U": -1.0}, ["A", "B"])


def test_artifact_rejects_changed_frozen_bounds(payload):
    values = fitted(payload).model_dump()
    values["temperature_max"] = 30.0
    with pytest.raises(ValidationError, match="bounds differ"):
        TemperatureArtifact.model_validate(values)


@pytest.mark.parametrize("target, expected_t", [("A", 0.05), ("B", 20.0)])
def test_fit_retains_bounded_optima(payload, target, expected_t):
    for row in payload["rows"]:
        row["target"] = target
    artifact = fitted(payload)
    assert artifact.temperature == expected_t
    assert artifact.nll_after <= artifact.nll_before
