import json
import math
import subprocess
import sys

import pytest
from pydantic import ValidationError

from data.schema import Prefix, Question, Sample, answer_tokens, validate_pool_isolation
from data.synth import generate_samples
from eval.gate import evaluate_gate
from eval.metrics import evaluate
from train.readout import read_probabilities
from train.template import check_answer_tokens, render_payload, render_prompt


def question(**changes):
    return Question.model_validate(
        {
            "qid": "q",
            "type": "choice",
            "instruction": "Choose.",
            "options": ["red", "blue"],
            **changes,
        }
    )


def test_generator_reproducible_and_program_truth():
    samples = generate_samples(8, 42)
    assert samples == generate_samples(8, 42)
    assert samples != generate_samples(8, 43)
    validate_pool_isolation(samples)
    for sample in samples:
        assert Sample.model_validate_json(sample.model_dump_json()) == sample
        left, right = int(sample.prefix.state.split()[2]), int(sample.prefix.state.split()[6])
        assert sample.questions[0].target.value == (
            "A" if left > right else "B" if right > left else "C"
        )
        assert sample.questions[2].target.value == int(left > 0) + int(right > 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"options": ["only"]},
        {"options": ["same", "same"]},
        {"options": [" ", "ok"]},
        {"options": [str(i) for i in range(21)]},
        {"unexpected": 1},
        {"target": {"kind": "hard", "value": "U", "source": "program"}},
        {"target": {"kind": "hard", "value": 0, "source": "program"}},
        {"target": {"kind": "soft", "probs": [0.4, 0.4], "source": "teacher"}},
        {"target": {"kind": "soft", "probs": [math.nan, 0.5], "source": "teacher"}},
        {"target": {"kind": "soft", "probs": [1.0, -1.0, 1.0], "source": "teacher"}},
        {"target": {"kind": "soft", "probs": [0.2, 0.3, 0.5], "source": "teacher"}},
        {"target": {"kind": "insufficient", "value": "A"}},
    ],
)
def test_invalid_question_rejected(changes):
    with pytest.raises(ValidationError):
        question(**changes)


def test_score_and_noul_contracts():
    score = question(
        type="score",
        options=None,
        score_levels=10,
        target={"kind": "hard", "value": 9, "source": "program"},
    )
    assert answer_tokens(score) == tuple(str(i) for i in range(10))
    for value in [-1, 10, "9", True]:
        with pytest.raises(ValidationError):
            Question.model_validate(
                {
                    **score.model_dump(),
                    "target": {"kind": "hard", "value": value, "source": "program"},
                }
            )
    for levels in [None, 1, 11, True]:
        with pytest.raises(ValidationError):
            question(type="score", options=None, score_levels=levels)
    assert answer_tokens(question(type="noul", options=None)) == ("yes", "no")
    with pytest.raises(ValidationError):
        question(type="noul")


def test_sample_integrity_and_pool_leakage():
    sample = generate_samples(1)[0]
    payload = sample.model_dump()
    payload["questions"][0]["target"] = None
    with pytest.raises(ValidationError):
        Sample.model_validate(payload)
    payload = sample.model_dump()
    payload["meta"]["license"] = "unknown"
    with pytest.raises(ValidationError):
        Sample.model_validate(payload)
    with pytest.raises(ValueError, match="duplicate"):
        validate_pool_isolation([sample, sample])
    other = sample.model_dump()
    other["id"] = "variant"
    other["meta"]["pool"] = "dev"
    with pytest.raises(ValueError, match="crosses pools"):
        validate_pool_isolation([sample, Sample.model_validate(other)])


def test_readout_conditions_answers_separately_from_abstention():
    result = read_probabilities({"A": math.log(3), "B": 0.0, "U": math.log(4)}, ["A", "B"])
    assert result.probabilities == pytest.approx({"A": 0.75, "B": 0.25})
    assert result.p_insufficient == pytest.approx(0.5)
    assert result.confidence == pytest.approx(0.5)
    assert result.prediction == "A"
    extreme = read_probabilities(
        {"A": -1e308, "B": -1e308, "U": 1e308}, ["A", "B"], temperature=1e-300
    )
    assert extreme.p_insufficient == 1
    assert extreme.probabilities == {"A": 0.5, "B": 0.5}
    assert extreme.confidence == 0


@pytest.mark.parametrize(
    "logits,options,temp",
    [
        ({"A": 0.0, "U": 0.0}, ["A"], 1.0),
        ({"A": 0.0, "B": 0.0}, ["A", "B"], 1.0),
        ({"A": 0.0, "B": 0.0, "U": math.nan}, ["A", "B"], 1.0),
        ({"A": 0.0, "B": 0.0, "U": math.inf}, ["A", "B"], 1.0),
        ({"A": 0.0, "B": 0.0, "U": 0.0}, ["A", "B"], 0.0),
        ({"A": 0.0, "B": 0.0, "U": 0.0}, ["A", "B"], math.nan),
        ({"A": 0.0, "B": 0.0, "U": 0.0}, ["A", "A"], 1.0),
    ],
)
def test_bad_readout_rejected(logits, options, temp):
    with pytest.raises(ValueError):
        read_probabilities(logits, options, temperature=temp)


class RecordingTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return "official:" + messages[0]["content"]

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return [ord(text)]


def test_template_is_deterministic_target_free_and_official():
    sample = generate_samples(1)[0]
    q = sample.questions[0]
    tokenizer = RecordingTokenizer()
    payload = render_payload(sample.prefix, q)
    assert "target" not in json.loads(payload)
    assert "source" not in json.loads(payload)
    assert render_prompt(tokenizer, sample.prefix, q) == "official:" + payload
    assert tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert check_answer_tokens(tokenizer, q) == {"A": 65, "B": 66, "C": 67, "U": 85}
    media = Prefix(state="image", media=[{"type": "image", "uri": "local.png", "sha256": "0" * 64}])
    with pytest.raises(ValueError, match="multimodal"):
        render_payload(media, q)


def test_metrics_exact_definitions_and_confidence_boundaries():
    result = evaluate([[0.8, 0.2], [0.8, 0.2]], [0, 1])
    assert result["accuracy"] == 0.5
    assert result["ece15"] == pytest.approx(0.3)
    assert result["brier"] == pytest.approx(0.68)
    assert result["coverage_at_error"] == 0.0
    assert evaluate([[1.0, 0.0]], [0])["ece15"] == 0
    # The longest acceptable prefix can appear after a shorter prefix fails.
    rows = [[1.0, 0.0]] * 20
    assert evaluate(rows, [1] + [0] * 19)["coverage_at_error"] == 1


@pytest.mark.parametrize(
    "rows,targets",
    [
        ([], []),
        ([[0.5, 0.5]], []),
        ([[math.nan, 0.5]], [0]),
        ([[0.5, 0.4]], [0]),
        ([[1.0]], [0]),
        ([[0.5, 0.5]], [True]),
        ([[0.5, 0.5]], [2]),
    ],
)
def test_metrics_refuse_malformed_data(rows, targets):
    with pytest.raises(ValueError):
        evaluate(rows, targets)


@pytest.mark.parametrize(
    "metrics,thresholds",
    [
        ({}, {}),
        ({}, {"accuracy": {"min": 0.9}}),
        ({"accuracy": math.nan}, {"accuracy": {"min": 0.9}}),
        ({"accuracy": True}, {"accuracy": {"min": 0.9}}),
        ({"accuracy": 1.0}, {"accuracy": {"minimum": 0.9}}),
        ({"accuracy": 1.0}, {"accuracy": {"min": 2.0, "max": 1.0}}),
        ({"accuracy": 1.0}, {"accuracy": {"min": math.inf}}),
    ],
)
def test_gate_fails_closed(metrics, thresholds):
    assert evaluate_gate(metrics, thresholds)["passed"] is False


def test_gate_cli_exits_fail_closed(tmp_path):
    metrics, thresholds = tmp_path / "metrics.json", tmp_path / "thresholds.json"
    metrics.write_text('{"accuracy": 0.98}')
    thresholds.write_text('{"accuracy": {"min": 0.98}}')
    cmd = [sys.executable, "-m", "eval.gate", str(metrics), str(thresholds)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)["passed"] is True
    thresholds.write_text('{"accuracy": {"min": 0.99}}')
    assert subprocess.run(cmd, capture_output=True).returncode == 1
    metrics.write_text('{"accuracy": NaN}')
    assert subprocess.run(cmd, capture_output=True).returncode == 2
    metrics.write_text('{"accuracy": 0.1, "accuracy": 1}')
    assert subprocess.run(cmd, capture_output=True).returncode == 2


def test_gate_honors_strict_boundaries():
    assert not evaluate_gate({"delta": 0.001}, {"delta": {"max_exclusive": 0.001}})["passed"]
    assert evaluate_gate({"delta": 0.0009}, {"delta": {"max_exclusive": 0.001}})["passed"]
    assert not evaluate_gate({"delta": 0.0}, {"delta": {"min_exclusive": 0.0}})["passed"]
    assert not evaluate_gate({"delta": 0.0}, {"delta": {"min": 0.0, "max_exclusive": 0.0}})[
        "passed"
    ]


def test_shared_state_precedes_question_and_score_rubric_is_preserved():
    sample = generate_samples(1)[0]
    first, second = [render_payload(sample.prefix, q) for q in sample.questions[:2]]
    shared = '{"state":' + json.dumps(sample.prefix.state, ensure_ascii=False) + ","
    assert first.startswith(shared) and second.startswith(shared)
    score = question(
        type="score", options=None, score_levels=2, score_descriptions=["No urgency", "Urgent"]
    )
    assert json.loads(render_payload(sample.prefix, score))["answers"] == {
        "0": "No urgency",
        "1": "Urgent",
    }
    with pytest.raises(ValidationError):
        question(type="score", options=None, score_levels=2, score_descriptions=["Missing rubric"])


@pytest.mark.parametrize("ids", [[1, 2], []])
def test_multitoken_answers_fail_preflight(ids):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return ids

    with pytest.raises(ValueError, match="exactly one"):
        check_answer_tokens(Tokenizer(), question())


def test_colliding_answer_ids_fail_preflight():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [1]

    with pytest.raises(ValueError, match="distinct"):
        check_answer_tokens(Tokenizer(), question())


def test_gate_rejects_integer_overflow_as_nonfinite():
    assert not evaluate_gate({"value": 10**1000}, {"value": {"min": 0.0}})["passed"]


def test_coverage_cannot_split_identical_confidence_by_input_order():
    rows = [[0.8, 0.2], [0.8, 0.2]]
    assert evaluate(rows, [0, 1])["coverage_at_error"] == 0
    assert evaluate(rows, [1, 0])["coverage_at_error"] == 0
