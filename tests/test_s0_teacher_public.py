"""Teacher adapter semantics; fake head is never acceptance evidence."""

import io
import json
from types import SimpleNamespace

from data.jevbench import JevBenchTask
from scripts.s0_teacher_public import collect, teacher_question


def noul_task():
    return JevBenchTask(
        id="public-noul",
        split_name="original",
        family="policy",
        group=None,
        state="State without the answer",
        question={
            "type": "noul",
            "instructions": "Decide",
            "criteria": {"true": "All constraints satisfied", "false": "Any constraint violated"},
        },
        labels=("no", "yes"),
        expected="no",
        source_sha256="a" * 64,
        project_tokens=("yes", "no"),
        source_to_project={"no": "no", "yes": "yes"},
        project_to_source={"yes": "yes", "no": "no"},
    )


def test_teacher_question_preserves_noul_rubric_without_gold():
    instruction, options = teacher_question(noul_task())
    assert "All constraints satisfied" in instruction
    assert "Any constraint violated" in instruction
    assert options == ("yes: yes", "no: no")
    assert "expected" not in instruction


def test_teacher_collect_preserves_source_label_order_and_no_fabricated_u():
    class FakeTeacher:
        def predict(self, *, state, question, options):
            assert state == "State without the answer"
            return SimpleNamespace(probabilities={options[0]: 0.2, options[1]: 0.8})

    stream = io.StringIO()
    records = collect((noul_task(),), FakeTeacher(), stream)
    assert len(records) == 2
    assert records[0]["probabilities"] == {"yes": 0.2, "no": 0.8}
    assert records[0]["p_U"] is None
    assert len([json.loads(line) for line in stream.getvalue().splitlines()]) == 2
