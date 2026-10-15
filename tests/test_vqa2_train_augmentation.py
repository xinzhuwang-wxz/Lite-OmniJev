import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from data.schema import Sample
from scripts.s2_media_pools import canonical_sample_sha
from scripts.vqa2_train_augmentation import build


def fixture_inputs(tmp_path: Path, rows: list[tuple[int, int, str, list[str]]]) -> dict:
    sample = Sample.model_validate(
        {
            "id": "parent",
            "prefix": {
                "state": "Answer the multiple-choice question using the attached image.",
                "media": [{"type": "image", "uri": "images/1.jpg", "sha256": "a" * 64}],
            },
            "questions": [
                {
                    "qid": "choice",
                    "type": "choice",
                    "instruction": "Old question?",
                    "options": ["yes", "no"],
                    "target": {"kind": "hard", "value": "A", "source": "gold"},
                }
            ],
            "meta": {
                "modality": ["image"],
                "source_dataset": "A-OKVQA",
                "source_revision": "r",
                "source_sha256": "b" * 64,
                "license": "QA:Apache-2.0;media:CC-BY-2.0",
                "usage": "commercial",
                "pool": "train",
                "family_id": "coco:1",
                "data_version": "parent-v1",
            },
        }
    )
    args = {
        "parent_train": tmp_path / "parent.jsonl",
        "parent_ledger": tmp_path / "ledger.jsonl",
        "output_dir": tmp_path / "out",
    }
    args["parent_train"].write_text(sample.model_dump_json() + "\n")
    args["parent_ledger"].write_text(
        json.dumps(
            {
                "item_id": "parent",
                "family_id": "coco:1",
                "pool": "train",
                "derived_sample_sha256": canonical_sample_sha(sample),
            }
        )
        + "\n"
    )
    for split in ("train", "val"):
        questions, annotations = [], []
        for qid, image_id, question, answers in rows if split == "train" else []:
            questions.append({"question_id": qid, "image_id": image_id, "question": question})
            annotations.append(
                {
                    "question_id": qid,
                    "image_id": image_id,
                    "question_type": "what color",
                    "multiple_choice_answer": answers[0],
                    "answers": [
                        {"answer": a, "answer_id": i + 1, "answer_confidence": "yes"}
                        for i, a in enumerate(answers)
                    ],
                }
            )
        for kind, values, member in [
            ("questions", questions, f"v2_OpenEnded_mscoco_{split}2014_questions.json"),
            ("annotations", annotations, f"v2_mscoco_{split}2014_annotations.json"),
        ]:
            path = tmp_path / f"{split}-{kind}.zip"
            args[f"{split}_{kind}"] = path
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    member,
                    json.dumps(
                        {
                            "license": {"url": "http://creativecommons.org/licenses/by/4.0/"},
                            kind: values,
                        }
                    ),
                )
    return args


def test_choices_mapping_provenance_and_determinism(tmp_path):
    args = fixture_inputs(
        tmp_path,
        [
            (1, 1, "Is it?", ["YES."] * 10),
            (2, 1, "How many?", ["20"] * 10),
            (3, 1, "What color?", ["red"] * 10),
        ],
    )
    report = build(**args)
    assert report["category_counts"] == {"yes_no": 1, "integer": 1, "color": 1}
    samples = [
        Sample.model_validate_json(s)
        for s in (args["output_dir"] / "train.jsonl").read_text().splitlines()
    ]
    for s in samples:
        q = s.questions[0]
        gold = {"vqa2-train-1": "yes", "vqa2-train-2": "20", "vqa2-train-3": "red"}[s.id]
        assert q.options[ord(q.target.value) - 65] == gold
        assert len(set(q.options)) == (2 if gold == "yes" else 4)
        assert s.meta.pool == "train" and s.meta.family_id == "coco:1"
        assert s.prefix.media[0].sha256 == "a" * 64
    provenance = [
        json.loads(s) for s in (args["output_dir"] / "provenance.jsonl").read_text().splitlines()
    ]
    assert all(len(p["answers"]) == 10 for p in provenance)
    second = build(**{**args, "output_dir": tmp_path / "second"})
    assert report == second
    with pytest.raises(ValueError, match="overwrite"):
        build(**args)


@pytest.mark.parametrize("mode", ["missing", "reserved", "changed"])
def test_denies_invalid_parent_allocation(tmp_path, mode):
    args = fixture_inputs(tmp_path, [])
    row = json.loads(args["parent_ledger"].read_text())
    if mode == "missing":
        args["parent_ledger"].write_text("")
    elif mode == "reserved":
        row["pool"] = "holdout"
        args["parent_ledger"].write_text(json.dumps(row))
    else:
        row["derived_sample_sha256"] = "f" * 64
        args["parent_ledger"].write_text(json.dumps(row))
    with pytest.raises(ValueError, match="membership"):
        build(**args)
    assert not args["output_dir"].exists()


def test_filters_consensus_conflicts_and_duplicates(tmp_path):
    args = fixture_inputs(
        tmp_path,
        [
            (1, 999, "DO NOT OUTPUT", ["secret"] * 10),
            (2, 1, "Disagree?", ["yes"] * 9 + ["no"]),
            (3, 1, "Old question?", ["yes"] * 10),
            (4, 1, "Conflict?", ["yes"] * 10),
            (5, 1, "CONFLICT", ["no"] * 10),
            (6, 1, "New?", ["yes"] * 10),
            (7, 1, "NEW", ["yes"] * 10),
            (8, 1, "Who?", ["someone"] * 10),
        ],
    )
    report = build(**args)
    assert report["sample_count"] == 1
    assert report["excluded_counts"] == {
        "outside_parent_train": 1,
        "consensus_disagreement": 1,
        "parent_duplicate": 1,
        "conflicting_duplicate": 2,
        "duplicate_question": 1,
        "unsupported_answer": 1,
    }
    assert "secret" not in "".join(p.read_text() for p in args["output_dir"].iterdir())


def test_cli(tmp_path):
    args = fixture_inputs(tmp_path, [(1, 1, "Is it?", ["yes"] * 10)])
    command = [sys.executable, "-m", "scripts.vqa2_train_augmentation"]
    for key, value in args.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["sample_count"] == 1
    assert subprocess.run(command, capture_output=True).returncode == 2


def test_reserved_sibling_family_denied(tmp_path):
    args = fixture_inputs(tmp_path, [])
    with args["parent_ledger"].open("a") as stream:
        stream.write(
            json.dumps({"item_id": "sibling", "family_id": "coco:1", "pool": "calibration"}) + "\n"
        )
    with pytest.raises(ValueError, match="membership"):
        build(**args)


def test_excluded_family_payload_not_processed(tmp_path):
    args = fixture_inputs(tmp_path, [])
    for kind, member in [
        ("questions", "v2_OpenEnded_mscoco_train2014_questions.json"),
        ("annotations", "v2_mscoco_train2014_annotations.json"),
    ]:
        with zipfile.ZipFile(args[f"train_{kind}"], "w") as archive:
            archive.writestr(
                member,
                json.dumps(
                    {
                        "license": {"url": "http://creativecommons.org/licenses/by/4.0/"},
                        kind: [{"image_id": 999, "question": {"malformed": True}, "answers": None}],
                    }
                ),
            )
    assert build(**args)["sample_count"] == 0
