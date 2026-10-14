"""JevBench public protocol adapter; research-eval only, never training input."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from data.schema import HardTarget, Metadata, Prefix, Question, Sample, answer_tokens
from serve.gateway.app import DecisionRequest, InputQuestion
from train.readout import read_probabilities

SOURCE_REVISION = "7625ecb1b075cfe4d09d34141d353325825eea1d"
SOURCE_URL = f"https://github.com/fstandhartinger/jevbench/tree/{SOURCE_REVISION}"
SOURCE_DATASET = "fstandhartinger/jevbench-public"
DATA_VERSION = f"jevbench-public-{SOURCE_REVISION}"
SPLIT_FILES = {
    "original": "datasets/public/original.jsonl",
    "easy": "datasets/public/easy.jsonl",
    "hard": "datasets/public/hard.jsonl",
}
EXPECTED_SPLIT_ROWS = {"original": 72, "easy": 48, "hard": 111}
EXPECTED_PRIMITIVE_COUNTS = {"choice": 139, "score": 18, "noul": 74}
EXPECTED_PUBLIC_DECISIONS = 231
REQUIRED_SOURCE_FILE_COUNT = 8


class JevBenchProtocolError(ValueError):
    """Raised when the frozen public source or adapter contract is violated."""


@dataclass(frozen=True)
class SourceFileReceipt:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class JevBenchTask:
    """One frozen public JevBench decision and its reversible project mapping."""

    id: str
    split_name: Literal["original", "easy", "hard"]
    family: str
    group: str | None
    state: str | dict[str, Any]
    question: dict[str, Any]
    labels: tuple[str, ...]
    expected: str | int
    source_sha256: str
    project_tokens: tuple[str, ...]
    source_to_project: dict[str, str]
    project_to_source: dict[str, str]

    @property
    def primitive(self) -> Literal["choice", "score", "noul"]:
        value = self.question["type"]
        if value not in {"choice", "score", "noul"}:
            raise JevBenchProtocolError(f"unsupported primitive: {value}")
        return cast(Literal["choice", "score", "noul"], value)

    @property
    def sample_family_id(self) -> str:
        return self.group or self.id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def deterministic_state_text(state: str | Mapping[str, Any]) -> str:
    if isinstance(state, str):
        if not state.strip():
            raise JevBenchProtocolError("state cannot be blank")
        return state
    if not isinstance(state, Mapping):
        raise JevBenchProtocolError("state must be a string or object")
    try:
        return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise JevBenchProtocolError("state object must be deterministic JSON") from error


def expected_source_files(config: Mapping[str, Any]) -> tuple[SourceFileReceipt, ...]:
    if config.get("revision") != SOURCE_REVISION:
        raise JevBenchProtocolError("JevBench source revision does not match locked adapter")
    files = config.get("files")
    if not isinstance(files, list) or len(files) != REQUIRED_SOURCE_FILE_COUNT:
        raise JevBenchProtocolError("JevBench config must declare the locked 8 public files")
    receipts: list[SourceFileReceipt] = []
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise JevBenchProtocolError("source file receipt must be an object")
        path = entry.get("path")
        size = entry.get("bytes")
        digest = entry.get("sha256")
        if (
            not isinstance(path, str)
            or path.startswith("/")
            or ".." in Path(path).parts
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise JevBenchProtocolError("invalid source file receipt")
        if path in seen:
            raise JevBenchProtocolError(f"duplicate source file receipt: {path}")
        seen.add(path)
        receipts.append(SourceFileReceipt(path, size, digest))
    for required in SPLIT_FILES.values():
        if required not in seen:
            raise JevBenchProtocolError(f"missing public split receipt: {required}")
    return tuple(receipts)


def verify_source_files(source_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    receipts = expected_source_files(config)
    observed: dict[str, Any] = {}
    for receipt in receipts:
        path = source_root / receipt.path
        if not path.is_file() or path.is_symlink():
            raise JevBenchProtocolError(f"missing or unsafe source file: {receipt.path}")
        stat = path.stat()
        if stat.st_size != receipt.bytes:
            raise JevBenchProtocolError(f"source size mismatch: {receipt.path}")
        digest = sha256_file(path)
        if digest != receipt.sha256:
            raise JevBenchProtocolError(f"source sha256 mismatch: {receipt.path}")
        observed[receipt.path] = {"bytes": stat.st_size, "sha256": digest}
    return observed


def _require_keys(row: Mapping[str, Any]) -> None:
    required = {
        "id",
        "family",
        "state",
        "question",
        "labels",
        "expected",
        "group",
        "split",
        "provenance",
    }
    if set(row) != required:
        raise JevBenchProtocolError(
            f"source row keys mismatch: missing={sorted(required - set(row))} "
            f"extra={sorted(set(row) - required)}"
        )


def _validate_question(question: Any) -> dict[str, Any]:
    if not isinstance(question, dict):
        raise JevBenchProtocolError("question must be an object")
    if set(question) != {"type", "instructions", "criteria"}:
        raise JevBenchProtocolError("question keys mismatch")
    primitive = question["type"]
    if primitive not in {"choice", "score", "noul"}:
        raise JevBenchProtocolError("unsupported question type")
    if not isinstance(question["instructions"], str) or not question["instructions"].strip():
        raise JevBenchProtocolError("question instructions must be nonempty")
    criteria = question["criteria"]
    if primitive in {"choice", "noul"}:
        if not isinstance(criteria, dict):
            raise JevBenchProtocolError("choice/noul criteria must be an object")
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value.strip()
            for key, value in criteria.items()
        ):
            raise JevBenchProtocolError("criteria keys and descriptions must be nonempty strings")
    elif (
        not isinstance(criteria, list)
        or not 2 <= len(criteria) <= 10
        or any(not isinstance(value, str) or not value.strip() for value in criteria)
    ):
        raise JevBenchProtocolError("score criteria must be 2-10 nonempty descriptions")
    return question


def _mapping_for(
    question: Mapping[str, Any], labels: Sequence[str]
) -> tuple[tuple[str, ...], dict[str, str], dict[str, str]]:
    primitive = question["type"]
    if primitive == "choice":
        project = tuple(chr(65 + index) for index in range(len(labels)))
    elif primitive == "score":
        project = tuple(str(index) for index in range(len(labels)))
    else:
        project = ("yes", "no")
    if primitive == "noul":
        source_to_project = {"no": "no", "yes": "yes"}
    else:
        source_to_project = dict(zip(labels, project, strict=True))
    project_to_source = {value: key for key, value in source_to_project.items()}
    if len(project_to_source) != len(project):
        raise JevBenchProtocolError("project token mapping is not reversible")
    return project, source_to_project, project_to_source


def _validate_row(row: Mapping[str, Any], split_name: str, line_number: int) -> JevBenchTask:
    _require_keys(row)
    if row["split"] != "public":
        raise JevBenchProtocolError("only public split rows are admissible")
    if not isinstance(row["id"], str) or not row["id"].strip():
        raise JevBenchProtocolError("row id must be nonempty")
    if not isinstance(row["family"], str) or not row["family"].strip():
        raise JevBenchProtocolError("source family must be nonempty")
    group = row["group"]
    if group is not None and (not isinstance(group, str) or not group.strip()):
        raise JevBenchProtocolError("group must be null or nonempty string")
    state = row["state"]
    deterministic_state_text(state)
    question = _validate_question(row["question"])
    labels = row["labels"]
    if (
        not isinstance(labels, list)
        or not 2 <= len(labels) <= 20
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise JevBenchProtocolError("labels must be 2-20 distinct nonempty strings")
    primitive = question["type"]
    expected = row["expected"]
    if primitive == "choice":
        criteria = question["criteria"]
        if not isinstance(criteria, dict) or set(criteria) != set(labels):
            raise JevBenchProtocolError("choice criteria must exactly cover source labels")
        if expected not in labels:
            raise JevBenchProtocolError("choice expected label missing from labels")
    elif primitive == "score":
        if labels != [str(index) for index in range(len(labels))]:
            raise JevBenchProtocolError("score labels must be contiguous numeric strings")
        if type(expected) is not int or str(expected) not in labels:
            raise JevBenchProtocolError("score expected must remain the original integer level")
        if len(question["criteria"]) != len(labels):
            raise JevBenchProtocolError("score criteria length must match labels")
    else:
        if labels != ["no", "yes"]:
            raise JevBenchProtocolError("noul source labels must stay in no/yes order")
        if set(question["criteria"]) != {"true", "false"}:
            raise JevBenchProtocolError("noul criteria must have true/false meanings")
        if expected not in labels:
            raise JevBenchProtocolError("noul expected label missing from labels")
    project, source_to_project, project_to_source = _mapping_for(question, labels)
    return JevBenchTask(
        id=row["id"],
        split_name=split_name,  # type: ignore[arg-type]
        family=row["family"],
        group=group,
        state=state,
        question=question,
        labels=tuple(labels),
        expected=expected,
        source_sha256=sha256_json(row),
        project_tokens=project,
        source_to_project=source_to_project,
        project_to_source=project_to_source,
    )


def load_public_tasks(source_root: Path, config: Mapping[str, Any]) -> tuple[JevBenchTask, ...]:
    verify_source_files(source_root, config)
    tasks: list[JevBenchTask] = []
    per_split: dict[str, int] = {}
    for split_name, relative in SPLIT_FILES.items():
        count = 0
        with (source_root / relative).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise JevBenchProtocolError(f"blank JSONL line in {relative}:{line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise JevBenchProtocolError("source row must be an object")
                tasks.append(_validate_row(row, split_name, line_number))
                count += 1
        per_split[split_name] = count
    if per_split != EXPECTED_SPLIT_ROWS:
        raise JevBenchProtocolError(f"public split row counts changed: {per_split}")
    if len(tasks) != EXPECTED_PUBLIC_DECISIONS:
        raise JevBenchProtocolError("public decision count changed")
    ids = [task.id for task in tasks]
    if len(set(ids)) != len(ids):
        raise JevBenchProtocolError("duplicate public decision IDs")
    primitive_counts = {
        primitive: sum(1 for task in tasks if task.primitive == primitive)
        for primitive in EXPECTED_PRIMITIVE_COUNTS
    }
    if primitive_counts != EXPECTED_PRIMITIVE_COUNTS:
        raise JevBenchProtocolError(f"primitive counts changed: {primitive_counts}")
    return tuple(tasks)


def build_input_question(task: JevBenchTask) -> InputQuestion:
    if task.primitive == "choice":
        criteria = {
            project: task.question["criteria"][source]
            for source, project in task.source_to_project.items()
        }
    elif task.primitive == "score":
        criteria = task.question["criteria"]
    else:
        criteria = task.question["criteria"]
    return InputQuestion.model_validate(
        {
            "type": task.primitive,
            "instructions": task.question["instructions"],
            "criteria": criteria,
        }
    )


def build_decision_request(task: JevBenchTask) -> DecisionRequest:
    return DecisionRequest.model_validate(
        {
            "state": task.state,
            "media": [],
            "questions": {task.id: build_input_question(task).model_dump()},
        }
    )


def build_sample(task: JevBenchTask) -> Sample:
    item = build_input_question(task)
    if task.primitive == "choice":
        question = Question.model_validate(
            {
                "qid": task.id,
                "type": "choice",
                "instruction": item.instructions,
                "options": list(item.criteria.values()) if isinstance(item.criteria, dict) else [],
                "target": HardTarget(
                    kind="hard",
                    value=task.source_to_project[str(task.expected)],
                    source="jevbench-public",
                ).model_dump(),
            }
        )
    elif task.primitive == "score":
        question = Question.model_validate(
            {
                "qid": task.id,
                "type": "score",
                "instruction": item.instructions,
                "score_levels": len(task.labels),
                "score_descriptions": list(task.question["criteria"]),
                "target": HardTarget(
                    kind="hard", value=task.expected, source="jevbench-public"
                ).model_dump(),
            }
        )
    else:
        question = Question.model_validate(
            {
                "qid": task.id,
                "type": "noul",
                "instruction": item.instructions,
                "target": HardTarget(
                    kind="hard", value=str(task.expected), source="jevbench-public"
                ).model_dump(),
            }
        )
    return Sample(
        id=task.id,
        prefix=Prefix(state=deterministic_state_text(task.state), media=[]),
        questions=[question],
        meta=Metadata(
            modality=["text"],
            source_dataset=SOURCE_DATASET,
            source_revision=SOURCE_REVISION,
            source_sha256=task.source_sha256,
            license="MIT",
            usage="research",
            pool="dev",
            family_id=task.sample_family_id,
            data_version=DATA_VERSION,
        ),
    )


def corpus_row(task: JevBenchTask) -> dict[str, Any]:
    payload = build_decision_request(task).model_dump(mode="json")
    return {
        "schema_version": 1,
        "source_id": task.id,
        "split": task.split_name,
        "source_family": task.family,
        "sample_family_id": task.sample_family_id,
        "source_labels": list(task.labels),
        "project_tokens": list(task.project_tokens),
        "source_to_project": task.source_to_project,
        "project_to_source": task.project_to_source,
        "sample": build_sample(task).model_dump(mode="json"),
        "model_payload": payload,
        "model_payload_sha256": sha256_json(payload),
        "gold_in_model_payload": False,
        "provenance_in_model_payload": False,
    }


def project_distribution_to_source(
    task: JevBenchTask, project_probabilities: Mapping[str, float]
) -> dict[str, float]:
    if set(project_probabilities) != set(task.project_tokens):
        raise JevBenchProtocolError("project probabilities must contain every non-U answer token")
    return {
        source: float(project_probabilities[project])
        for source, project in task.source_to_project.items()
    }


def logits_to_source_distribution(
    task: JevBenchTask, project_logits: Mapping[str, float]
) -> tuple[dict[str, float], float]:
    readout = read_probabilities(project_logits, answer_tokens(build_sample(task).questions[0]))
    return project_distribution_to_source(task, readout.probabilities), readout.p_insufficient


def public_protocol_summary(tasks: Sequence[JevBenchTask]) -> dict[str, Any]:
    groups = [task.group for task in tasks]
    primitive_counts = {
        primitive: sum(1 for task in tasks if task.primitive == primitive)
        for primitive in EXPECTED_PRIMITIVE_COUNTS
    }
    return {
        "public_decisions": len(tasks),
        "per_split": {
            split: {
                "rows": sum(1 for task in tasks if task.split_name == split),
                "primitive_counts": {
                    primitive: sum(
                        1
                        for task in tasks
                        if task.split_name == split and task.primitive == primitive
                    )
                    for primitive in EXPECTED_PRIMITIVE_COUNTS
                },
            }
            for split in SPLIT_FILES
        },
        "primitive_counts": primitive_counts,
        "literal_group_values_including_null": len(set(groups)),
        "nonempty_group_values": len({group for group in groups if group}),
        "null_group_rows": sum(1 for group in groups if group is None),
        "official_paraphrase_pair_groups": sum(
            1
            for group in {task.group for task in tasks if task.group}
            if sum(1 for task in tasks if task.group == group) >= 2
        ),
        "teacher_matched_protocol_verified": False,
        "training_admitted": False,
        "gpu_used": False,
    }
