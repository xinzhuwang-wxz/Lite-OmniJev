"""Strict TRAIN cross-video bindings for the fixed matched course; no media preprocessing."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data.schema import HardTarget, Sample
from scripts.s0_media_dev import source_hash_for_video
from scripts.s1_text_pools import canonical_sample_sha
from train.video_pair_objective import video_pair_loss

if TYPE_CHECKING:
    from train.trainer import LoopConfig


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def bound_bytes(path: Path, expected: str | None) -> bytes:
    payload = path.read_bytes()
    require(digest(payload) == expected, f"video pair file SHA mismatch: {path}")
    return payload


def question_hash(sample: Sample) -> str:
    prefix = sample.prefix.model_dump(mode="json")
    prefix.pop("media")
    question = sample.questions[0].model_dump(mode="json")
    question.pop("target")
    return digest(json.dumps({"prefix": prefix, "question": question}, sort_keys=True).encode())


def member(sample: Sample, official: dict[str, Any]) -> dict[str, Any]:
    require(sample.meta.pool == "train" and sample.meta.modality == ["video"], "TRAIN video only")
    require(
        len(sample.questions) == 1
        and len(sample.prefix.media) == 1
        and sample.prefix.media[0].type == "video",
        "single video/question required",
    )
    question = sample.questions[0]
    target = question.target
    require(question.type == "noul" and isinstance(target, HardTarget), "hard official noul gold")
    assert isinstance(target, HardTarget)
    require(
        target.value in ("yes", "no") and target.source == "clevrer_descriptive_exist_answer",
        "official CLEVRER gold only",
    )
    require(
        official["question"] == question.instruction
        and str(official["answer"]).casefold() == target.value,
        "official text/gold mismatch",
    )
    scene = int(sample.meta.family_id.split(":")[1])
    require(
        sample.id == f"clevrer-train-{scene:05d}-{official['question_id']}", "official scene/id"
    )
    require(isinstance(official["program"], list) and bool(official["program"]), "official program")
    require(
        source_hash_for_video({"scene_index": scene}, official, sample.prefix.media[0].sha256)
        == sample.meta.source_sha256,
        "official program/media source binding mismatch",
    )
    return {
        "item_id": sample.id,
        "family_id": sample.meta.family_id,
        "sample_sha256": canonical_sample_sha(sample),
        "official_source_sha256": sample.meta.source_sha256,
        "media_sha256": sample.prefix.media[0].sha256,
        "original_gold": target.value,
    }


def build_video_pairs(
    samples: list[Sample],
    metadata_json: str,
    official_questions: dict[str, dict[str, Any]],
    ledger_sha256: str,
) -> dict[str, Any]:
    """CPU builder; selected official records are checked against frozen sample provenance.

    The loader additionally checks every exact canonical sample against the SHA-bound
    TRAIN ledger, including target gold (official source hashes alone omit answers).
    """
    require(
        len(samples) == 1200 and len({s.id for s in samples}) == 1200, "fixed unique1200 course"
    )
    metadata = json.loads(metadata_json)
    require(
        metadata.get("metadata_only") is True and len(metadata["rows"]) == 300,
        "original matched metadata300 required",
    )
    order = list(range(1200))
    random.Random(42).shuffle(order)
    expected_ids = {
        samples[order[step * 12 + offset]].id for step in range(100) for offset in range(6)
    }
    require(set(official_questions) == expected_ids, "exact600 original question records required")
    pairs = []
    for pair_index, row in enumerate(metadata["rows"]):
        draw = (pair_index // 3) * 12 + (pair_index % 3) * 2
        indices = order[draw : draw + 2]
        left, right = [samples[index] for index in indices]
        records = [official_questions[s.id] for s in (left, right)]
        members = [member(s, record) for s, record in zip((left, right), records, strict=True)]
        require(
            type(row["pair"]) is int
            and row["pair"] == pair_index
            and row["metadata_only"] is True
            and row["members"] == members,
            "actual shuffle/member metadata mismatch",
        )
        require(
            left.meta.family_id != right.meta.family_id
            and members[0]["media_sha256"] != members[1]["media_sha256"],
            "pair must use distinct video families/media",
        )
        require({m["original_gold"] for m in members} == {"yes", "no"}, "opposite official gold")
        qhash = question_hash(left)
        phash = digest(json.dumps(records[0]["program"], sort_keys=True).encode())
        require(
            qhash == question_hash(right) == row["model_nonmedia_question_sha256"]
            and records[0]["program"] == records[1]["program"]
            and phash == row["official_program_sha256"]
            and digest(left.questions[0].instruction.encode()) == row["text_sha256"],
            "complete nonmedia question/program binding mismatch",
        )
        positive = 0 if members[0]["original_gold"] == "yes" else 1
        pairs.append(
            {
                "draw": draw,
                "indices": indices,
                "members": members,
                "positive_offset": positive,
                "match_binding": {
                    "pair_id": f"matched-{pair_index}",
                    "positive_question_sha256": qhash,
                    "negative_question_sha256": qhash,
                    "positive_program_sha256": phash,
                    "negative_program_sha256": phash,
                },
            }
        )
    # Replay positions retain ordinary CE, and cannot silently acquire extra video pairs.
    require(
        all(
            samples[order[step * 12 + offset]].meta.modality != ["video"]
            for step in range(100)
            for offset in range(6, 12)
        ),
        "six nonvideo replay rows",
    )
    data = ("\n".join(s.model_dump_json() for s in samples) + "\n").encode()
    return {
        "header": {
            "schema": "train-cross-video-pairs-v1",
            "data_sha256": digest(data),
            "ledger_sha256": ledger_sha256,
            "metadata_sha256": digest(metadata_json.encode()),
            "seed": 42,
            "steps": 100,
            "accumulation": 12,
        },
        "matched_metadata_json": metadata_json,
        "official_questions": official_questions,
        "pairs": pairs,
    }


def load_video_pairs(
    config: LoopConfig, samples: list[Sample], data_path: Path | None
) -> dict[int, dict[str, Any]]:
    require(data_path is not None, "video pairs require explicit data")
    assert data_path is not None
    ledger_bytes = bound_bytes(
        Path(config.video_pair_ledger or ""), config.video_pair_ledger_sha256
    )
    ledger: dict[str, dict[str, Any]] = {}
    for line in ledger_bytes.splitlines():
        if line.strip():
            row = json.loads(line)
            previous = ledger.setdefault(row["item_id"], row)
            require(previous == row, "conflicting video pair ledger rows")
    for sample in samples:
        row = ledger.get(sample.id, {})
        require(
            row.get("pool") == sample.meta.pool == "train"
            and row.get("derived_sample_sha256") == canonical_sample_sha(sample),
            "exact canonical TRAIN ledger binding mismatch",
        )
    payload = json.loads(
        bound_bytes(Path(config.video_pair_sidecar or ""), config.video_pair_sidecar_sha256)
    )
    expected = build_video_pairs(
        samples,
        payload["matched_metadata_json"],
        payload["official_questions"],
        config.video_pair_ledger_sha256 or "",
    )
    require(
        json.dumps(payload, sort_keys=True) == json.dumps(expected, sort_keys=True),
        "video pair sidecar/header/draw binding mismatch",
    )
    require(
        digest(data_path.read_bytes()) == expected["header"]["data_sha256"],
        "video pair data changed",
    )
    return {row["draw"]: row for row in expected["pairs"]}


def validate_video_pair_example(example: dict[str, Any], row: dict[str, Any], offset: int) -> None:
    member_row = row["members"][offset]
    mapping = example.get("answer_token_ids", {})
    require(
        example.get("id") == member_row["item_id"]
        and set(mapping) == {"yes", "no", "U"}
        and len(set(mapping.values())) == 3
        and example.get("token_ids") == [mapping["yes"], mapping["no"], mapping["U"]]
        and example.get("target")
        == ([1.0, 0.0, 0.0] if member_row["original_gold"] == "yes" else [0.0, 1.0, 0.0])
        and "pixel_values_videos" in example,
        "video pair runtime sample/mapping/gold mismatch",
    )


def backward_video_pair(
    first: tuple[Any, Any, dict[str, Any]],
    second: tuple[Any, Any, dict[str, Any]],
    row: dict[str, Any],
    *,
    weight: float,
    accumulation: int,
) -> tuple[float, float]:
    """Backpropagate both CE terms plus one pair term, releasing graphs in the caller."""
    require(weight == 0.1 and accumulation == 12, "fixed video pair normalization")
    for offset, (_, _, example) in enumerate((first, second)):
        validate_video_pair_example(example, row, offset)
    ordered = (first, second)
    pi = row["positive_offset"]
    positive, negative = ordered[pi], ordered[1 - pi]
    pl, pc, pe = positive
    nl, nc, ne = negative
    pair = video_pair_loss(
        pl,
        nl,
        positive_token_ids=pe["answer_token_ids"],
        negative_token_ids=ne["answer_token_ids"],
        positive_gold="yes",
        negative_gold="no",
        positive_family_id=row["members"][pi]["family_id"],
        negative_family_id=row["members"][1 - pi]["family_id"],
        match_binding=row["match_binding"],
    )
    pm, nm = pe["answer_token_ids"], ne["answer_token_ids"]
    margin = (
        pl[pm["yes"]].float() - pl[pm["no"]].float() - nl[nm["yes"]].float() + nl[nm["no"]].float()
    )
    ((pc + nc + weight * pair) / accumulation).backward()
    return float(pair.detach().cpu()), float(margin.detach().cpu())
