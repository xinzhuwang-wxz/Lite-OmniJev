"""Compose the first image+audio development cohort from approved dev builders."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from data.schema import Sample, answer_tokens
from eval.sealed_gate import DevelopmentRecord

SEED = 42
DEV_COUNT = 500
DATA_VERSION = "image-audio-compositional-noul-dev-v1"
PROTOCOL = "s0-crossmodal-dev-v1"


class CrossmodalDevError(ValueError):
    """User-correctable input or output contract failure."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def script_sha256() -> str:
    return sha256_file(Path(__file__))


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def allocation_key(modality: Literal["image", "audio"], family_id: str) -> str:
    return sha256_bytes(f"{PROTOCOL}:{SEED}:{modality}:{family_id}".encode())


def row_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise CrossmodalDevError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(raw, dict):
                raise CrossmodalDevError(f"{path}:{line_no}: record must be an object")
            rows.append(raw)
    return rows


def load_samples(path: Path, *, modality: Literal["image", "audio"]) -> list[Sample]:
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    seen_families: set[str] = set()
    seen_media_sha: set[str] = set()
    for raw in load_jsonl(path):
        sample = Sample.model_validate(raw)
        if sample.id in seen_ids:
            raise CrossmodalDevError(f"duplicate {modality} sample id")
        seen_ids.add(sample.id)
        if sample.meta.pool != "dev":
            raise CrossmodalDevError(f"{modality} sample is not dev pool")
        if sample.meta.usage != "commercial":
            raise CrossmodalDevError(f"{modality} sample is not commercial usage")
        if sample.meta.family_id in seen_families:
            raise CrossmodalDevError(f"duplicate {modality} family")
        seen_families.add(sample.meta.family_id)
        if modality not in sample.meta.modality or len(sample.prefix.media) != 1:
            raise CrossmodalDevError(f"{modality} sample must contain one declared media item")
        media = sample.prefix.media[0]
        if media.type != modality:
            raise CrossmodalDevError(f"{modality} sample media type mismatch")
        if media.sha256 in seen_media_sha:
            raise CrossmodalDevError(f"duplicate {modality} media sha256")
        seen_media_sha.add(media.sha256)
        if len(sample.questions) != 1:
            raise CrossmodalDevError(f"{modality} sample must contain one question")
        question = sample.questions[0]
        if modality == "image" and question.type != "choice":
            raise CrossmodalDevError("image source sample must be choice")
        if modality == "audio" and question.type != "noul":
            raise CrossmodalDevError("audio source sample must be noul")
        samples.append(sample)
    return samples


def validate_source_report(
    path: Path | None, *, modality: Literal["image", "audio"], expected_dev_sha256: str
) -> str:
    if path is None:
        raise CrossmodalDevError(f"{modality} source report is required")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("status") != "passed" or raw.get("exit_code") != 0:
        raise CrossmodalDevError(f"{modality} source report did not pass")
    outputs = raw.get("outputs")
    if not isinstance(outputs, dict):
        raise CrossmodalDevError(f"{modality} source report missing outputs")
    if outputs.get("dev_jsonl_sha256") != expected_dev_sha256:
        raise CrossmodalDevError(f"{modality} source report dev hash mismatch")
    return sha256_file(path)


def validate_dev_count(dev_count: int) -> None:
    if dev_count <= 0 or dev_count % 4 != 0:
        raise CrossmodalDevError("dev_count must be a positive multiple of 4")


def safe_relative(uri: str) -> Path:
    relative = Path(uri)
    if (
        ":" in uri
        or relative.is_absolute()
        or ".." in relative.parts
        or "\x00" in uri
        or any(part != PurePosixPath(part).name for part in relative.parts)
    ):
        raise CrossmodalDevError("unsafe relative media uri")
    return relative


def source_media_path(root: Path, uri: str) -> Path:
    if root.is_symlink() or root.absolute() != root.resolve() or not root.is_dir():
        raise CrossmodalDevError("media root must be a real directory")
    relative = safe_relative(uri)
    path = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise CrossmodalDevError("media symlinks are forbidden")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise CrossmodalDevError("source media missing or outside root")
    return path


def validate_output_paths(paths: list[Path], media_out_dir: Path) -> None:
    resolved = [path.resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise CrossmodalDevError("output paths must be unique")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise CrossmodalDevError("refusing to overwrite existing outputs")
    if media_out_dir.exists() or media_out_dir.is_symlink():
        raise CrossmodalDevError("refusing to overwrite media output directory")


def copy_component_media(
    *,
    root: Path,
    media_out_dir: Path,
    source_uri: str,
    expected_sha: str,
    modality: Literal["image", "audio"],
) -> tuple[str, int]:
    source = source_media_path(root, source_uri)
    if sha256_file(source) != expected_sha:
        raise CrossmodalDevError(f"{modality} media sha256 mismatch")
    original = safe_relative(source_uri)
    suffix = original
    if original.parts and original.parts[0] == modality:
        suffix = Path(*original.parts[1:])
    relative = Path(modality) / suffix
    destination = media_out_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise CrossmodalDevError("duplicate output media path")
    shutil.copyfile(source, destination)
    if sha256_file(destination) != expected_sha:
        raise CrossmodalDevError(f"copied {modality} media sha256 mismatch")
    return relative.as_posix(), destination.stat().st_size


def target_value(sample: Sample) -> str:
    target = sample.questions[0].target
    if target is None or target.kind != "hard" or type(target.value) is not str:
        raise CrossmodalDevError("component target must be a hard string label")
    return target.value


def wrong_image_choice(sample: Sample, pair_index: int) -> str:
    question = sample.questions[0]
    tokens = list(answer_tokens(question))
    gold = target_value(sample)
    wrong = [token for token in tokens if token != gold]
    if not wrong:
        raise CrossmodalDevError("image choice has no wrong option")
    return wrong[
        int(sha256_bytes(f"{SEED}:wrong:{sample.id}:{pair_index}".encode()), 16) % len(wrong)
    ]


def option_text(sample: Sample, token: str) -> str:
    question = sample.questions[0]
    if question.options is None:
        raise CrossmodalDevError("image question missing options")
    index = ord(token) - ord("A")
    if not 0 <= index < len(question.options):
        raise CrossmodalDevError("image target token outside options")
    return question.options[index]


def proposal_for(
    image: Sample, audio: Sample, *, pair_index: int, dev_count: int
) -> tuple[Literal["both_true", "image_false_audio_true", "image_true_audio_false"], str, str, str]:
    image_gold = target_value(image)
    audio_gold = target_value(audio)
    half = dev_count // 2
    three_quarters = half + dev_count // 4
    if pair_index < half:
        return "both_true", image_gold, audio_gold, "yes"
    if pair_index < three_quarters:
        return "image_false_audio_true", wrong_image_choice(image, pair_index), audio_gold, "no"
    return "image_true_audio_false", image_gold, ("no" if audio_gold == "yes" else "yes"), "no"


def source_hash_for(
    *,
    image: Sample,
    audio: Sample,
    image_proposal: str,
    audio_proposal: str,
    rule: str,
) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "protocol": PROTOCOL,
                "image_sample_id": image.id,
                "audio_sample_id": audio.id,
                "image_source_sha256": image.meta.source_sha256,
                "audio_source_sha256": audio.meta.source_sha256,
                "image_media_sha256": image.prefix.media[0].sha256,
                "audio_media_sha256": audio.prefix.media[0].sha256,
                "image_proposal": image_proposal,
                "audio_proposal": audio_proposal,
                "proposal_rule": rule,
            }
        ).encode()
    )


def family_id_for(image: Sample, audio: Sample) -> str:
    return (
        "crossmodal:"
        + sha256_bytes(
            canonical_json(
                {
                    "image_family_id": image.meta.family_id,
                    "audio_family_id": audio.meta.family_id,
                }
            ).encode()
        )[:24]
    )


def build_sample(
    *,
    image: Sample,
    audio: Sample,
    image_uri: str,
    audio_uri: str,
    pair_index: int,
    dev_count: int,
) -> tuple[Sample, dict[str, Any], dict[str, Any]]:
    rule, image_proposal, audio_proposal, gold = proposal_for(
        image, audio, pair_index=pair_index, dev_count=dev_count
    )
    image_question = image.questions[0]
    audio_question = audio.questions[0]
    source_hash = source_hash_for(
        image=image,
        audio=audio,
        image_proposal=image_proposal,
        audio_proposal=audio_proposal,
        rule=rule,
    )
    image_options = ", ".join(
        f"{token}: {text}"
        for token, text in zip(
            answer_tokens(image_question), image_question.options or [], strict=True
        )
    )
    instruction = (
        "Are BOTH of these claims true? "
        f"(1) In the image, the answer to '{image_question.instruction}' "
        f"with options [{image_options}] is "
        f"{image_proposal}: {option_text(image, image_proposal)}. "
        f"(2) For the audio, the answer to '{audio_question.instruction}' is {audio_proposal}."
    )
    sample = Sample.model_validate(
        {
            "id": f"crossmodal-dev-{source_hash[:16]}",
            "prefix": {
                "state": "Answer the yes/no question using both the attached image and audio.",
                "media": [
                    {"type": "image", "uri": image_uri, "sha256": image.prefix.media[0].sha256},
                    {"type": "audio", "uri": audio_uri, "sha256": audio.prefix.media[0].sha256},
                ],
            },
            "questions": [
                {
                    "qid": "noul",
                    "type": "noul",
                    "instruction": instruction,
                    "target": {
                        "kind": "hard",
                        "value": gold,
                        "source": "programmatic_component_answer_and",
                    },
                }
            ],
            "meta": {
                "modality": ["image", "audio"],
                "source_dataset": f"{image.meta.source_dataset}+{audio.meta.source_dataset}",
                "source_revision": f"{image.meta.source_revision}+{audio.meta.source_revision}",
                "source_sha256": source_hash,
                "license": f"image:{image.meta.license};audio:{audio.meta.license}",
                "usage": "commercial",
                "pool": "dev",
                "family_id": family_id_for(image, audio),
                "data_version": DATA_VERSION,
            },
        }
    )
    component = {
        "item_id": sample.id,
        "crossmodal_family_id": sample.meta.family_id,
        "crossmodal_family_hash": sha256_bytes(sample.meta.family_id.encode()),
        "pool": "dev",
        "proposal_rule": rule,
        "program_truth": gold,
        "image": {
            "item_id": image.id,
            "family_id": image.meta.family_id,
            "family_hash": sha256_bytes(image.meta.family_id.encode()),
            "source_hash": image.meta.source_sha256,
            "media_sha256": image.prefix.media[0].sha256,
            "proposal": image_proposal,
            "proposal_option_text_sha256": sha256_bytes(
                option_text(image, image_proposal).encode()
            ),
        },
        "audio": {
            "item_id": audio.id,
            "family_id": audio.meta.family_id,
            "family_hash": sha256_bytes(audio.meta.family_id.encode()),
            "source_hash": audio.meta.source_sha256,
            "media_sha256": audio.prefix.media[0].sha256,
            "proposal": audio_proposal,
        },
        "component_family_separate_ledger": True,
    }
    media = {
        "item_id": sample.id,
        "image_uri": image_uri,
        "image_sha256": image.prefix.media[0].sha256,
        "audio_uri": audio_uri,
        "audio_sha256": audio.prefix.media[0].sha256,
    }
    return sample, component, media


def development_record_for(sample: Sample) -> DevelopmentRecord:
    return DevelopmentRecord(
        item_id=sample.id,
        modality="crossmodal",
        family_hash=sha256_bytes(sample.meta.family_id.encode()),
        source_hash=sample.meta.source_sha256,
        source_dataset=sample.meta.source_dataset,
        source_revision=sample.meta.source_revision,
        license=sample.meta.license,
        label_origin="program_truth",
    )


def write_jsonl(
    path: Path, rows: list[Sample] | list[DevelopmentRecord] | list[dict[str, Any]]
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        if isinstance(row, Sample | DevelopmentRecord):
            lines.append(row.model_dump_json())
        else:
            lines.append(canonical_json(row))
    payload = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(payload, encoding="utf-8")
    return sha256_bytes(payload.encode())


def output_report(
    *,
    args: argparse.Namespace,
    started_at: float,
    status: str,
    exit_code: int,
    blocker: str | None,
    outputs: dict[str, str] | None = None,
    split: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finished_at = time.time()
    config = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "dev_count": args.dev_count,
        "data_version": DATA_VERSION,
    }
    return {
        "status": status,
        "exit_code": exit_code,
        "blocker": blocker,
        "command": [sys.executable, "-m", "scripts.s0_crossmodal_dev", *sys.argv[1:]],
        "protocol": PROTOCOL,
        "config": {**config, "config_sha256": sha256_bytes(canonical_json(config).encode())},
        "primitive_scope": {
            "compositional_noul_only": True,
            "complete_distribution_or_quality": False,
        },
        "training_admitted": False,
        "quality_gate": "not_run",
        "gpu_used": False,
        "holdout_accessed": False,
        "cost": None,
        "real_human_kappa_verified": False,
        "not_native_crossmodal_annotation": True,
        "source": source,
        "split": split,
        "outputs": outputs,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_head": git_head(),
            "script_sha256": script_sha256(),
        },
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": finished_at - started_at,
        },
    }


def run(args: argparse.Namespace) -> int:
    started_at = time.time()
    output_paths = [
        args.dev_out,
        args.dev_ledger_out,
        args.component_ledger_out,
        args.media_manifest_out,
    ]
    if args.report.exists() or args.report.is_symlink():
        return 1
    if args.report.resolve(strict=False) in {path.resolve(strict=False) for path in output_paths}:
        return 1
    try:
        validate_output_paths([*output_paths, args.report], args.media_out_dir)
        validate_dev_count(args.dev_count)
        image_dev_sha = sha256_file(args.image_dev)
        audio_dev_sha = sha256_file(args.audio_dev)
        image_report_sha = validate_source_report(
            args.image_source_report, modality="image", expected_dev_sha256=image_dev_sha
        )
        audio_report_sha = validate_source_report(
            args.audio_source_report, modality="audio", expected_dev_sha256=audio_dev_sha
        )
        image_samples = load_samples(args.image_dev, modality="image")
        audio_samples = load_samples(args.audio_dev, modality="audio")
        if len(image_samples) < args.dev_count or len(audio_samples) < args.dev_count:
            raise CrossmodalDevError("not enough source dev samples")
        image_media_sha = {sample.prefix.media[0].sha256 for sample in image_samples}
        audio_media_sha = {sample.prefix.media[0].sha256 for sample in audio_samples}
        if image_media_sha.intersection(audio_media_sha):
            raise CrossmodalDevError("same media sha256 appears in image and audio components")
        images = sorted(
            image_samples,
            key=lambda sample: (allocation_key("image", sample.meta.family_id), sample.id),
        )[: args.dev_count]
        audios = sorted(
            audio_samples,
            key=lambda sample: (allocation_key("audio", sample.meta.family_id), sample.id),
        )[: args.dev_count]
        samples: list[Sample] = []
        components: list[dict[str, Any]] = []
        media_rows: list[dict[str, Any]] = []
        for pair_index, (image, audio) in enumerate(zip(images, audios, strict=True)):
            image_uri, image_bytes = copy_component_media(
                root=args.image_media_root,
                media_out_dir=args.media_out_dir,
                source_uri=image.prefix.media[0].uri,
                expected_sha=image.prefix.media[0].sha256,
                modality="image",
            )
            audio_uri, audio_bytes = copy_component_media(
                root=args.audio_media_root,
                media_out_dir=args.media_out_dir,
                source_uri=audio.prefix.media[0].uri,
                expected_sha=audio.prefix.media[0].sha256,
                modality="audio",
            )
            sample, component, media = build_sample(
                image=image,
                audio=audio,
                image_uri=image_uri,
                audio_uri=audio_uri,
                pair_index=pair_index,
                dev_count=args.dev_count,
            )
            samples.append(sample)
            components.append(component)
            media_rows.append({**media, "image_bytes": image_bytes, "audio_bytes": audio_bytes})
        shuffle = random.Random(SEED)
        order = list(range(len(samples)))
        shuffle.shuffle(order)
        samples = [samples[index] for index in order]
        components = [components[index] for index in order]
        media_rows = [media_rows[index] for index in order]
        outputs = {
            "dev_jsonl_sha256": write_jsonl(args.dev_out, samples),
            "development_ledger_sha256": write_jsonl(
                args.dev_ledger_out, [development_record_for(sample) for sample in samples]
            ),
            "component_ledger_sha256": write_jsonl(args.component_ledger_out, components),
            "media_manifest_sha256": write_jsonl(args.media_manifest_out, media_rows),
        }
        yes = sum(1 for sample in samples if target_value(sample) == "yes")
        no = sum(1 for sample in samples if target_value(sample) == "no")
        report = output_report(
            args=args,
            started_at=started_at,
            status="passed",
            exit_code=0,
            blocker=None,
            outputs=outputs,
            split={
                "dev_items": len(samples),
                "yes": yes,
                "no": no,
                "image_source_items": len(image_samples),
                "audio_source_items": len(audio_samples),
                "image_families": len({sample.meta.family_id for sample in images}),
                "audio_families": len({sample.meta.family_id for sample in audios}),
                "proposal_rules": {
                    rule: sum(1 for row in components if row["proposal_rule"] == rule)
                    for rule in ("both_true", "image_false_audio_true", "image_true_audio_false")
                },
            },
            source={
                "image_dev_sha256": image_dev_sha,
                "audio_dev_sha256": audio_dev_sha,
                "image_source_report_sha256": image_report_sha,
                "audio_source_report_sha256": audio_report_sha,
                "source_mode": "compositional_from_image_choice_and_audio_noul_dev",
            },
        )
        code = 0
    except (OSError, CrossmodalDevError, ValueError, json.JSONDecodeError) as exc:
        report = output_report(
            args=args,
            started_at=started_at,
            status="blocked",
            exit_code=1,
            blocker=str(exc) if isinstance(exc, CrossmodalDevError) else type(exc).__name__,
        )
        code = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return code


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dev", type=Path, required=True)
    parser.add_argument("--audio-dev", type=Path, required=True)
    parser.add_argument("--image-media-root", type=Path, required=True)
    parser.add_argument("--audio-media-root", type=Path, required=True)
    parser.add_argument("--image-source-report", type=Path)
    parser.add_argument("--audio-source-report", type=Path)
    parser.add_argument("--dev-out", type=Path, required=True)
    parser.add_argument("--dev-ledger-out", type=Path, required=True)
    parser.add_argument("--component-ledger-out", type=Path, required=True)
    parser.add_argument("--media-manifest-out", type=Path, required=True)
    parser.add_argument("--media-out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dev-count", type=int, default=DEV_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
