"""Convert public MMAU test-mini and MVBench 9-task rows into Sample-contract jsonl.

Rows, hashing and row identities come from the exact loaders the S0 teacher runs used
(``eval.teacher_benchmarks``, imported — never copied), so a student evaluation over
these samples stays row-aligned with the teacher two-pass evidence:
``reports/teacher-benchmark-two-pass-complete-001.json`` (card scores in
``configs/video-eval-assets.json``: mmau_micro 0.631, mvbench_micro 0.5309).

Media protocol, recorded honestly for score interpretation:
- MMAU: audio bytes are embedded in the source parquet and are written out verbatim;
  the Sample ``sha256`` is the loader's ``media_bytes_sha256``. Rare clips over the
  student loader's hard 30 s audio limit are stream-copy trimmed to 29.999 s and
  listed under ``trimmed_rows_over_30s`` in ``build-summary.json`` (their Media sha
  covers the trimmed file; parquet provenance stays in ``meta.source_sha256``).
- MVBench: the original videos (Charades/FunQA/Perception/STA) are not staged on this
  box. The teacher protocol froze exactly 16 canonical PNG frames per row
  (``artifacts/public-video-protocol-freeze-001``). ``scripts/s1_dev_eval`` requires a
  single video container per sample, so each row's 16 canonical PNGs are assembled
  losslessly (libx264rgb qp=0, 4 fps, 16 frames, mp4/mov). The loader's even
  16-frame sampling over a 16-frame file selects exactly the teacher's frames, and a
  decode roundtrip asserts every frame's RGB24 sha256 matches the frozen manifest.
  Remaining protocol delta vs the teacher: ``train.multimodal.load_media`` scales
  frames to fit 640x640 before the processor; the teacher path did not.

GPU evaluation commands (main session launches these; this ticket stays CPU-only)::

    cd /root/autodl-tmp/s1-text-lora-001
    python -m scripts.s1_dev_eval \\
      --model-dir <E4B base model dir> \\
      --adapter <D3P-600 run adapter dir> \\
      --data /root/autodl-tmp/s1-text-lora-001/artifacts/public-bench-samples-001/mmau-test-mini-samples-v1.jsonl \\
      --passes 1 --video-frames 16 \\
      --media-root /root/autodl-tmp/s1-text-lora-001/artifacts/public-bench-samples-001/media \\
      --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/public-bench-eval-001/mmau \\
      --report /root/autodl-tmp/s1-text-lora-001/reports/s4-public-bench-mmau-001.json

    # same command with mvbench-public-9task-samples-v1.jsonl,
    # --output-dir .../public-bench-eval-001/mvbench,
    # --report .../reports/s4-public-bench-mvbench-001.json

Run on the box with an env that has pyarrow + PIL + pydantic (``envs/quant``)::

    cd /root/autodl-tmp/s1-text-lora-001 && PYTHONPATH=. \\
      /root/autodl-tmp/envs/quant/bin/python .scratch/s4/public_bench_samples.py build-all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from data.schema import HardTarget, Media, Metadata, Prefix, Question, Sample
from eval.teacher_benchmarks import (
    MMAU_EXPECTED_ROWS,
    MVBench_EXPECTED_ROWS,
    file_sha256,
    load_mmau_examples,
    load_mvbench_examples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "public-bench-samples-001"
DEFAULT_EVAL_TREE = Path("/root/autodl-tmp/s5-calibration-code-fa83acc")
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "video-eval-assets.json"
DEFAULT_MMAU_PARQUET = (
    Path("/root/autodl-tmp/lite-omnijev/artifacts/data/research-eval/mmau")
    / "ccd9696c0111ea7060827598f310558df0b71b0a"
    / "test_mini.parquet"
)
DEFAULT_FRAME_MANIFEST = Path(
    "/root/autodl-tmp/before-gpu-release-4366197/artifacts/"
    "public-video-protocol-freeze-001/canonical-frame-manifest.json"
)
DEFAULT_METADATA_DIR = (
    Path("/root/autodl-tmp/lite-omnijev/artifacts/data/research-eval")
    / "video-eval-source-metadata"
)
DEFAULT_CACHE_ROOT = (
    Path("/root/autodl-tmp/lite-omnijev/artifacts/public-video-protocol-freeze-001")
    / "canonical-mac"
)

VIDEO_FPS = 4  # 16 frames -> 4.0 s container, inside load_media's 60 s probe limit
VIDEO_FRAMES_PER_ROW = 16
MMAU_AUDIO_MAX_SECONDS = 30  # train.multimodal.load_media limit for audio
MEDIA_MAX_BYTES = 64 * 1024 * 1024  # train.multimodal.resolve_media limit
MMAU_LICENSE = "CC-BY-NC-4.0"  # configs/video-eval-assets.json: dataset and embedded audio
# reports/video-eval-source-audit-001.json: MVBench metadata is MIT; media is
# academic-research-only under underlying source agreements.
MVBENCH_LICENSE = "metadata:MIT;media:academic-research-only"
MMAU_DATA_VERSION = "mmau-test-mini-public-bench-samples-v1"
MVBENCH_DATA_VERSION = "mvbench-public-9task-public-bench-samples-v1"
TARGET_SOURCE_MMAU = "mmau-test-mini:parquet:answer"
TARGET_SOURCE_MVBENCH = "mvbench-public-9task:task-json:answer"
CHOICE_STATE_AUDIO = "Answer the multiple-choice question using the attached audio."
NOUL_STATE_AUDIO = "Answer the yes/no question using the attached audio."
CHOICE_STATE_VIDEO = "Answer the multiple-choice question using the attached video."


def _file_sha256(path: Path) -> str:
    return file_sha256(path)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.decode(errors='replace')[-2000:]}"
        )


def _ffprobe_json(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,avg_frame_rate,nb_frames,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    return json.loads(result.stdout)


def _probe_duration(path: Path) -> float:
    probed = _ffprobe_json(path)
    return float(probed["format"]["duration"])


def _trim_wav(path: Path, max_seconds: float) -> float:
    """Sample-exact PCM WAV head cut to ``max_seconds``; returns the new duration."""
    import wave

    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())
    bytes_per_frame = params.nchannels * params.sampwidth
    keep_frames = int(params.framerate * max_seconds)
    if keep_frames <= 0 or keep_frames * bytes_per_frame > len(frames):
        raise ValueError(f"{path}: cannot trim to {max_seconds}s")
    tmp_path = path.with_suffix(".trim.wav")
    with wave.open(str(tmp_path), "wb") as target:
        target.setnchannels(params.nchannels)
        target.setsampwidth(params.sampwidth)
        target.setframerate(params.framerate)
        target.writeframes(frames[: keep_frames * bytes_per_frame])
    os.replace(tmp_path, path)
    return keep_frames / params.framerate


def _sample_for(
    *,
    sample_id: str,
    family_id: str,
    state: str,
    media_type: str,
    uri: str,
    media_sha256: str,
    instruction: str,
    question_type: str,
    options: list[str] | None,
    target_value: str,
    target_source: str,
    source_dataset: str,
    source_revision: str,
    source_sha256: str,
    license_spdx: str,
    data_version: str,
) -> Sample:
    return Sample(
        id=sample_id,
        prefix=Prefix(
            state=state,
            media=[Media(type=media_type, uri=uri, sha256=media_sha256)],
        ),
        questions=[
            Question(
                qid="q0",
                type=question_type,  # type: ignore[arg-type]
                instruction=instruction,
                options=options,
                target=HardTarget(kind="hard", value=target_value, source=target_source),
            )
        ],
        meta=Metadata(
            modality=[media_type],  # type: ignore[list-item]
            source_dataset=source_dataset,
            source_revision=source_revision,
            source_sha256=source_sha256,
            license=license_spdx,
            usage="research",
            pool="dev",
            family_id=family_id,
            data_version=data_version,
        ),
    )


def build_mmau_samples(
    *, parquet: Path, config_path: Path, output_root: Path, workers: int = 8
) -> dict:
    config = json.loads(Path(config_path).read_text())
    mmau_config = config["mmau"]
    examples = load_mmau_examples(Path(parquet), mmau_config["file"])
    if len(examples) != MMAU_EXPECTED_ROWS:
        raise ValueError(f"mmau loader returned {len(examples)} rows, expected {MMAU_EXPECTED_ROWS}")

    media_dir = output_root / "media" / "mmau"
    media_dir.mkdir(parents=True, exist_ok=True)

    def stage(example_index: int) -> tuple[int, str, float, bool, str]:
        example = examples[example_index]
        media_path = media_dir / f"{example.report_id}.wav"
        if not media_path.exists() or _file_sha256(media_path) != example.media_bytes_sha256:
            assert example.media_bytes is not None
            if not example.media_bytes.startswith(b"RIFF"):
                raise ValueError(
                    f"mmau {example.row_id}: embedded audio is not RIFF/WAV; "
                    "container mapping unknown, stopping"
                )
            media_path.write_bytes(example.media_bytes)
        duration = _probe_duration(media_path)
        trimmed = False
        if duration > MMAU_AUDIO_MAX_SECONDS:
            # load_media refuses audio over 30 s for the whole cohort; trim the rare
            # overlong clip by exact sample cut at 29.999 s and record it. The Media
            # sha256 then covers the trimmed file; parquet provenance stays in meta.
            trimmed_duration = _trim_wav(media_path, 29.999)
            if not 0 < trimmed_duration <= MMAU_AUDIO_MAX_SECONDS:
                raise ValueError(
                    f"mmau {example.row_id}: trimmed duration {trimmed_duration:.3f}s "
                    f"still outside (0, {MMAU_AUDIO_MAX_SECONDS}]"
                )
            duration, trimmed = trimmed_duration, True
        if duration <= 0:
            raise ValueError(f"mmau {example.row_id}: empty audio")
        disk_sha256 = _file_sha256(media_path)
        if not trimmed and disk_sha256 != example.media_bytes_sha256:
            raise ValueError(f"mmau {example.row_id}: staged audio sha mismatch")
        return example_index, example.row_id, duration, trimmed, disk_sha256

    with ThreadPoolExecutor(max_workers=workers) as pool:
        staged = list(pool.map(stage, range(len(examples))))
    durations = {index: duration for index, _rid, duration, _t, _s in staged}
    trimmed_ids = {row_id: round(duration, 3) for _i, row_id, duration, t, _s in staged if t}

    type_counts: dict[str, int] = {}
    rows: list[str] = []
    seen_ids: set[str] = set()
    disk_shas = {index: sha for index, _rid, _d, _t, sha in staged}
    for example_index, example in enumerate(examples):
        lowered = tuple(option.lower() for option in example.options)
        if lowered == ("yes", "no"):
            question_type = "noul"
            options = None
            target_value = example.options[example.label_index].lower()
            state = NOUL_STATE_AUDIO
        else:
            question_type = "choice"
            options = list(example.options)
            if len(options) < 2 or len(options) > 20 or len(set(options)) != len(options):
                raise ValueError(f"mmau {example.row_id}: options not 2-20 distinct values")
            target_value = chr(ord("A") + example.label_index)
            state = CHOICE_STATE_AUDIO
        type_counts[question_type] = type_counts.get(question_type, 0) + 1
        sample_id = f"mmau-test-mini:{example.row_id}"
        if sample_id in seen_ids:
            raise ValueError(f"duplicate mmau sample id: {sample_id}")
        seen_ids.add(sample_id)
        sample = _sample_for(
            sample_id=sample_id,
            family_id=sample_id,
            state=state,
            media_type="audio",
            uri=f"mmau/{example.report_id}.wav",
            media_sha256=disk_shas[example_index],
            instruction=example.question,
            question_type=question_type,
            options=options,
            target_value=target_value,
            target_source=TARGET_SOURCE_MMAU,
            source_dataset=mmau_config["repo_id"],
            source_revision=mmau_config["revision"],
            source_sha256=example.source_sha256,
            license_spdx=MMAU_LICENSE,
            data_version=MMAU_DATA_VERSION,
        )
        sample.model_validate(sample.model_dump(mode="json"))
        rows.append(sample.model_dump_json())

    out_path = output_root / "mmau-test-mini-samples-v1.jsonl"
    _write_lines(out_path, rows)
    if len(rows) != MMAU_EXPECTED_ROWS:
        raise ValueError(f"mmau jsonl has {len(rows)} rows, expected {MMAU_EXPECTED_ROWS}")
    return {
        "benchmark": "mmau-test-mini",
        "rows": len(rows),
        "question_types": type_counts,
        "jsonl": str(out_path),
        "jsonl_sha256": _file_sha256(out_path),
        "media_dir": str(media_dir),
        "media_files": len(list(media_dir.iterdir())),
        "source_parquet_sha256": examples[0].source_sha256,
        "max_audio_duration_seconds": round(max(durations.values()), 3),
        "trimmed_rows_over_30s": trimmed_ids,
        "license": MMAU_LICENSE,
    }


def _stage_row_frames(
    example_frame_paths: list[Path], expected_sha256: list[str], frame_dir: Path
) -> None:
    frame_dir.mkdir(parents=True)
    for index, (source, sha) in enumerate(zip(example_frame_paths, expected_sha256, strict=True)):
        if _file_sha256(source) != sha:
            raise ValueError(f"canonical PNG sha mismatch for staged row frame {index}")
        shutil.copyfile(source, frame_dir / f"{index:04d}.png")


def _assemble_row_video(
    example: object, cache_root: Path, target_path: Path, work_dir: Path
) -> None:
    """Losslessly pack a row's 16 canonical PNGs into the mp4 that s1_dev_eval loads."""
    frames = example.video_frames  # type: ignore[attr-defined]
    if len(frames) != VIDEO_FRAMES_PER_ROW:
        raise ValueError(f"mvbench {example.row_id}: expected 16 canonical frames")  # type: ignore[attr-defined]
    paths = [cache_root / frame.png for frame in frames]
    expected = [frame.png_sha256 for frame in frames]
    tmp_path = work_dir / f"{example.report_id}.mp4"  # type: ignore[attr-defined]
    with tempfile.TemporaryDirectory(prefix="mvbench-mp4-", dir=work_dir) as tmp:
        frame_dir = Path(tmp) / "frames"
        _stage_row_frames(paths, expected, frame_dir)
        _run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-v",
                "error",
                "-f",
                "image2",
                "-framerate",
                str(VIDEO_FPS),
                "-i",
                str(frame_dir / "%04d.png"),
                "-c:v",
                "libx264rgb",
                "-qp",
                "0",
                "-preset",
                "medium",
                "-pix_fmt",
                "rgb24",
                "-threads",
                "1",
                "-y",
                str(tmp_path),
            ]
        )
        _verify_row_container(example, tmp_path)
    if tmp_path.stat().st_size > MEDIA_MAX_BYTES:
        raise ValueError(f"mvbench {example.row_id}: assembled video exceeds byte limit")
    os.replace(tmp_path, target_path)


def _verify_row_container(example: object, path: Path) -> None:
    """ffprobe + RGB24 roundtrip: the decoded frames must be the canonical frames."""
    probed = _ffprobe_json(path)
    duration = float(probed["format"]["duration"])
    if not 0 < duration <= 60:
        raise ValueError(f"mvbench {example.row_id}: container duration {duration:.2f}s")
    streams = [s for s in probed["streams"] if s["codec_type"] == "video"]
    if len(streams) != 1:
        raise ValueError(f"mvbench {example.row_id}: expected exactly one video stream")
    stream = streams[0]
    if int(stream["nb_frames"]) != VIDEO_FRAMES_PER_ROW:
        raise ValueError(f"mvbench {example.row_id}: nb_frames={stream['nb_frames']}")
    if stream["avg_frame_rate"] != f"{VIDEO_FPS}/1":
        raise ValueError(f"mvbench {example.row_id}: avg_frame_rate={stream['avg_frame_rate']}")
    width, height = int(stream["width"]), int(stream["height"])
    if (width, height) != (example.width, example.height):  # type: ignore[attr-defined]
        raise ValueError(
            f"mvbench {example.row_id}: container {width}x{height} != manifest "
            f"{example.width}x{example.height}"  # type: ignore[attr-defined]
        )
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mvbench {example.row_id}: roundtrip decode failed")
    payload = result.stdout
    frame_bytes = width * height * 3
    expected = [frame.rgb24_sha256 for frame in example.video_frames]  # type: ignore[attr-defined]
    if len(payload) != frame_bytes * len(expected):
        raise ValueError(
            f"mvbench {example.row_id}: roundtrip returned {len(payload)} bytes, "
            f"expected {frame_bytes * len(expected)}"
        )
    for index, frame in enumerate(expected):
        digest = hashlib.sha256(
            payload[index * frame_bytes : (index + 1) * frame_bytes]
        ).hexdigest()
        if digest != frame:
            raise ValueError(
                f"mvbench {example.row_id}: frame {index} RGB24 sha mismatch after lossless pack"
            )


def build_mvbench_samples(
    *,
    frame_manifest: Path,
    metadata_dir: Path,
    config_path: Path,
    cache_root: Path,
    output_root: Path,
    workers: int = 6,
) -> dict:
    examples = load_mvbench_examples(
        frame_manifest=Path(frame_manifest),
        metadata_dir=Path(metadata_dir),
        config_path=Path(config_path),
    )
    if len(examples) != MVBench_EXPECTED_ROWS:
        raise ValueError(
            f"mvbench loader returned {len(examples)} rows, expected {MVBench_EXPECTED_ROWS}"
        )
    cache_root = Path(cache_root)
    media_dir = output_root / "media" / "mvbench"
    media_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    mvbench_config = config["mvbench"]

    def assemble(example_index: int) -> int:
        example = examples[example_index]
        target_path = media_dir / f"{example.report_id}.mp4"
        if not target_path.exists() or target_path.stat().st_size == 0:
            _assemble_row_video(example, cache_root, target_path, work_dir)
        return example_index

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(assemble, range(len(examples))))

    task_counts: dict[str, int] = {}
    rows: list[str] = []
    seen_ids: set[str] = set()
    for example in examples:
        options = list(example.options)
        if len(options) < 2 or len(options) > 20 or len(set(options)) != len(options):
            raise ValueError(f"mvbench {example.row_id}: options not 2-20 distinct values")
        task_counts[example.group] = task_counts.get(example.group, 0) + 1
        sample_id = f"mvbench-public-9task:{example.row_id}"
        if sample_id in seen_ids:
            raise ValueError(f"duplicate mvbench sample id: {sample_id}")
        seen_ids.add(sample_id)
        sample = _sample_for(
            sample_id=sample_id,
            family_id=f"mvbench-video:{example.source_sha256}",
            state=CHOICE_STATE_VIDEO,
            media_type="video",
            uri=f"mvbench/{example.report_id}.mp4",
            media_sha256=_file_sha256(media_dir / f"{example.report_id}.mp4"),
            instruction=example.question,
            question_type="choice",
            options=options,
            target_value=chr(ord("A") + example.label_index),
            target_source=TARGET_SOURCE_MVBENCH,
            source_dataset=mvbench_config["repo_id"],
            source_revision=mvbench_config["revision"],
            source_sha256=example.source_sha256,
            license_spdx=MVBENCH_LICENSE,
            data_version=MVBENCH_DATA_VERSION,
        )
        sample.model_validate(sample.model_dump(mode="json"))
        rows.append(sample.model_dump_json())

    out_path = output_root / "mvbench-public-9task-samples-v1.jsonl"
    _write_lines(out_path, rows)
    if len(rows) != MVBench_EXPECTED_ROWS:
        raise ValueError(f"mvbench jsonl has {len(rows)} rows, expected {MVBench_EXPECTED_ROWS}")
    if any(count != 200 for count in task_counts.values()) or len(task_counts) != 9:
        raise ValueError(f"mvbench task denominators broken: {task_counts}")
    return {
        "benchmark": "mvbench-public-9task",
        "rows": len(rows),
        "task_counts": task_counts,
        "jsonl": str(out_path),
        "jsonl_sha256": _file_sha256(out_path),
        "media_dir": str(media_dir),
        "media_files": len(list(media_dir.iterdir())),
        "license": MVBENCH_LICENSE,
    }


def _write_lines(path: Path, rows: list[str]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")
    os.replace(tmp_path, path)


VERIFY_SNIPPET = """
import json, sys
from pathlib import Path
from scripts.s1_dev_eval import load_samples
from train.multimodal import resolve_media

media_root = Path(sys.argv[1])
expected = {"mmau": 1000, "mvbench": 1800}
summary = {}
for key, want in expected.items():
    path = Path(sys.argv[2 if key == "mmau" else 3])
    samples = load_samples(path, media_root=media_root)
    if len(samples) != want:
        raise SystemExit(f"{key}: loaded {len(samples)} rows, expected {want}")
    for sample in samples:
        media = sample.prefix.media[0]
        resolve_media(media_root, media.uri, media.sha256, max_bytes=64 * 1024 * 1024)
    summary[key] = {
        "rows": len(samples),
        "types": sorted({s.questions[0].type for s in samples}),
        "jsonl_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
    }
print("VERIFY-OK " + json.dumps(summary, sort_keys=True))
"""


def verify_samples(
    *, eval_tree: Path, mmau_jsonl: Path, mvbench_jsonl: Path, media_root: Path
) -> dict:
    """Load both jsonl with the frozen S5 eval tree's own loader and re-verify media."""
    result = subprocess.run(
        [sys.executable, "-c", VERIFY_SNIPPET, str(media_root), str(mmau_jsonl), str(mvbench_jsonl)],
        cwd=str(eval_tree),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"verify failed in {eval_tree}:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
        )
    payload = result.stdout.strip().splitlines()[-1]
    print(payload)
    return json.loads(payload.removeprefix("VERIFY-OK "))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=["build-mmau", "build-mvbench", "build-all", "verify"]
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_MMAU_PARQUET)
    parser.add_argument("--frame-manifest", type=Path, default=DEFAULT_FRAME_MANIFEST)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--eval-tree", type=Path, default=DEFAULT_EVAL_TREE)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    started = datetime.now(UTC)
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict = {"started_utc": started.isoformat(), "command": args.command}
    if args.command in ("build-mmau", "build-all"):
        summary["mmau"] = build_mmau_samples(
            parquet=args.parquet, config_path=args.config, output_root=output_root
        )
        print(json.dumps(summary["mmau"], ensure_ascii=False, indent=2))
    if args.command in ("build-mvbench", "build-all"):
        summary["mvbench"] = build_mvbench_samples(
            frame_manifest=args.frame_manifest,
            metadata_dir=args.metadata_dir,
            config_path=args.config,
            cache_root=args.cache_root,
            output_root=output_root,
            workers=args.workers,
        )
        print(json.dumps(summary["mvbench"], ensure_ascii=False, indent=2))
    if args.command == "verify":
        summary["verify"] = verify_samples(
            eval_tree=args.eval_tree,
            mmau_jsonl=output_root / "mmau-test-mini-samples-v1.jsonl",
            mvbench_jsonl=output_root / "mvbench-public-9task-samples-v1.jsonl",
            media_root=output_root / "media",
        )
    summary["finished_utc"] = datetime.now(UTC).isoformat()
    if args.command.startswith("build"):
        summary_path = output_root / "build-summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
