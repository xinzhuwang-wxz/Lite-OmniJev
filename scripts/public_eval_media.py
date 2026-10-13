"""CPU-only validation of pinned public MMAU-mini and MVBench STAR media."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import selectors
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

MMAU_PUBLIC_JSON_SHA256 = "9f18fda99f8dbc2bc5ecd6323fb5063309f54969810b5c6f1caeb3b8d904cf1c"


def soundfile_worker() -> int:
    """Keep the already installed media environment alive; never load Arrow here."""
    soundfile = importlib.import_module("soundfile")
    numpy = importlib.import_module("numpy")
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with soundfile.SoundFile(request["path"]) as audio:
                frames = 0
                for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
                    if not bool(numpy.isfinite(block).all()):
                        raise ValueError("non_finite_audio_samples")
                    frames += len(block)
                if frames != audio.frames or frames <= 0 or audio.samplerate <= 0:
                    raise ValueError("audio_frame_count_mismatch")
                result = {
                    "status": "decoded",
                    "decoder": "libsndfile",
                    "soundfile_version": soundfile.__version__,
                    "libsndfile_version": soundfile.__libsndfile_version__,
                    "codec": f"{audio.format}/{audio.subtype}",
                    "sample_rate": str(audio.samplerate),
                    "channels": audio.channels,
                    "width": None,
                    "height": None,
                    "duration_seconds": frames / audio.samplerate,
                    "frames_declared": audio.frames,
                    "frames_decoded": frames,
                    "finite_samples_verified": True,
                }
        except Exception as error:
            unsupported = type(error).__name__ == "LibsndfileError" and getattr(
                error, "code", None
            ) in {1, 4}
            result = {
                "status": "unsupported_codec" if unsupported else "failed",
                "error_type": type(error).__name__,
                "reason": str(error)[:200],
            }
        print(json.dumps(result), flush=True)
    return 0


@contextmanager
def audio_worker(python: Path) -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(
        [str(python), str(Path(__file__).resolve()), "--soundfile-worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield process
    finally:
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def worker_decode(process: subprocess.Popen[str], path: Path) -> dict[str, Any]:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps({"path": str(path.resolve())}) + "\n")
    process.stdin.flush()
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout=30):
            raise TimeoutError("soundfile_worker_timeout")
    line = process.stdout.readline()
    if not line:
        raise ValueError("soundfile_worker_exited_without_result")
    result: dict[str, Any] = json.loads(line)
    if result["status"] == "failed":
        raise ValueError("soundfile_audio_decode_failed")
    return result


def bounded_audio_decode(
    process: subprocess.Popen[str], path: Path, fallbacks: int
) -> dict[str, Any]:
    result = worker_decode(process, path)
    if result["status"] != "unsupported_codec":
        return result
    if fallbacks >= 32:
        raise ValueError("unsupported_codec_fallback_count_limit")
    metadata = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            timeout=30,
        )
    )
    stream = next(x for x in metadata["streams"] if x["codec_type"] == "audio")
    expanded = (
        float(metadata["format"]["duration"])
        * int(stream["sample_rate"])
        * int(stream["channels"])
        * 4
    )
    if not 0 < expanded <= 128 * 1024**2:
        raise ValueError("fallback_audio_expansion_limit")
    converted = path.with_suffix(".decoded.wav")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-n",
                "-threads",
                "1",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-c:a",
                "pcm_f32le",
                "-threads",
                "1",
                str(converted),
            ],
            capture_output=True,
            timeout=180,
            check=True,
        )
        if converted.stat().st_size > 128 * 1024**2 + 4096:
            raise ValueError("fallback_audio_expansion_limit")
        result = worker_decode(process, converted)
        if result["status"] != "decoded":
            raise ValueError("converted_audio_decode_failed")
        result.update(decoder="ffmpeg_then_libsndfile", original_codec=stream["codec_name"])
        return result
    finally:
        converted.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_zip_members(
    archive: zipfile.ZipFile, *, max_expanded_bytes: int = 2 * 1024**3
) -> dict[str, zipfile.ZipInfo]:
    """Reject the entire archive before extracting even one selected member."""
    members: dict[str, zipfile.ZipInfo] = {}
    expanded = 0
    for entry in archive.infolist():
        name = entry.filename.rstrip("/")
        path = PurePosixPath(name)
        mode = stat.S_IFMT(entry.external_attr >> 16)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or "\x00" in name
            or "\x00" in entry.orig_filename
            or str(path) != name
        ):
            raise ValueError("unsafe_zip_path")
        if name in members:
            raise ValueError("duplicate_zip_path")
        if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError("zip_special_file_denied")
        if mode == stat.S_IFDIR and not entry.is_dir():
            raise ValueError("zip_directory_mode_mismatch")
        if entry.flag_bits & 1:
            raise ValueError("encrypted_zip_denied")
        if entry.file_size > 256 * 1024**2:
            raise ValueError("zip_member_expansion_limit")
        expanded += entry.file_size
        if expanded > max_expanded_bytes or len(members) >= 10_000:
            raise ValueError("zip_total_expansion_limit")
        members[name] = entry
    return members


def probe_decode(path: Path, kind: str) -> dict[str, Any]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        timeout=30,
        check=True,
    )
    metadata = json.loads(probe.stdout)
    streams = [s for s in metadata["streams"] if s["codec_type"] == kind]
    if not streams:
        raise ValueError("required_media_stream_missing")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            f"0:{'a' if kind == 'audio' else 'v'}:0",
            "-threads",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        timeout=180,
        check=True,
    )
    stream = streams[0]
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": stream.get("sample_rate"),
        "channels": stream.get("channels"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "duration_seconds": float(metadata["format"].get("duration", 0)),
        "decoder": "ffmpeg",
    }


def aggregate_decodes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [row["duration_seconds"] for row in rows]
    return {
        "decoded": len(rows),
        "codecs": dict(Counter(str(row["codec"]) for row in rows)),
        "decoders": dict(Counter(str(row["decoder"]) for row in rows)),
        "fallback_original_codecs": dict(
            Counter(str(row["original_codec"]) for row in rows if "original_codec" in row)
        ),
        "finite_samples_verified": (
            sum(row.get("finite_samples_verified", False) for row in rows)
            if any("finite_samples_verified" in row for row in rows)
            else None
        ),
        "frames_decoded": (
            sum(row.get("frames_decoded", 0) for row in rows)
            if any("frames_decoded" in row for row in rows)
            else None
        ),
        "sample_rates": dict(
            Counter(str(row["sample_rate"]) for row in rows if row["sample_rate"] is not None)
        ),
        "channels": dict(
            Counter(str(row["channels"]) for row in rows if row["channels"] is not None)
        ),
        "video_dimensions": dict(
            Counter(f"{row['width']}x{row['height']}" for row in rows if row["width"] is not None)
        ),
        "duration_seconds": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "total": sum(durations),
            "above_30_seconds": sum(duration > 30 for duration in durations),
        },
        "decode_scope": "complete selected stream through EOF; decoder-specific checks reported",
    }


def preflight_audio(file: Any, audio_python: Path) -> dict[str, Any]:
    """Probe fixed real public rows before the full corpus scan."""
    decoded = []
    domains: Counter[str] = Counter()
    with (
        tempfile.TemporaryDirectory(prefix="mmau-public-preflight-") as temporary,
        audio_worker(audio_python) as decoder,
    ):
        # The fixed parquet has ten 100-row groups. These are rows 0, 333, 666.
        for group, offset in [(0, 0), (3, 33), (6, 66)]:
            for index, batch in enumerate(
                file.iter_batches(
                    batch_size=1,
                    row_groups=[group],
                    columns=["context", "other_attributes"],
                    use_threads=False,
                )
            ):
                if index != offset:
                    continue
                row = batch.to_pylist()[0]
                attributes = json.loads(row["other_attributes"])
                if attributes["split"] != "test-mini":
                    raise ValueError("preflight_non_public_split")
                audio = row["context"]["bytes"]
                if not audio or len(audio) > 64 * 1024**2:
                    raise ValueError("preflight_audio_size_limit")
                path = Path(temporary) / f"{group}.wav"
                path.write_bytes(audio)
                decoded.append(bounded_audio_decode(decoder, path, 0))
                domains[attributes["task"]] += 1
                path.unlink()
                break
            del batch
            importlib.import_module("pyarrow").default_memory_pool().release_unused()
    if len(decoded) != 3 or decoder.returncode != 0:
        raise ValueError("real_audio_preflight_incomplete")
    result = {"public_rows": [0, 333, 666], "domains": dict(domains), **aggregate_decodes(decoded)}
    print(json.dumps({"audio_preflight": result}), flush=True)
    return result


def validate_mmau(
    parquet: Path, source: Path, expected: dict[str, Any], audio_python: Path
) -> dict[str, Any]:
    if parquet.stat().st_size != expected["bytes"] or sha256(parquet) != expected["sha256"]:
        raise ValueError("mmau_source_hash_mismatch")
    if sha256(source) != MMAU_PUBLIC_JSON_SHA256:
        raise ValueError("public_mini_json_hash_mismatch")
    public = json.loads(source.read_text())
    by_id = {row["id"]: row for row in public}
    if len(public) != 1000 or len(by_id) != 1000:
        raise ValueError("public_mini_id_count_mismatch")
    parquet_lib = importlib.import_module("pyarrow.parquet")
    file = parquet_lib.ParquetFile(parquet)
    preflight = preflight_audio(file, audio_python)
    seen: set[str] = set()
    identities = []
    decoded: list[dict[str, Any]] = []
    fallback_count = 0
    columns = ["context", "instruction", "choices", "answer", "other_attributes"]
    with (
        tempfile.TemporaryDirectory(prefix="mmau-public-cpu-") as temporary,
        audio_worker(audio_python) as decoder,
    ):
        for group in range(file.metadata.num_row_groups):
            for batch in file.iter_batches(
                batch_size=1, row_groups=[group], columns=columns, use_threads=False
            ):
                paths = []
                for row in batch.to_pylist():
                    attributes = json.loads(row["other_attributes"])
                    identity = attributes["id"]
                    if identity in seen or identity not in by_id:
                        raise ValueError("duplicate_or_unexpected_mmau_id")
                    original = by_id[identity]
                    if attributes.get("split") != "test-mini":
                        raise ValueError("non_public_mini_split_denied")
                    if (
                        row["instruction"] != original["question"]
                        or row["choices"]
                        != [f"({chr(65 + i)}) {text}" for i, text in enumerate(original["choices"])]
                        or original["answer"] not in original["choices"]
                        or row["answer"]
                        != f"({chr(65 + original['choices'].index(original['answer']))}) "
                        + original["answer"]
                        or PurePosixPath(original["audio_id"]).stem != identity
                    ):
                        raise ValueError("public_mini_metadata_mapping_mismatch")
                    audio = row["context"]["bytes"]
                    if not audio or len(audio) > 64 * 1024**2:
                        raise ValueError("missing_or_oversize_embedded_audio")
                    path = Path(temporary) / f"{len(seen)}.wav"
                    path.write_bytes(audio)
                    paths.append(path)
                    identities.append((identity, hashlib.sha256(audio).hexdigest(), len(audio)))
                    seen.add(identity)
                for path in paths:
                    decoded_audio = bounded_audio_decode(decoder, path, fallback_count)
                    fallback_count += decoded_audio["decoder"] == "ffmpeg_then_libsndfile"
                    decoded.append(decoded_audio)
                for path in paths:
                    path.unlink()
            del batch
            importlib.import_module("pyarrow").default_memory_pool().release_unused()
            print(json.dumps({"mmau_audio_decoded": len(decoded), "row_group": group}), flush=True)
    if decoder.returncode != 0:
        raise ValueError("soundfile_worker_exit_nonzero")
    if seen != set(by_id):
        raise ValueError("mmau_public_id_coverage_incomplete")
    return {
        "status": "public_mini_mapping_and_all_audio_decode_verified",
        "real_audio_preflight": preflight,
        "parquet_sha256": expected["sha256"],
        "public_metadata_sha256": sha256(source),
        "schema": str(file.schema_arrow),
        "parquet_rows": file.metadata.num_rows,
        "public_ids_matched": len(seen),
        "public_audio_ids_unique": len({row["audio_id"] for row in public}),
        "embedded_audio_count": len(identities),
        "ffmpeg_fallback_count": fallback_count,
        "soundfile_worker_exit_code": decoder.returncode,
        "metadata_comparison": "HF choices/answer exactly equal public choices with (A) prefixes",
        "unique_audio_content_hashes": len({row[1] for row in identities}),
        "embedded_audio_total_bytes": sum(row[2] for row in identities),
        "id_audio_manifest_sha256": hashlib.sha256(
            json.dumps(sorted(identities), separators=(",", ":")).encode()
        ).hexdigest(),
        "null_context_path_count_expected": "HF embeds audio bytes; path is not required",
        "decode": aggregate_decodes(decoded),
        "soundfile_version": decoded[0]["soundfile_version"],
        "libsndfile_version": decoded[0]["libsndfile_version"],
        "training_admitted": False,
        "teacher_same_version_reproduction": "pending_teacher_dataset_revision",
    }


def validate_star(
    path: Path, source: Path, expected: dict[str, Any], tasks: list[dict[str, Any]]
) -> dict[str, Any]:
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError("star_source_hash_mismatch")
    coverage = []
    with zipfile.ZipFile(path) as archive:
        members = checked_zip_members(archive)
        if archive.testzip() is not None:
            raise ValueError("zip_crc_failed")
        for task in tasks:
            metadata = source / ("mvbench-" + task["file"].replace("/", "-"))
            if sha256(metadata) != task["sha256"]:
                raise ValueError("mvbench_metadata_hash_mismatch")
            rows = json.loads(metadata.read_text())
            prefix = task["source_prefix"].replace("your_data_path/", "")
            wanted = [prefix + row["video"] for row in rows]
            matched = [name for name in wanted if name in members]
            coverage.append(
                {
                    "task": task["task"],
                    "rows": len(rows),
                    "present_rows": len(matched),
                    "unique_media_references": len(set(wanted)),
                    "present_unique_media": len(set(matched)),
                    "coverage": len(matched) / len(rows),
                }
            )
        videos = sorted(name for name in members if name.endswith(".mp4"))
        selected = videos[:: max(1, len(videos) // 32)][:32]
        decoded: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="star-public-cpu-") as temporary:
            for index, name in enumerate(selected):
                target = Path(temporary) / f"{index}.mp4"
                with archive.open(members[name]) as stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output, length=1024**2)
                if target.stat().st_size != members[name].file_size:
                    raise ValueError("extracted_media_size_mismatch")
                decoded.append(probe_decode(target, "video"))
                target.unlink()
    return {
        "status": "archive_safety_crc_coverage_and_sample_decode_verified",
        "archive_sha256": expected["sha256"],
        "members": len(members),
        "expanded_bytes": sum(entry.file_size for entry in members.values()),
        "safety_scope": "all members checked before extraction; full archive CRC checked",
        "video_members": len(videos),
        "task_coverage": coverage,
        "source_tasks_coverage": {
            "expected_rows": sum(
                x["rows"]
                for x, task in zip(coverage, tasks, strict=True)
                if task["source_prefix"].startswith("your_data_path/star/")
            ),
            "present_rows": sum(x["present_rows"] for x in coverage),
            "complete": all(
                x["present_rows"] == x["rows"]
                for x, task in zip(coverage, tasks, strict=True)
                if task["source_prefix"].startswith("your_data_path/star/")
            ),
        },
        "sample_selection": "sorted names, stride floor(n/32), first 32; seed not needed",
        "sample_members_sha256": hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
        "sample_decode": aggregate_decodes(decoded),
        "all_video_members_decoded": len(selected) == len(videos),
        "training_admitted": False,
        "teacher_14_task_2786_reproduction": "blocked_missing_exact_task_row_manifest",
    }


def main() -> int:
    if sys.argv[1:] == ["--soundfile-worker"]:
        return soundfile_worker()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    result: dict[str, Any] = {
        "status": "failed",
        "exit_code": 1,
        "gpu_seconds": 0,
        "cost_usd": 0,
        "seed": 0,
        "script_sha256": sha256(Path(__file__)),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "environment": platform.platform(),
        "python_version": platform.python_version(),
        "training_admitted": False,
        "cgroup_limits": {
            name: path.read_text().strip() if path.is_file() else None
            for name, path in {
                "cpu_max": Path("/sys/fs/cgroup/cpu.max"),
                "memory_max": Path("/sys/fs/cgroup/memory.max"),
            }.items()
        },
    }
    try:
        result["config_sha256"] = sha256(args.config)
        config = json.loads(args.config.read_text())
        mmau = config["mmau"]
        result["mmau"] = validate_mmau(
            args.data_root / "mmau" / mmau["revision"] / mmau["file"]["path"],
            args.sources / "mmau-mini",
            mmau["file"],
            args.audio_python,
        )
        result["pyarrow_version"] = importlib.import_module("pyarrow").__version__
        result["ffmpeg_version"] = subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0]
        result["ffprobe_version"] = subprocess.check_output(
            ["ffprobe", "-version"], text=True
        ).splitlines()[0]
        result["status"] = "mmau_verified_star_check_running"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
        star = next(x for x in config["mvbench"]["archives"] if x["path"] == "video/star.zip")
        result["star"] = validate_star(
            args.data_root / "mvbench" / config["mvbench"]["revision"] / "star.zip",
            args.sources,
            star,
            config["mvbench"]["tasks"],
        )
        result.update(status="public_media_cpu_checks_completed", exit_code=0)
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, reason=str(error)[:400])
    result["elapsed_seconds"] = time.monotonic() - start
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "exit_code": result["exit_code"]}))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
