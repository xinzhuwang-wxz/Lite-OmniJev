"""Fixed-scope CPU evidence driver; no downloads or source archive modification."""

import json
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from public_eval_media import checked_zip_members, probe_decode, sha256

BASE = Path(__file__).resolve().parent
PROJECT = Path("/root/autodl-tmp/lite-omnijev")
CONFIG = BASE / "video-eval-assets.json"
OUTPUT = BASE / "public-eval-archives-validation-002.json"
config = json.loads(CONFIG.read_text())
revision = config["mvbench"]["revision"]
archives = PROJECT / "artifacts/data/research-eval/mvbench" / revision
metadata = PROJECT / "artifacts/data/research-eval/video-eval-source-metadata"
started = time.monotonic()
report = {
    "started_utc": datetime.now(UTC).isoformat(),
    "code_revision": "bae53385a5e0d1a6a27bb7c6aaf6f7fb4f2097aa",
    "driver_sha256": sha256(Path(__file__)),
    "helper_sha256": sha256(BASE / "public_eval_media.py"),
    "config_sha256": sha256(CONFIG),
    "mvbench_revision": revision,
    "platform": platform.platform(),
    "python": platform.python_version(),
    "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
    "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
    "heavy_lock": "/tmp/omnijev-cpu-heavy.lock",
    "decoder_threads": 1,
    "training_admitted": False,
    "commercial_release_admitted": False,
    "teacher_exact_14_tasks_2786_ids": "pending_authoritative_manifest",
    "archives": [],
    "exit_code": None,
}


def save():
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")


try:
    for name, prefix in [
        ("perception.zip", "perception/videos/"),
        ("FunQA_test.zip", "FunQA_test/test/"),
        ("sta.zip", "sta/sta_video/"),
    ]:
        entry = next(a for a in config["mvbench"]["archives"] if a["path"] == "video/" + name)
        path = archives / name
        actual_sha = sha256(path)
        if path.stat().st_size != entry["bytes"] or actual_sha != entry["sha256"]:
            raise ValueError("archive_size_or_sha_mismatch:" + name)
        item = {
            "file": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "tasks": [],
            "sample_decodes": [],
        }
        report["archives"].append(item)
        with zipfile.ZipFile(path) as archive:
            members = checked_zip_members(archive, max_expanded_bytes=16 * 1024**3)
            item.update(
                member_count=len(members),
                expanded_bytes=sum(m.file_size for m in members.values()),
                safe_members=True,
            )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError("archive_crc_failed:" + bad_member)
            item["all_member_crc_verified"] = True
            samples = set()
            for task in config["mvbench"]["tasks"]:
                if task["source_prefix"].removeprefix("your_data_path/") != prefix:
                    continue
                source = metadata / ("mvbench-json-" + Path(task["file"]).name)
                if sha256(source) != task["sha256"]:
                    raise ValueError("public_task_sha_mismatch")
                rows = json.loads(source.read_text())
                present = [
                    prefix + row["video"] for row in rows if prefix + row["video"] in members
                ]
                missing = [
                    {"row_index": i, "video": row["video"]}
                    for i, row in enumerate(rows)
                    if prefix + row["video"] not in members
                ]
                item["tasks"].append(
                    {
                        "task": task["task"],
                        "public_json_sha256": task["sha256"],
                        "rows": len(rows),
                        "present_rows": len(present),
                        "missing": missing,
                    }
                )
                if present:
                    samples.update(present[i] for i in [0, len(present) // 2, len(present) - 1])
            videos = sorted(
                (m for m in members.values() if m.filename.endswith(".mp4")),
                key=lambda m: (m.file_size, m.filename),
            )
            samples.update(videos[i].filename for i in [0, len(videos) // 2, len(videos) - 1])
            item["video_members"] = len(videos)
            item["sample_selection"] = (
                "first/middle/last public row per task plus "
                "smallest/median/largest archive video by size"
            )
            with tempfile.TemporaryDirectory(prefix="public-eval-002-", dir=BASE) as temporary:
                for index, member in enumerate(sorted(samples)):
                    extracted = Path(temporary) / f"{index}.mp4"
                    with archive.open(member) as source, extracted.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024**2)
                    if extracted.stat().st_size != members[member].file_size:
                        raise ValueError("extracted_size_mismatch")
                    decoded = probe_decode(extracted, "video")
                    item["sample_decodes"].append(
                        {
                            "member": member,
                            "bytes": extracted.stat().st_size,
                            "sha256": sha256(extracted),
                            **decoded,
                        }
                    )
                    extracted.unlink()
            item["full_corpus_decodability_verified"] = False
        save()
        print(name, "CRC + coverage + sampled full-stream decode complete", flush=True)
    report["exit_code"] = 0
except Exception as error:
    report["exit_code"] = 1
    report["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
    raise
finally:
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["completed_utc"] = datetime.now(UTC).isoformat()
    save()
