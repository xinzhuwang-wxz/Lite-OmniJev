"""Reconstruct the small public CPU cohort from previously verified source assets."""

import copy
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

base = Path("/root/autodl-tmp/lite-omnijev")
out = base / "artifacts/s0-prepare-public-001"
out.mkdir(exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


clotho = base / "artifacts/data/quarantine/clotho-aqa/6473207"
with (clotho / "clotho_aqa_metadata.csv").open() as f:
    rows = list(csv.DictReader(f))
print("metadata_columns", list(rows[0]))
selected = next(
    r
    for r in rows
    if "creativecommons.org/publicdomain/zero/" in str(r)
    and (clotho / "approved-audio-002" / r["file_name"]).is_file()
)
shutil.copyfile(clotho / "approved-audio-002" / selected["file_name"], out / "audio.wav")
gui = (
    base
    / "artifacts/data/quarantine/gui-odyssey-source-audit/media"
    / "app_split_train_under19_5gb/screenshots"
)
shutil.copyfile(sorted(gui.glob("*.png"))[0], out / "screenshot.png")
archive = base / "source-audit/clevrer-official/files/videos/train/video_train.zip"
with zipfile.ZipFile(archive) as z:
    with (
        z.open("video_00000-01000/video_00000.mp4") as source,
        (out / "video.mp4").open("wb") as dest,
    ):
        shutil.copyfileobj(source, dest)
subprocess.run(
    [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(out / "video.mp4"),
        "-frames:v",
        "1",
        "-y",
        str(out / "image.png"),
    ],
    check=True,
)
manifest = []
for category, kind, filename, source, revision, license, family in [
    ("text", None, None, "programmatic-arithmetic", "v1", "CC0-1.0", "arithmetic"),
    (
        "image",
        "image",
        "image.png",
        "https://data.csail.mit.edu/clevrer/videos/train/video_train.zip",
        "archive-sha256:c7401bc86f7ff230bda3b193619f41a79828bed1f0890948b16dd57c89dfdd3f",
        "CC0-1.0",
        "clevrer-train-00000",
    ),
    (
        "screenshot",
        "image",
        "screenshot.png",
        "hflqf88888/GUIOdyssey",
        "61632d0f3f4d51d7e9561ce4f84347dd06b2019d",
        "CC-BY-4.0",
        "gui-train-" + sorted(gui.glob("*.png"))[0].stem.split("_")[0],
    ),
    (
        "audio",
        "audio",
        "audio.wav",
        "https://zenodo.org/records/6473207",
        "6473207",
        "CC0-1.0",
        "clotho-" + selected["file_name"],
    ),
    (
        "video",
        "video",
        "video.mp4",
        "https://data.csail.mit.edu/clevrer/videos/train/video_train.zip",
        "archive-sha256:c7401bc86f7ff230bda3b193619f41a79828bed1f0890948b16dd57c89dfdd3f",
        "CC0-1.0",
        "clevrer-train-00000",
    ),
]:
    question = {
        "qid": "prepare-" + category,
        "type": "noul",
        "instruction": "Is there sufficient evidence to decide whether a red object is present?",
    }
    if category == "text":
        question = {
            "qid": "prepare-text",
            "type": "choice",
            "instruction": "What is 2 + 2?",
            "options": ["4", "5"],
        }
    if category == "video":
        question = {
            "qid": "prepare-video",
            "type": "score",
            "instruction": "Rate the visibility of objects.",
            "score_levels": 3,
        }
    prefix = {"state": "CPU input preparation only. No truth labels assigned."}
    if filename:
        prefix["media"] = [{"type": kind, "uri": filename, "sha256": sha(out / filename)}]
    else:
        prefix["state"] = "2 + 2 = 4"
    manifest.append(
        {
            "id": "public-prepare-" + category,
            "category": category,
            "prefix": prefix,
            "question": question,
            "meta": {
                "modality": [kind or "text"],
                "source_dataset": source,
                "source_revision": revision,
                "source_sha256": sha(out / filename) if filename else sha(Path(__file__)),
                "license": license,
                "usage": "research",
                "pool": "dev",
                "family_id": family,
                "data_version": "s0-cpu-public-001",
            },
        }
    )
expanded = []
for row in manifest:
    for index in range(20):
        case = copy.deepcopy(row)
        case["id"] += f"-{index:02d}"
        case["question"]["qid"] += f"-{index:02d}"
        case["prefix"]["state"] += f" Public fixed preparation variant {index:02d}."
        shapes = ["sphere", "cube", "cylinder", "object"]
        colors = ["red", "blue", "green", "yellow", "purple"]
        color = colors[index % 5]
        shape = shapes[index // 5]
        if row["category"] == "text":
            case["prefix"]["state"] = f"{index} + 2 = {index + 2}"
            case["question"]["instruction"] = f"What is {index} + 2?"
            case["question"]["options"] = [str(index + 2), str(index + 3)]
        elif row["category"] == "audio":
            sounds = [
                "speech",
                "birdsong",
                "water",
                "vehicle",
                "footsteps",
                "wind",
                "music",
                "laughter",
                "rain",
                "bells",
                "barking",
                "clapping",
                "machinery",
                "insects",
                "waves",
                "breathing",
                "doors",
                "rustling",
                "typing",
                "whistling",
            ]
            case["question"]["instruction"] = f"Is {sounds[index]} audible in this recording?"
        elif row["category"] == "screenshot":
            case["question"]["instruction"] = (
                f"Is there a visible control numbered {index + 1} in the screenshot?"
            )
        elif row["category"] == "image":
            case["question"]["instruction"] = f"Is a {color} {shape} visible in the image?"
        elif row["category"] == "video":
            case["question"]["instruction"] = (
                f"Rate the visibility of a {color} {shape} throughout the video."
            )
            case["question"]["score_descriptions"] = [
                "not visible",
                "partly visible",
                "clearly visible",
            ]
        case["video_frames"] = 16 if row["category"] == "video" and index >= 10 else 8
        expanded.append(case)
(out / "manifest.json").write_text(json.dumps(expanded, indent=2))
(out / "fixture-provenance.json").write_text(
    json.dumps(
        {
            "audio_metadata_sha256": sha(clotho / "clotho_aqa_metadata.csv"),
            "audio_source_file": selected["file_name"],
            "audio_license": selected,
            "image_derivation": "first frame of CLEVRER train video_00000.mp4, ffmpeg RGB PNG",
            "screenshot_source_file": str(sorted(gui.glob("*.png"))[0].relative_to(base)),
            "truth_labels_assigned": False,
        },
        indent=2,
    )
)
