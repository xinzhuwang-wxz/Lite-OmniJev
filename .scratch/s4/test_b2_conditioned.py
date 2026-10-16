"""Fixture self-test for the B2 conditioning builder."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.schema import Sample  # noqa: E402

tmp = Path(tempfile.mkdtemp())
caption = {
    "id": "v1",
    "family_id": "clevrer:1",
    "media_sha256": "a" * 64,
    "prompt_sha256": "b" * 64,
    "raw_text_sha256": "c" * 64,
    "summary_valid": True,
    "summary": {
        "objects": [{"name": "red cube", "first_seen": "start", "last_seen": "end"}],
        "entry_order": [],
        "collisions": [],
        "present_at_start": ["red cube"],
        "present_at_end": ["red cube"],
    },
}
bad = {"id": "v2", "summary_valid": False, "summary": None}
(tmp / "cap.jsonl").write_text(json.dumps(caption) + "\n" + json.dumps(bad) + "\n")
src = {
    "id": "v1",
    "prefix": {
        "state": "Answer the yes/no question using the attached video.",
        "media": [{"type": "video", "uri": "video/v.mp4", "sha256": "a" * 64}],
    },
    "questions": [
        {
            "qid": "noul",
            "type": "noul",
            "instruction": "Q?",
            "options": None,
            "score_levels": None,
            "score_descriptions": None,
            "target": {"kind": "hard", "value": "yes", "source": "s"},
            "pair_of": None,
        }
    ],
    "meta": {
        "modality": ["video"],
        "source_dataset": "CLEVRER",
        "source_revision": "r",
        "source_sha256": "d" * 64,
        "license": "CC0-1.0",
        "usage": "commercial",
        "pool": "train",
        "family_id": "clevrer:1",
        "data_version": "clevrer-train-video-noul-s3-pools-v1",
    },
}
src2 = json.loads(json.dumps(src))
src2["id"] = "v2"
(tmp / "src.jsonl").write_text(json.dumps(src) + "\n" + json.dumps(src2) + "\n")
for extra, name in ([], "withvideo"), (["--drop-video"], "summaryonly"):
    out = tmp / f"cond-{name}.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(__file__).with_name("build_b2_conditioned.py")),
            "--captions",
            str(tmp / "cap.jsonl"),
            "--source",
            str(tmp / "src.jsonl"),
            "--output",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    print(name, result.stdout.strip())
    row = json.loads(out.read_text().splitlines()[0])
    print("  media kept:", len(row["prefix"]["media"]), "version:", row["meta"]["data_version"])
    print("  state head:", row["prefix"]["state"][:70])
sample = Sample.model_validate(json.loads((tmp / "cond-withvideo.jsonl").read_text().splitlines()[0]))
sample_only = Sample.model_validate(
    json.loads((tmp / "cond-summaryonly.jsonl").read_text().splitlines()[0])
)
assert len(sample.prefix.media) == 1 and len(sample_only.prefix.media) == 0
assert "Perception summary" in sample.prefix.state
print("pydantic valid both variants: OK")
