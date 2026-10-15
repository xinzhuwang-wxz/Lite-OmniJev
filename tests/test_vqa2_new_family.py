import json
import zipfile

import pytest

from data.schema import Sample
from scripts.s2_media_pools import canonical_sample_sha, sha256_file
from scripts.train_s1 import verify_data_is_train_pool
from scripts.vqa2_new_family import MEDIA_LICENSE, build, plan


def fixtures(tmp_path):
    args = {
        key: tmp_path / key
        for key in (
            "coco_metadata",
            "aok_train",
            "train_questions",
            "train_annotations",
            "output_dir",
        )
    }
    images = [
        {
            "id": i,
            "license": 4,
            "file_name": f"{i:012d}.jpg",
            "coco_url": f"http://images.cocodataset.org/train2017/{i:012d}.jpg",
            "flickr_url": f"https://flickr.com/photos/example/{i}",
        }
        for i in range(1, 9)
    ]
    images[5]["coco_url"] = "http://images.cocodataset.org/val2017/000000000006.jpg"
    args["coco_metadata"].write_text(
        json.dumps({"licenses": [{"id": 4, "url": MEDIA_LICENSE}], "images": images})
    )
    args["aok_train"].write_text(json.dumps([{"image_id": 1, "answers": "MUST NOT READ"}]))
    questions, annotations = [], []
    for i, answer in enumerate([None, "red", "3", "yes", "no", None, None, "yes"], 1):
        questions.append(
            {"image_id": i, "question_id": i, "question": "What color?" if i == 2 else "Question?"}
        )
        annotations.append(
            {
                "image_id": i,
                "question_id": i,
                "answers": [{"answer": answer}] * 10 if answer else None,
                "multiple_choice_answer": answer,
            }
        )
    annotations[-1]["answers"][-1] = {"answer": "no"}
    for key, rows, member in (
        ("train_questions", questions, "v2_OpenEnded_mscoco_train2014_questions.json"),
        ("train_annotations", annotations, "v2_mscoco_train2014_annotations.json"),
    ):
        with zipfile.ZipFile(args[key], "w") as z:
            z.writestr(
                member,
                json.dumps(
                    {
                        "license": {"url": "http://creativecommons.org/licenses/by/4.0/"},
                        key.removeprefix("train_"): rows,
                    }
                ),
            )
    ledger = tmp_path / "excluded.jsonl"
    ledger.write_text(json.dumps({"family_id": "coco:7", "pool": "holdout"}) + "\n")
    args["allocation_ledger"] = [ledger]
    return args


def test_plan_exclusions_consensus_determinism(tmp_path):
    args = fixtures(tmp_path)
    first = plan(**args)
    second = plan(**{**args, "output_dir": tmp_path / "again"})
    assert first == second
    rows = [
        json.loads(s) for s in (args["output_dir"] / "candidates.jsonl").read_text().splitlines()
    ]
    assert {r["image_id"] for r in rows} == {2, 3, 4, 5}
    assert first["category_counts"] == {"color": 1, "integer": 1, "yes_no": 2}
    assert all(r["gold"] in r["options"] for r in rows)
    with pytest.raises(ValueError, match="overwrite"):
        plan(**args)


def manifest(tmp_path, args):
    image_module = pytest.importorskip("PIL.Image")
    rows = []
    for candidate in (args["output_dir"] / "candidates.jsonl").read_text().splitlines():
        row = json.loads(candidate)
        path = tmp_path / row["file_name"]
        image_module.new("RGB", (3, 3), (row["image_id"] * 30, 0, 0)).save(path)
        rows.append(
            {
                "image_id": row["image_id"],
                "url": row["url"],
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    result = tmp_path / "downloads.jsonl"
    result.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return result


def test_build_gold_ledger_and_hashes(tmp_path):
    args = fixtures(tmp_path)
    plan(**args)
    downloads = manifest(tmp_path, args)
    out = tmp_path / "build"
    result = build(
        media_root=tmp_path,
        plan_dir=args["output_dir"],
        download_manifest=downloads,
        output_dir=out,
    )
    assert (
        verify_data_is_train_pool(out / "train.jsonl", [out / "allocation-ledger.jsonl"])[
            "verified_train"
        ]
        == 4
    )
    samples = [
        Sample.model_validate_json(s) for s in (out / "train.jsonl").read_text().splitlines()
    ]
    ledger = [json.loads(s) for s in (out / "allocation-ledger.jsonl").read_text().splitlines()]
    for sample, allocation in zip(samples, ledger, strict=True):
        assert allocation["derived_sample_sha256"] == canonical_sample_sha(sample)
        q = sample.questions[0]
        assert (
            q.options[ord(q.target.value) - 65]
            == {"coco:2": "red", "coco:3": "3", "coco:4": "yes", "coco:5": "no"}[
                sample.meta.family_id
            ]
        )
    again = build(
        media_root=tmp_path,
        plan_dir=args["output_dir"],
        download_manifest=downloads,
        output_dir=tmp_path / "build2",
    )
    assert result == again


@pytest.mark.parametrize("failure", ["hash", "existing_media", "corrupt", "source", "candidate"])
def test_build_fails_closed(tmp_path, failure):
    args = fixtures(tmp_path)
    plan(**args)
    downloads = manifest(tmp_path, args)
    records = [json.loads(s) for s in downloads.read_text().splitlines()]
    if failure == "hash":
        records[0]["sha256"] = "0" * 64
    elif failure == "corrupt":
        from pathlib import Path

        Path(records[0]["path"]).write_bytes(b"not an image")
        records[0]["sha256"] = sha256_file(Path(records[0]["path"]))
    elif failure == "existing_media":
        ledger = args["allocation_ledger"][0]
        ledger.write_text(
            ledger.read_text() + json.dumps({"media_sha256": records[0]["sha256"]}) + "\n"
        )
        args["output_dir"] = tmp_path / "fresh_plan"
        plan(**args)
    elif failure == "source":
        args["aok_train"].write_text("[]")
    else:
        p = args["output_dir"] / "candidates.jsonl"
        p.write_text(p.read_text() + "\n")
    downloads.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises((ValueError, OSError)):
        build(
            media_root=tmp_path,
            plan_dir=args["output_dir"],
            download_manifest=downloads,
            output_dir=tmp_path / "build",
        )
    assert not (tmp_path / "build").exists()


def test_unknown_media_license(tmp_path):
    args = fixtures(tmp_path)
    p = args["coco_metadata"]
    p.write_text(p.read_text().replace(MEDIA_LICENSE, "https://example.com/unknown"))
    with pytest.raises(ValueError, match="license"):
        plan(**args)
