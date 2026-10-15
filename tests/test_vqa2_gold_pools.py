"""Synthetic-only gold builder canaries."""

import json
from pathlib import Path
from typing import Any

import pytest
from test_vqa2_gold_capacity import fixture, rewrite

from scripts.gold_inventory import digest
from scripts.vqa2_gold_pools import (
    choose,
    download_transport,
    main,
    materialize,
    official_url,
    plan,
    rgb,
)


def sources(tmp_path: Path) -> dict[str, Any]:
    config = fixture(tmp_path)
    rewrite(
        config,
        "coco_metadata",
        lambda p: p["images"].extend(
            {"id": i, "license": 4, "file_name": f"{i:012d}.jpg", "coco_url": official_url(i)}
            for i in range(7, 22)
        ),
    )

    def questions(p: dict[str, Any]) -> None:
        p["questions"].extend(
            {"image_id": i, "question_id": i * 10 + j, "question": f"color {j}?"}
            for i in range(7, 22)
            for j in range(3)
        )

    def annotations(p: dict[str, Any]) -> None:
        p["annotations"].extend(
            {
                "image_id": i,
                "question_id": i * 10 + j,
                "multiple_choice_answer": answer,
                "answers": [{"answer": answer}] * 10,
            }
            for i in range(7, 22)
            for j, answer in enumerate(("yes", "2", "red"))
        )

    rewrite(config, "train_questions", questions)
    rewrite(config, "train_annotations", annotations)
    return config


def test_deterministic_fractions_and_family_uniqueness(tmp_path: Path) -> None:
    config = sources(tmp_path)
    quotas = dict.fromkeys(("calibration", "holdout", "dev"), 5)
    rows, _ = choose(config, quotas, True)
    assert rows == choose(config, quotas, True)[0]
    assert len({r["image_id"] for r in rows}) == 15
    assert not {1, 2, 3} & {r["image_id"] for r in rows}
    for pool in quotas:
        selected = [r for r in rows if r["pool"] == pool]
        assert [r["category"] for r in selected].count("yes_no") == 3
        assert [r["category"] for r in selected].count("integer") == 1
        assert [r["category"] for r in selected].count("color") == 1
    assert all(r["gold"] in r["options"] and len(r["options"]) in (2, 4) for r in rows)


def test_plan_nooverwrite_pins_and_redaction(tmp_path: Path, capsys: Any) -> None:
    config = sources(tmp_path)
    path = tmp_path / "config"
    path.write_text(json.dumps(config))
    output, report = tmp_path / "private", tmp_path / "receipt"
    args = [
        "plan",
        "--config",
        str(path),
        "--output-dir",
        str(output),
        "--report",
        str(report),
        "--synthetic-canary",
        "--calibration-count",
        "5",
        "--holdout-count",
        "5",
        "--dev-count",
        "5",
    ]
    assert main(args) == 0
    receipt = capsys.readouterr().out
    assert str(tmp_path) not in receipt and "question_id" not in receipt
    assert output.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.iterdir())
    before = report.read_bytes()
    assert main(args) == 1
    assert report.read_bytes() == before
    with pytest.raises(ValueError):
        plan(config, tmp_path / "real")


def test_source_mutation_rejected(tmp_path: Path) -> None:
    config = sources(tmp_path)
    output = tmp_path / "plan"
    result = plan(
        config, output, canary=True, quotas=dict.fromkeys(("calibration", "holdout", "dev"), 5)
    )
    Path(config["train_questions"]["path"]).write_bytes(b"changed")
    download = tmp_path / "download"
    download.write_bytes(b"")
    with pytest.raises(ValueError):
        materialize(
            output / "plan.json",
            result["plan_sha256"],
            download,
            digest(b""),
            tmp_path,
            tmp_path / "built",
        )


def test_rgb_decode_hash_and_alias(tmp_path: Path) -> None:
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / "image.jpg"
    image.new("RGB", (2, 2), "red").save(path)
    sha = digest(path.read_bytes())
    assert len(rgb(path, sha)) == 64
    alias = tmp_path / "alias.jpg"
    alias.write_bytes(path.read_bytes() + b"private-metadata")
    assert digest(alias.read_bytes()) != sha
    assert rgb(alias, digest(alias.read_bytes())) == rgb(path, sha)
    with pytest.raises(ValueError):
        rgb(path, "0" * 64)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"bad")
    with pytest.raises(OSError):
        rgb(bad, digest(b"bad"))


def test_nonunanimous_conflicts_and_url_rejection(tmp_path: Path) -> None:
    config = sources(tmp_path)
    rewrite(
        config,
        "train_annotations",
        lambda p: p["annotations"][9].update(answers=[{"answer": "yes"}] * 9 + [{"answer": "no"}]),
    )
    rewrite(
        config,
        "train_questions",
        lambda p: p["questions"][13].update(question=p["questions"][12]["question"].upper()),
    )
    rewrite(
        config,
        "coco_metadata",
        lambda p: p["images"][5].update(coco_url="https://flickr.com/private"),
    )
    rows, _ = choose(config, dict.fromkeys(("calibration", "holdout", "dev"), 5), True)
    assert all(not (r["image_id"] == 4 and r["category"] == "yes_no") for r in rows)
    assert all(r["image_id"] != 6 for r in rows)
    assert all(r["url"] == official_url(r["image_id"]) for r in rows)


def test_materialize_bad_manifest_and_path(tmp_path: Path) -> None:
    config = sources(tmp_path)
    output = tmp_path / "plan"
    result = plan(
        config, output, canary=True, quotas=dict.fromkeys(("calibration", "holdout", "dev"), 5)
    )
    download = tmp_path / "downloads"
    download.write_bytes(b"\n")
    with pytest.raises(ValueError):
        materialize(
            output / "plan.json",
            result["plan_sha256"],
            download,
            "0" * 64,
            tmp_path,
            tmp_path / "built",
        )
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        plan(config, linked / "output", canary=True)


def test_materialize_preserves_inventory_and_rejects_duplicates(tmp_path: Path) -> None:
    image = pytest.importorskip("PIL.Image")
    config = sources(tmp_path)
    inventory_path = Path(config["inventory"]["path"])
    inventory = json.loads(inventory_path.read_bytes())
    history_path = Path(inventory["samples"][0]["path"])
    history = json.loads(history_path.read_text())
    old_image = tmp_path / "old.png"
    image.new("RGB", (4, 4), "black").save(old_image)
    history["prefix"]["media"][0].update(uri="old.png", sha256=digest(old_image.read_bytes()))
    history_path.write_text(json.dumps(history) + "\n")
    inventory["samples"][0]["sha256"] = digest(history_path.read_bytes())
    inventory_path.write_text(json.dumps(inventory))
    config["inventory"]["sha256"] = digest(inventory_path.read_bytes())
    output = tmp_path / "plan"
    result = plan(
        config, output, canary=True, quotas=dict.fromkeys(("calibration", "holdout", "dev"), 5)
    )
    rows = [json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()]
    downloads = []
    for i, row in enumerate(rows):
        path = tmp_path / f"asset-{i}.jpg"
        image.new("RGB", (4, 4), (i * 13 + 1, 40, 90)).save(path)
        downloads.append(
            {
                "image_id": row["image_id"],
                "path": str(path),
                "url": row["url"],
                "sha256": digest(path.read_bytes()),
            }
        )
    manifest = tmp_path / "downloads"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in downloads))
    built = tmp_path / "built"
    receipt = materialize(
        output / "plan.json",
        result["plan_sha256"],
        manifest,
        digest(manifest.read_bytes()),
        tmp_path,
        built,
    )
    assert receipt["counts"] == {"calibration": 5, "holdout": 5, "dev": 5}
    combined = json.loads((built / "inventory-manifest.json").read_bytes())
    assert combined["samples"][: len(inventory["samples"])] == inventory["samples"]
    for pool in receipt["counts"]:
        sample = json.loads((built / f"{pool}.jsonl").read_text().splitlines()[0])
        assert not sample["prefix"]["media"][0]["uri"].startswith("file:")
    historical_jpeg = tmp_path / "same-pixels.jpg"
    image.new("RGB", (4, 4), "black").save(historical_jpeg)
    assert rgb(old_image, digest(old_image.read_bytes()), historical=True) == rgb(
        historical_jpeg, digest(historical_jpeg.read_bytes())
    )
    with pytest.raises(ValueError, match="media_format"):
        rgb(old_image, digest(old_image.read_bytes()))
    original_download = dict(downloads[0])
    for label, payload in (
        ("history", old_image.read_bytes()),
        ("history-cross-format-alias", historical_jpeg.read_bytes()),
        ("rgb-alias", Path(downloads[1]["path"]).read_bytes() + b"metadata"),
        ("invalid-jpeg", b"broken"),
    ):
        alias_path = tmp_path / f"{label}.jpg"
        alias_path.write_bytes(payload)
        downloads[0].update(path=str(alias_path), sha256=digest(payload))
        manifest.write_text("".join(json.dumps(row) + "\n" for row in downloads))
        failed = materialize(
            output / "plan.json",
            result["plan_sha256"],
            manifest,
            digest(manifest.read_bytes()),
            tmp_path,
            tmp_path / label,
        )
        assert failed["exit_code"] == 1
        assert not (tmp_path / label).exists()
        downloads[0] = dict(original_download)
    downloads[0]["final_url"] = "https://flickr.com/private"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in downloads))
    redirect = materialize(
        output / "plan.json",
        result["plan_sha256"],
        manifest,
        digest(manifest.read_bytes()),
        tmp_path,
        tmp_path / "redirect",
    )
    assert redirect["exit_code"] == 1
    downloads[0] = original_download
    downloads[0]["path"] = downloads[1]["path"]
    downloads[0]["sha256"] = downloads[1]["sha256"]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in downloads))
    rejected = materialize(
        output / "plan.json",
        result["plan_sha256"],
        manifest,
        digest(manifest.read_bytes()),
        tmp_path,
        tmp_path / "rejected",
    )
    assert rejected["exit_code"] == 1
    assert not (tmp_path / "rejected").exists()


def duplicate_originals(config: dict[str, Any]) -> None:
    """Multiple distinct original questions per image/category; not normalized aliases."""

    def questions(payload: dict[str, Any]) -> None:
        original = list(payload["questions"])
        payload["questions"].extend(
            {
                **row,
                "question_id": int(row["question_id"]) + 10000 * n,
                "question": row["question"] + f" distinct original {n}",
            }
            for n in range(1, 6)
            for row in original
        )

    def annotations(payload: dict[str, Any]) -> None:
        original = list(payload["annotations"])
        payload["annotations"].extend(
            {**row, "question_id": int(row["question_id"]) + 10000 * n}
            for n in range(1, 6)
            for row in original
        )

    rewrite(config, "train_questions", questions)
    rewrite(config, "train_annotations", annotations)


def test_multiple_originals_allocate_unique_image_families(tmp_path: Path) -> None:
    config = sources(tmp_path)
    duplicate_originals(config)
    rows, _ = choose(config, dict.fromkeys(("calibration", "holdout", "dev"), 5), True)
    assert len(rows) == len({r["image_id"] for r in rows}) == 15
    assert rows == choose(config, dict.fromkeys(("calibration", "holdout", "dev"), 5), True)[0]
    for pool in ("calibration", "holdout", "dev"):
        assert len([r for r in rows if r["pool"] == pool]) == 5


def test_original_question_count_cannot_manufacture_family_capacity(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    duplicate_originals(config)
    # 54 eligible original questions, only three unused image families.
    with pytest.raises(ValueError, match="insufficient_capacity"):
        plan(
            config,
            tmp_path / "rejected",
            canary=True,
            quotas=dict.fromkeys(("calibration", "holdout", "dev"), 5),
        )
    assert not (tmp_path / "rejected").exists()


def test_exact_pinned_http_transport_only() -> None:
    https = official_url(42)
    http = https.replace("https:", "http:")
    row = {"image_id": 42, "attribution": {"coco_url": http}}
    assert download_transport(row, {"url": https}) == https
    assert download_transport(row, {"url": https, "transport_url": http, "final_url": http}) == http
    for transport in (
        "http://evil.example/train2017/000000000042.jpg",
        http + "?query=1",
        http.replace(".org/", ".org:80/"),
        http.replace("000000000042", "000000000043"),
    ):
        with pytest.raises(ValueError, match="download_url"):
            download_transport(row, {"url": https, "transport_url": transport})
    for final in (https, "https://flickr.com/private", http + "?query=1"):
        with pytest.raises(ValueError, match="download_url"):
            download_transport(row, {"url": https, "transport_url": http, "final_url": final})
    with pytest.raises(ValueError, match="download_url"):
        download_transport(
            {"image_id": 42, "attribution": {"coco_url": https}},
            {"url": https, "transport_url": http},
        )
    with pytest.raises(ValueError, match="download_url"):
        download_transport(row, {"url": https, "final_url": http})
