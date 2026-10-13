import binascii
import hashlib
import json
import struct
import zlib
from pathlib import Path

import pytest

from scripts.validate_gui_media import main


def chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data))
    )


PNG = (
    b"\x89PNG\r\n\x1a\n"
    + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    + chunk(b"IDAT", zlib.compress(b"\x00\x01\x02\x03"))
    + chunk(b"IEND", b"")
)


def fixture_paths(root: Path, payload: bytes = PNG) -> tuple[Path, Path, Path, Path]:
    manifest = root / "members.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "name": "screenshots/111_0.png",
                "method": 0,
                "crc32": binascii.crc32(payload) & 0xFFFFFFFF,
                "compressed_size": len(payload),
                "uncompressed_size": len(payload),
                "disk_start": 0,
                "local_header_offset": 0,
            }
        )
        + "\n"
    )
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "partition": "train",
                "selected_uncompressed_bytes": len(payload),
                "selected_episodes": [
                    {
                        "episode_id": "111",
                        "partition": "train",
                        "screenshot_count": 1,
                        "uncompressed_bytes": len(payload),
                    }
                ],
            }
        )
    )
    media = root / "media"
    (media / "screenshots").mkdir(parents=True)
    (media / "screenshots/111_0.png").write_bytes(payload)
    return manifest, plan, media, root / "report.json"


def run(paths: tuple[Path, Path, Path, Path]) -> tuple[int, dict]:
    manifest, plan, media, report = paths
    code = main(
        [
            "--manifest",
            str(manifest),
            "--plan",
            str(plan),
            "--media-root",
            str(media),
            "--report",
            str(report),
        ]
    )
    return code, json.loads(report.read_text())


def test_cli_fully_decodes_selected_png_and_preserves_media(tmp_path):
    pytest.importorskip("PIL.Image")
    paths = fixture_paths(tmp_path)
    code, report = run(paths)
    assert code == 0
    assert report["decoded_images"] == 1
    assert report["verified_bytes"] == len(PNG)
    assert report["test_media_read"] is False
    assert report["training_admitted"] is False
    assert report["manifest_sha256"] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    assert (paths[2] / "screenshots/111_0.png").read_bytes() == PNG
    with pytest.raises(SystemExit):
        run(paths)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "not_png", "symlink", "test"])
def test_cli_rejects_bad_media_or_test_plan_without_mutation(tmp_path, failure):
    if failure == "not_png":
        pytest.importorskip("PIL.Image")
    paths = fixture_paths(tmp_path, b"not an image" if failure == "not_png" else PNG)
    screenshot = paths[2] / "screenshots/111_0.png"
    if failure == "missing":
        screenshot.unlink()
    elif failure == "corrupt":
        screenshot.write_bytes(PNG[:-5])
    elif failure == "symlink":
        target = tmp_path / "original.png"
        screenshot.rename(target)
        screenshot.symlink_to(target)
    elif failure == "test":
        plan = json.loads(paths[1].read_text())
        plan["selected_episodes"][0]["partition"] = "test"
        paths[1].write_text(json.dumps(plan))
    before = screenshot.read_bytes() if screenshot.exists() else None
    code, report = run(paths)
    assert code == 1
    assert report["status"] == "failed"
    assert report["decoded_images"] == 0
    assert report["error_type"] == (
        "UnidentifiedImageError" if failure == "not_png" else "ValueError"
    )
    assert (screenshot.read_bytes() if screenshot.exists() else None) == before
