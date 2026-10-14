"""Public input boundaries; these tests are not real-processor acceptance."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.s0_prepare import main, read_manifest
from train.multimodal import PreparationCase, file_hash, resolve_media


def public_case() -> dict:
    return {
        "id": "public-text",
        "category": "text",
        "prefix": {"state": "2 + 2 = 4"},
        "question": {"qid": "q", "type": "noul", "instruction": "Is the statement true?"},
        "meta": {
            "modality": ["text"],
            "source_dataset": "programmatic-arithmetic",
            "source_revision": "v1",
            "source_sha256": "a" * 64,
            "license": "CC0-1.0",
            "usage": "commercial",
            "pool": "dev",
            "family_id": "arithmetic-1",
            "data_version": "v1",
        },
    }


def test_holdout_and_empty_are_rejected() -> None:
    row = public_case()
    row["meta"]["pool"] = "holdout"
    with pytest.raises(ValidationError, match="holdout"):
        PreparationCase.model_validate(row)
    row = public_case()
    row["prefix"]["state"] = " "
    with pytest.raises(ValidationError, match="at least 1 character|empty"):
        PreparationCase.model_validate(row)


@pytest.mark.parametrize("uri", ["https://example.com/a", "file:a", "../a", "/tmp/a"])
def test_resolver_rejects_nonlocal_paths(tmp_path: Path, uri: str) -> None:
    with pytest.raises(ValueError, match="relative"):
        resolve_media(tmp_path, uri, "a" * 64, max_bytes=20)


def test_resolver_rejects_links_hash_empty_and_limits(tmp_path: Path) -> None:
    asset = tmp_path / "asset"
    asset.write_bytes(b"real local bytes")
    digest = file_hash(asset)
    assert resolve_media(tmp_path, "asset", digest, max_bytes=100) == asset
    with pytest.raises(ValueError, match="SHA256"):
        resolve_media(tmp_path, "asset", "a" * 64, max_bytes=100)
    with pytest.raises(ValueError, match="limit"):
        resolve_media(tmp_path, "asset", digest, max_bytes=1)
    (tmp_path / "link").symlink_to(asset)
    with pytest.raises(ValueError, match="symlinks"):
        resolve_media(tmp_path, "link", digest, max_bytes=100)
    asset.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        resolve_media(tmp_path, "asset", digest, max_bytes=100)


def test_screenshot_is_image_with_provenance() -> None:
    row = public_case()
    row.update(category="screenshot")
    row["meta"]["modality"] = ["image"]
    row["prefix"]["media"] = [{"type": "image", "uri": "app.png", "sha256": "b" * 64}]
    assert PreparationCase.model_validate(row).category == "screenshot"
    row["prefix"]["media"][0]["type"] = "video"
    with pytest.raises(ValidationError, match="agree"):
        PreparationCase.model_validate(row)


def test_manifest_requires_explicit_partial_and_unique_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([public_case()]))
    with pytest.raises(ValueError, match="100"):
        read_manifest(path)
    assert len(read_manifest(path, allow_partial=True)) == 1
    path.write_text(json.dumps([public_case(), public_case()]))
    with pytest.raises(ValueError, match="duplicate"):
        read_manifest(path, allow_partial=True)


def test_cli_failure_keeps_private_content_out_of_report(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    row = public_case()
    row["prefix"]["state"] = "PRIVATE INPUT MUST NOT APPEAR"
    row["meta"]["pool"] = "holdout"
    manifest.write_text(json.dumps([row]))
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--asset-root",
                str(tmp_path),
                "--model-dir",
                str(tmp_path),
                "--output",
                str(output),
                "--allow-partial",
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["status"] == "blocked"
    assert report["complete_100_cases"] is False
    assert "PRIVATE INPUT" not in output.read_text()


def test_bad_media_is_rejected_before_processor(tmp_path: Path) -> None:
    import shutil
    import subprocess

    from train.multimodal import load_media

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe unavailable; not media-decode acceptance")
    media = tmp_path / "bad.wav"
    media.write_bytes(b"not a WAV file")
    with pytest.raises(subprocess.CalledProcessError):
        load_media(media, "audio", num_frames=8, max_seconds=30)


def test_manifest_rejects_cross_pool_family(tmp_path: Path) -> None:
    first, second = public_case(), public_case()
    second["id"] = "second"
    second["meta"]["pool"] = "train"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([first, second]))
    with pytest.raises(ValueError, match="family crosses"):
        read_manifest(manifest, allow_partial=True)


def test_audio_duration_limit_rejects_without_truncating(tmp_path: Path) -> None:
    import shutil
    import wave

    from train.multimodal import load_media

    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe unavailable; not media-decode acceptance")
    media = tmp_path / "duration.wav"
    with wave.open(str(media), "wb") as source:
        source.setnchannels(1)
        source.setsampwidth(2)
        source.setframerate(8000)
        source.writeframes(b"\0\0" * 800)
    with pytest.raises(ValueError, match="duration"):
        load_media(media, "audio", num_frames=8, max_seconds=0.01)


@pytest.mark.parametrize("existing_kind", ["file", "directory", "symlink", "dangling_symlink"])
def test_cli_refuses_existing_output_without_changing_evidence(
    tmp_path: Path, existing_kind: str
) -> None:
    output = tmp_path / "report.json"
    sentinel = tmp_path / "previous-evidence"
    before = b'{"status":"historical", "evidence":"keep exact bytes"}\n'
    sentinel.write_bytes(before)
    if existing_kind == "file":
        output.write_bytes(before)
    elif existing_kind == "directory":
        output.mkdir()
        (output / "historical.json").write_bytes(before)
    elif existing_kind == "symlink":
        output.symlink_to(sentinel)
    else:
        output.symlink_to(tmp_path / "missing-target")
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "--manifest",
                str(tmp_path / "missing-manifest"),
                "--asset-root",
                str(tmp_path),
                "--model-dir",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
    assert caught.value.code == 2
    assert sentinel.read_bytes() == before
    if existing_kind in {"file", "symlink"}:
        assert output.read_bytes() == before
    elif existing_kind == "directory":
        assert (output / "historical.json").read_bytes() == before
    else:
        assert output.is_symlink()
        assert not output.exists()


@pytest.mark.parametrize("kinds", [["video", "video"], ["audio", "audio"], ["image", "video"]])
def test_pending_canonical_combinations_fail_before_loading_files(
    tmp_path: Path, kinds: list[str]
) -> None:
    from data.schema import Prefix, Question
    from train.multimodal import prepare_decision

    prefix = Prefix.model_validate(
        {
            "state": "public fixture",
            "media": [{"type": kind, "uri": "missing", "sha256": "a" * 64} for kind in kinds],
        }
    )
    question = Question(qid="q", type="noul", instruction="Is evidence sufficient?")
    with pytest.raises(ValueError, match="canonical|compatibility"):
        prepare_decision(prefix, question, object(), tmp_path, profile="canonical_http")


def test_verified_playlist_cannot_read_unhashed_external_media(tmp_path: Path) -> None:
    import shutil
    import subprocess

    from train.multimodal import load_media

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg required for real external-reference regression")
    root = tmp_path / "assets"
    root.mkdir()
    outside = tmp_path / "outside.ts"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=1",
            "-c:a",
            "aac",
            "-f",
            "mpegts",
            str(outside),
        ],
        check=True,
        capture_output=True,
    )
    playlist = root / "verified.m3u8"
    playlist.write_text(
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:2\n"
        "#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\n../outside.ts\n#EXT-X-ENDLIST\n"
    )
    path = resolve_media(root, "verified.m3u8", file_hash(playlist), max_bytes=1000)
    with pytest.raises(subprocess.CalledProcessError):
        load_media(path, "audio", num_frames=8, max_seconds=30)


def test_total_image_budget_rejects_before_any_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from data.schema import Prefix, Question
    from train.multimodal import prepare_decision

    image = pytest.importorskip("PIL.Image")
    for name in ("a.png", "b.png"):
        image.new("RGB", (10, 10)).save(tmp_path / name)
    prefix = Prefix(
        state="public",
        media=[
            {"type": "image", "uri": name, "sha256": file_hash(tmp_path / name)}
            for name in ("a.png", "b.png")
        ],
    )
    monkeypatch.setattr(
        "train.multimodal.load_media", lambda *a, **k: pytest.fail("must reject before decode")
    )
    with pytest.raises(ValueError, match="total image pixel"):
        prepare_decision(
            prefix,
            Question(qid="q", type="noul", instruction="Public question"),
            None,
            tmp_path,
            max_total_image_pixels=199,
        )


@pytest.mark.parametrize("missing", [None, "pixel_values_videos", "video_position_ids"])
def test_official_video_tensor_contract_allows_version_optional_frame_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from data.schema import Prefix, Question
    from train.multimodal import prepare_decision

    class Tokenizer:
        all_special_ids: list[int] = []
        unk_token_id = -1

        def encode(self, text: str, **kwargs: object) -> list[int]:
            return [ord(char) for char in text]

        def decode(self, ids: list[int], **kwargs: object) -> str:
            return "".join(chr(value) for value in ids)

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, *args: object, **kwargs: object) -> str:
            return "public prompt"

        def __call__(self, **kwargs: object) -> dict:
            result = {
                "input_ids": torch.tensor([self.tokenizer.encode("public prompt")]),
                "attention_mask": torch.ones((1, 13), dtype=torch.int64),
                "mm_token_type_ids": torch.zeros((1, 13), dtype=torch.int64),
                "pixel_values_videos": torch.zeros((1, 8, 1, 3)),
                "video_position_ids": torch.zeros((1, 8, 1, 2), dtype=torch.int64),
            }
            if missing:
                result.pop(missing)
            return result

    frame = SimpleNamespace(tobytes=lambda: b"public frame")
    monkeypatch.setattr("train.multimodal.resolve_media", lambda *args, **kwargs: tmp_path / "v")
    monkeypatch.setattr(
        "train.multimodal.load_media",
        lambda *args, **kwargs: {
            "frames": [frame] * 8,
            "metadata": {"frames_indices": list(range(8))},
        },
    )
    prefix = Prefix.model_validate(
        {"state": "public", "media": [{"type": "video", "uri": "v", "sha256": "a" * 64}]}
    )
    question = Question(
        qid="q", type="choice", instruction="Choose evidence", options=["first", "second"]
    )
    if missing:
        with pytest.raises(ValueError, match="omitted required modality tensors"):
            prepare_decision(prefix, question, Processor(), tmp_path, video_frames=8)
    else:
        prepared = prepare_decision(prefix, question, Processor(), tmp_path, video_frames=8)
        assert "num_frames_per_video" not in prepared.inputs
        assert len(prepared.summary["decoded_video_frame_sha256"][0]) == 8


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_media_decoder_preserves_callers_piped_stdin(tmp_path: Path, kind: str) -> None:
    import hashlib
    import os
    import shutil
    import subprocess
    import sys
    import wave

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("actual FFmpeg decoder required")
    pytest.importorskip("numpy")
    path = tmp_path / "public.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * 16000 * 15)
    child_env = os.environ.copy()
    if kind == "video":
        pytest.importorskip("PIL.Image")
        path = tmp_path / "public.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x64:rate=25",
                "-t",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            check=True,
            capture_output=True,
            timeout=30,
        )
        # A real-time source gives FFmpeg time to poll its interactive stdin.
        real_ffmpeg = shutil.which("ffmpeg")
        wrapper = tmp_path / "ffmpeg"
        wrapper.write_text(
            f"#!{sys.executable}\nimport os,sys\nos.execv({real_ffmpeg!r}, "
            f"[{real_ffmpeg!r}, '-re', *sys.argv[1:]])\n"
        )
        wrapper.chmod(0o755)
        child_env["PATH"] = str(tmp_path) + os.pathsep + os.environ["PATH"]
    sentinel = b"q\npublic caller continuation must remain available\n"
    code = (
        "import hashlib,json,sys; from pathlib import Path; "
        "from train.multimodal import load_media; "
        "media=load_media(Path(sys.argv[1]),sys.argv[2],num_frames=8,max_seconds=30); "
        "remaining=sys.stdin.buffer.read(); "
        "print(json.dumps({'stdin_sha256':hashlib.sha256(remaining).hexdigest()}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), kind],
        input=sentinel,
        capture_output=True,
        timeout=30,
        check=False,
        env=child_env,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert json.loads(result.stdout) == {
        "stdin_sha256": hashlib.sha256(sentinel).hexdigest(),
    }
