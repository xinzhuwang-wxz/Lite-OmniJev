import csv
import hashlib
import stat
import zipfile

import pytest

from scripts import prepare_clotho as prep


@pytest.fixture
def source(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["file_name", "license", "sound_link", "manufacturer"]
        )
        writer.writeheader()
        for name, license_url in [
            ("public.wav", "https://creativecommons.org/publicdomain/zero/1.0/"),
            ("credit.wav", "http://creativecommons.org/licenses/by/3.0/"),
            ("research.wav", "https://creativecommons.org/licenses/by-nc/3.0/"),
            ("unknown.wav", ""),
            ("spoof.wav", "https://creativecommons.org.evil/licenses/by/3.0/"),
        ]:
            writer.writerow(
                {
                    "file_name": name,
                    "license": license_url,
                    "sound_link": "https://freesound.org/example",
                    "manufacturer": "creator",
                }
            )
    monkeypatch.setattr(prep, "METADATA_SHA256", hashlib.sha256(metadata.read_bytes()).hexdigest())
    return metadata


@pytest.fixture(autouse=True)
def decode_stub(monkeypatch):
    monkeypatch.setattr(
        prep,
        "decode_representative_audio",
        lambda output, names: {
            "sample_limit": prep.DECODE_SAMPLE_COUNT,
            "selected": min(len(names), prep.DECODE_SAMPLE_COUNT),
            "selection": "test stub",
            "decode_scope": "test stub",
        },
    )


def archive(tmp_path, monkeypatch, *, malicious=None, missing=False):
    path = tmp_path / "audio.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr("audio_files/public.wav", b"public fixture")
        if not missing:
            zipped.writestr("audio_files/credit.wav", b"credited fixture")
        zipped.writestr("audio_files/research.wav", b"must not extract")
        if malicious == "traversal":
            zipped.writestr("../outside.wav", b"bad")
        elif malicious == "symlink":
            info = zipfile.ZipInfo("link.wav")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zipped.writestr(info, "/private/secret")
    monkeypatch.setattr(prep, "ARCHIVE_BYTES", path.stat().st_size)
    monkeypatch.setattr(prep, "ARCHIVE_MD5", hashlib.md5(path.read_bytes()).hexdigest())
    monkeypatch.setattr(prep, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_license_plan_excludes_nc_unknown_and_spoofed_hosts(source):
    result = prep.plan(source)
    assert {row["file_name"] for row in result["approved_audio"]} == {"public.wav", "credit.wav"}
    assert result["excluded_audio_count"] == 3
    assert result["training_admitted"] is False
    assert result["approved_audio"][1]["attribution"] == "creator"


def test_tampered_license_metadata_rejected(source):
    source.write_text(source.read_text().replace("by-nc/", "by/"))
    with pytest.raises(ValueError, match="metadata_hash"):
        prep.plan(source)


def test_extract_only_whitelist_and_verified_resume(source, tmp_path, monkeypatch):
    path = archive(tmp_path, monkeypatch)
    dest = tmp_path / "approved"
    first = prep.extract(path, source, dest)
    assert first["extracted_audio_count"] == 2
    assert first["approved_audio_count"] == 2
    assert sorted(p.name for p in dest.iterdir()) == ["credit.wav", "public.wav"]
    before = {p.name: p.stat().st_mtime_ns for p in dest.iterdir()}
    second = prep.extract(path, source, dest)
    assert first == second
    assert before == {p.name: p.stat().st_mtime_ns for p in dest.iterdir()}


@pytest.mark.parametrize("malicious", ["traversal", "symlink"])
def test_malicious_archive_has_no_extraction(source, tmp_path, monkeypatch, malicious):
    path = archive(tmp_path, monkeypatch, malicious=malicious)
    dest = tmp_path / "approved"
    with pytest.raises(ValueError, match="unsafe_zip|special_file"):
        prep.extract(path, source, dest)
    assert not dest.exists()


def test_missing_audio_fails_without_shrinking_denominator(source, tmp_path, monkeypatch):
    path = archive(tmp_path, monkeypatch, missing=True)
    with pytest.raises(ValueError, match="approved_audio_missing"):
        prep.extract(path, source, tmp_path / "approved")


def test_existing_corrupt_audio_preserved(source, tmp_path, monkeypatch):
    path = archive(tmp_path, monkeypatch)
    dest = tmp_path / "approved"
    dest.mkdir()
    (dest / "public.wav").write_bytes(b"preserve me")
    with pytest.raises(ValueError, match="existing_audio_preserved"):
        prep.extract(path, source, dest)
    assert (dest / "public.wav").read_bytes() == b"preserve me"


def test_corrupt_archive_rejected_before_extract(source, tmp_path, monkeypatch):
    path = archive(tmp_path, monkeypatch)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="checksum"):
        prep.extract(path, source, tmp_path / "approved")


def test_fixed_revision_alias_preserves_metadata_name_and_records_archive_member(
    tmp_path, monkeypatch
):
    metadata = tmp_path / "metadata.csv"
    canonical = "souffle_me_tallique.wav"
    archive_name = "souffle_me.tallique.wav"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["file_name", "license", "sound_link", "manufacturer"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "file_name": canonical,
                "license": "https://creativecommons.org/publicdomain/zero/1.0/",
                "sound_link": "https://freesound.org/example",
                "manufacturer": "creator",
            }
        )
    monkeypatch.setattr(prep, "METADATA_SHA256", hashlib.sha256(metadata.read_bytes()).hexdigest())
    path = tmp_path / "audio.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr(f"audio_files/{archive_name}", b"aliased fixture")
    monkeypatch.setattr(prep, "ARCHIVE_BYTES", path.stat().st_size)
    monkeypatch.setattr(prep, "ARCHIVE_MD5", hashlib.md5(path.read_bytes()).hexdigest())
    monkeypatch.setattr(prep, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())

    result = prep.extract(path, metadata, tmp_path / "approved")

    assert (tmp_path / "approved" / canonical).read_bytes() == b"aliased fixture"
    assert result["alias_count"] == 1
    assert result["archive_name_aliases"] == [
        {
            "metadata_file_name": canonical,
            "archive_member_name": archive_name,
            "mode": "fixed_revision_alias",
        }
    ]


def test_alias_conflict_rejected_without_fuzzy_matching(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["file_name", "license", "sound_link", "manufacturer"]
        )
        writer.writeheader()
        for name in ["souffle_me_tallique.wav", "souffle_me.tallique.wav"]:
            writer.writerow(
                {
                    "file_name": name,
                    "license": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "sound_link": "https://freesound.org/example",
                    "manufacturer": "creator",
                }
            )
    monkeypatch.setattr(prep, "METADATA_SHA256", hashlib.sha256(metadata.read_bytes()).hexdigest())
    path = tmp_path / "audio.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr("audio_files/souffle_me.tallique.wav", b"fixture")
    monkeypatch.setattr(prep, "ARCHIVE_BYTES", path.stat().st_size)
    monkeypatch.setattr(prep, "ARCHIVE_MD5", hashlib.md5(path.read_bytes()).hexdigest())
    monkeypatch.setattr(prep, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())

    with pytest.raises(ValueError, match="archive_alias_conflict"):
        prep.extract(path, metadata, tmp_path / "approved")
