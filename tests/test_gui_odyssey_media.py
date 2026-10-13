import pathlib
import zipfile

from scripts.gui_odyssey_media import (
    EntryTask,
    EpisodePlan,
    build_entry_batches,
    compressed_from_local_blob,
    parse_central_directory,
    part_ranges,
    select_under_budget,
)


def _central_directory_bytes(zip_path: pathlib.Path) -> bytes:
    data = zip_path.read_bytes()
    eocd = data.rfind(b"PK\x05\x06")
    assert eocd >= 0
    cd_size = int.from_bytes(data[eocd + 12 : eocd + 16], "little")
    cd_offset = int.from_bytes(data[eocd + 16 : eocd + 20], "little")
    return data[cd_offset : cd_offset + cd_size]


def test_parse_central_directory_reads_zip_members(tmp_path):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("screenshots/111_0.png", b"fake-png-0")
        archive.writestr("screenshots/111_1.png", b"fake-png-1")
    entries = parse_central_directory(_central_directory_bytes(path))
    names = [entry.name for entry in entries]
    assert names == ["screenshots/111_0.png", "screenshots/111_1.png"]
    assert all(entry.method == 8 for entry in entries)
    assert all(entry.compressed_size > 0 for entry in entries)


def test_part_ranges_crosses_split_boundaries():
    assert list(part_ranges(0, 8, 5, [10, 10])) == [(0, 8, 9), (1, 0, 2)]


def test_select_under_budget_keeps_whole_episodes_in_lexical_order():
    plans = [
        EpisodePlan("002", "002.json", "app", "train", 1, 7, 8, 1),
        EpisodePlan("001", "001.json", "app", "train", 1, 6, 7, 1),
        EpisodePlan("003", "003.json", "app", "train", 1, 5, 6, 1),
    ]
    selected = select_under_budget(plans, 12)
    assert [item.episode_id for item in selected] == ["001", "003"]
    assert sum(item.compressed_bytes for item in selected) == 11


def test_build_entry_batches_groups_adjacent_members_under_cap(tmp_path):
    zip_path = _make_zip(tmp_path)
    entries = parse_central_directory(_central_directory_bytes(zip_path))
    task_a = EntryTask(entry=entries[0], out_path=tmp_path / "a.png")
    task_b = EntryTask(entry=entries[1], out_path=tmp_path / "b.png")
    batches = build_entry_batches([task_b, task_a], max_batch_bytes=1_000_000)
    assert len(batches) == 1
    assert [task.entry.name for task in batches[0].tasks] == [
        "screenshots/111_0.png",
        "screenshots/111_1.png",
    ]


def _make_zip(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "sample-batch.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("screenshots/111_0.png", b"fake-png-0")
        archive.writestr("screenshots/111_1.png", b"fake-png-1")
    return path


def test_compressed_from_local_blob_extracts_payload_from_combined_range(tmp_path):
    zip_path = _make_zip(tmp_path)
    data = zip_path.read_bytes()
    entries = parse_central_directory(_central_directory_bytes(zip_path))
    first = entries[0]
    blob = data[first.local_header_offset : first.local_header_offset + 512]
    compressed = compressed_from_local_blob(first, blob, first.local_header_offset)
    assert compressed
