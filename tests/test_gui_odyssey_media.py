import pathlib
import zipfile

from scripts.gui_odyssey_media import (
    EpisodePlan,
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
