"""Controlled variants use synthetic TRAIN fixtures, never sealed pools."""

import json
from pathlib import Path

import pytest
from test_s4_crossmodal_train import fixture

from data.schema import Sample
from scripts.gold_inventory import digest
from scripts.s4_crossmodal_pairs import build, main
from scripts.s4_crossmodal_train import build as build_original


def original(tmp_path: Path) -> tuple[Path, str]:
    manifest, _, _ = fixture(tmp_path)
    root = tmp_path / "original"
    build_original(manifest, root, count=4)
    return root, digest((root / "provenance.json").read_bytes())


def test_variants_preserve_components_gold_and_family(tmp_path: Path) -> None:
    root, sha = original(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "paired"
    report = build(root, sha, output)
    assert report["status"] == "new_prepared_not_admitted"
    assert report["count"] == 12 and report["group_count"] == 4
    samples = {
        s.id: s
        for s in (
            Sample.model_validate_json(x) for x in (output / "train.jsonl").read_text().splitlines()
        )
    }
    groups = [json.loads(x) for x in (output / "groups.jsonl").read_text().splitlines()]
    components = {
        c["item_id"]: c
        for c in map(json.loads, (output / "components.jsonl").read_text().splitlines())
    }
    for group in groups:
        variants = [samples[i] for i in group["variants"].values()]
        assert len({s.meta.family_id for s in variants}) == 1
        assert len({s.questions[0].instruction for s in variants}) == 3
        assert [s.questions[0].target.value for s in variants] == ["yes", "no", "no"]
        for modality in ("image", "audio"):
            assert len({components[s.id][modality]["item_id"] for s in variants}) == 1
            assert len({components[s.id][modality]["media_sha256"] for s in variants}) == 1
        assert components[variants[0].id]["audio"] == components[variants[1].id]["audio"]
        assert components[variants[0].id]["image"] == components[variants[2].id]["image"]
    assert all(p.read_bytes() == content for p, content in before.items())
    assert all(s.meta.pool == "train" for s in samples.values())
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.rglob("*") if p.is_file())
    with pytest.raises(ValueError, match="output_exists"):
        build(root, sha, output)


@pytest.mark.parametrize("change", ["sha", "ledger", "allocation", "source", "pool"])
def test_reject_binding_or_source_mutation(tmp_path: Path, change: str) -> None:
    root, sha = original(tmp_path)
    provenance = json.loads((root / "provenance.json").read_text())
    if change == "sha":
        sha = "0" * 64
    elif change == "source":
        p = Path(provenance["input_bindings"]["sources"][0]["path"])
        p.write_bytes(p.read_bytes() + b"\n")
    else:
        name = "allocation" if change == "allocation" else "components"
        p = root / (name + ".jsonl")
        rows = list(map(json.loads, p.read_text().splitlines()))
        if change == "pool":
            rows[0]["pool"] = "dev"
        elif change == "ledger":
            rows[0]["program_truth"] = "U"
        else:
            rows[0]["derived_sample_sha256"] = "0" * 64
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        provenance["output_sha256"][name] = digest(p.read_bytes())
        (root / "provenance.json").write_text(json.dumps(provenance))
        sha = digest((root / "provenance.json").read_bytes())
    with pytest.raises(ValueError):
        build(root, sha, tmp_path / "paired")
    assert not (tmp_path / "paired").exists()


def test_cli_rejection_receipt_no_overwrite(tmp_path: Path) -> None:
    root, _ = original(tmp_path)
    report = tmp_path / "report.json"
    args = [
        "--original-dir",
        str(root),
        "--provenance-sha256",
        "0" * 64,
        "--output-dir",
        str(tmp_path / "paired"),
        "--report",
        str(report),
    ]
    assert main(args) == 1
    before = report.read_bytes()
    assert json.loads(before)["status"] == "rejected"
    assert main(args) == 1 and report.read_bytes() == before
