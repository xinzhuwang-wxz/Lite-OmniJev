"""Only synthetic screenshots and annotations; never reads real sealed data."""

import json
import stat
from pathlib import Path

import pytest

from scripts import gui_gold_pools as gold
from scripts import s0_gui_dev as gui

Image = pytest.importorskip("PIL.Image", reason="GUI full decode requires Pillow")


def fixture(tmp_path):
    def artifact(name, content):
        path = tmp_path / name
        path.write_bytes(content)
        return dict(path=str(path), sha256=gold.digest(content))

    annotations = tmp_path / "annotations"
    annotations.mkdir()
    media = tmp_path / "media"
    (media / "screenshots").mkdir(parents=True)
    index = dict(
        schema_version="gui-gold-source-index-v1",
        revision=gold.REVISION,
        license="CC-BY-4.0",
        annotations={},
        media={},
    )
    entries, samples = [], []
    for i in range(8):
        episode = f"{i:016d}"
        entries.append(dict(episode_id=episode, partition="train"))
        steps = []
        for j in range(2):
            name = f"{episode}_{j}.png"
            path = media / "screenshots" / name
            Image.new("RGB", (100, 100), (i * 20, j * 60, 19)).save(path)
            index["media"][name] = gold.file_digest(path)
            raw = dict(
                step=j,
                screenshot=name,
                action="CLICK",
                low_level_instruction=f"Private secret {i} {j}",
                sam2_bbox=[5, 5, 15, 15],
                info=[[10, 10], [10, 10]],
            )
            steps.append(raw)
            row = dict(
                episode_id=episode,
                step=j,
                screenshot=name,
                instruction=raw["low_level_instruction"],
                true_box=(5, 5, 15, 15),
                point=(10, 10),
                negative_boxes=gui.candidate_negative_boxes(
                    (5, 5, 15, 15), (10, 10), width=100, height=100
                ),
                image=gui.image_info(path),
                raw_step_sha256=gold.digest(gui.canonical_json(raw).encode()),
            )
            sample, _, _ = gui.sample_for(row)
            record = sample.model_dump(mode="json")
            record["prefix"]["media"][0]["uri"] = path.as_uri()
            samples.append(record)
        path = annotations / f"{episode}.json"
        path.write_bytes(gold.encode(dict(episode_id=episode, step_length=2, steps=steps)))
        index["annotations"][episode] = gold.file_digest(path)
    reserves, artifacts = [], []
    for i, pool in enumerate(("calibration", "holdout", "dev")):
        sample = samples[i * 2]
        sample["meta"]["pool"] = pool
        a = artifact(f"old-{pool}.jsonl", gold.encode(sample))
        artifacts.append(dict(**a, cohort="screenshot", exposure="current", expected_pool=pool))
        if pool != "dev":
            reserves.append(a)
    manifest = dict(
        schema_version="gui-gold-build-v1",
        plan=artifact("plan.json", gold.encode(dict(partition="train", selected_episodes=entries))),
        source_index=artifact("index.json", gold.encode(index)),
        inventory=artifact(
            "inventory.json",
            gold.encode(
                dict(
                    schema_version="gold-inventory-manifest-v1",
                    samples=artifacts,
                    component_ledgers=[],
                )
            ),
        ),
        quarantine=artifact("quarantine.json", gold.encode(dict(families=[], media=[]))),
        current_reserves=reserves,
        annotation_root=str(annotations),
        media_root=str(media),
    )
    path = tmp_path / "manifest.json"
    path.write_bytes(gold.encode(manifest))
    return path, manifest


def run(tmp_path, manifest, *, canary=True, cal=6, hold=3):
    argv = [
        "--manifest",
        str(manifest),
        "--output-dir",
        str(tmp_path / "output"),
        "--report",
        str(tmp_path / "receipt.json"),
        "--calibration-count",
        str(cal),
        "--holdout-count",
        str(hold),
    ]
    if canary:
        argv.append("--canary")
    code = gold.main(argv)
    return code, json.loads((tmp_path / "receipt.json").read_bytes())


def test_preservation_quota_redaction_permissions(tmp_path, capsys):
    path, manifest = fixture(tmp_path)
    code, receipt = run(tmp_path, path)
    assert code == 0, receipt
    assert receipt["counts"]["preserved_items"] == 2
    assert receipt["counts"]["calibration"] == 6
    assert receipt["counts"]["holdout"] == 3
    assert not receipt["real_source_verified"]
    assert "Private secret" not in capsys.readouterr().out
    for pool, old in zip(("calibration", "holdout"), manifest["current_reserves"], strict=True):
        original = json.loads(Path(old["path"]).read_bytes())
        records = [
            json.loads(line)
            for line in (tmp_path / "output" / f"{pool}.jsonl").read_text().splitlines()
        ]
        assert original in records
    combined = json.loads((tmp_path / "output/inventory-manifest.json").read_bytes())
    assert [a["exposure"] for a in combined["samples"][:3]] == [
        "historical",
        "historical",
        "current",
    ]
    assert gold.audit_manifest(tmp_path / "output/inventory-manifest.json").outcome == "isolated"
    assert stat.S_IMODE((tmp_path / "output").stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(p.stat().st_mode) == 0o600
        for p in (tmp_path / "output").rglob("*")
        if p.is_file()
    )
    saved = (tmp_path / "output/calibration.jsonl").read_bytes()
    assert (
        gold.main(
            [
                "--manifest",
                str(path),
                "--output-dir",
                str(tmp_path / "output"),
                "--report",
                str(tmp_path / "other.json"),
                "--canary",
            ]
        )
        == 1
    )
    assert saved == (tmp_path / "output/calibration.jsonl").read_bytes()


def rebind(manifest, key):
    manifest[key]["sha256"] = gold.file_digest(Path(manifest[key]["path"]))


@pytest.mark.parametrize(
    "attack", ["label", "annotation", "media", "symlink", "reserve_binding", "pin", "quarantine"]
)
def test_rejections(tmp_path, attack, capsys):
    path, manifest = fixture(tmp_path)
    if attack == "label":
        a = manifest["current_reserves"][0]
        old = json.loads(Path(a["path"]).read_bytes())
        old["questions"][0]["target"]["value"] = (
            "A" if old["questions"][0]["target"]["value"] != "A" else "B"
        )
        Path(a["path"]).write_bytes(gold.encode(old))
        a["sha256"] = gold.file_digest(Path(a["path"]))
        inv = json.loads(Path(manifest["inventory"]["path"]).read_bytes())
        inv["samples"][0]["sha256"] = a["sha256"]
        Path(manifest["inventory"]["path"]).write_bytes(gold.encode(inv))
        rebind(manifest, "inventory")
    elif attack == "annotation":
        (Path(manifest["annotation_root"]) / "0000000000000004.json").write_text("{}")
    elif attack == "media":
        (Path(manifest["media_root"]) / "screenshots/0000000000000004_0.png").write_bytes(
            (Path(manifest["media_root"]) / "screenshots/0000000000000005_0.png").read_bytes()
        )
    elif attack == "symlink":
        media = Path(manifest["media_root"]) / "screenshots/0000000000000004_0.png"
        media.unlink()
        media.symlink_to(media.parent / "0000000000000005_0.png")
    elif attack == "reserve_binding":
        manifest["current_reserves"].pop()
    elif attack == "quarantine":
        Path(manifest["quarantine"]["path"]).write_bytes(
            gold.encode(
                dict(families=[gold.canonical_family_hash("gui:0000000000000000")], media=[])
            )
        )
        rebind(manifest, "quarantine")
    path.write_bytes(gold.encode(manifest))
    code, receipt = run(
        tmp_path,
        path,
        canary=attack != "pin",
        cal=3000 if attack == "pin" else 6,
        hold=1000 if attack == "pin" else 3,
    )
    assert code == 1
    assert receipt["category"] in {
        "invalid_existing_gold",
        "annotation_source_mismatch",
        "media_source_mismatch",
        "unsafe_path",
        "invalid_metadata",
        "invalid_reserve_binding",
        "source_not_pinned",
    }
    assert "Private secret" not in capsys.readouterr().out
    assert not (tmp_path / "output").exists()


def test_transitive_shared_media_components_and_determinism(tmp_path):
    path, manifest = fixture(tmp_path)
    graph = gold.Components()
    receipt = gold.Receipt()
    m = gold.BuildManifest.model_validate(manifest)
    _, assignments, existing = gold.inventory_state(m, receipt, graph)
    rows = gold.collect(m, gold.Components())
    # Shared candidate RGB ties entire episode groups before selection.
    group = [row for row in rows.values() if row["episode_id"] in {f"{i:016d}" for i in (3, 4, 5)}]
    for row in group:
        row["image"]["decoded_rgb_sha256"] = "a" * 64
    original_parents = dict(graph.parents)
    args = (existing, graph, assignments, gold.Quarantine(families=[], media=[]), 4, 3)
    first = gold.allocate(rows, *args)
    second_graph = gold.Components()
    second_graph.parents = original_parents
    second = gold.allocate(
        dict(reversed(list(rows.items()))),
        existing,
        second_graph,
        assignments,
        gold.Quarantine(families=[], media=[]),
        4,
        3,
    )
    assert first == second
    component_pools = {}
    for pool, ids in first.items():
        for item in ids:
            node = "f:" + gold.canonical_family_hash(f"gui:{rows[item]['episode_id']}")
            component = graph.find(node)
            assert component_pools.setdefault(component, pool) == pool


def test_historical_unavailable_rgb_is_explicit_gap(tmp_path):
    path, manifest = fixture(tmp_path)
    invpath = Path(manifest["inventory"]["path"])
    inv = json.loads(invpath.read_bytes())
    a = inv["samples"][2]
    sample = json.loads(Path(a["path"]).read_bytes())
    sample["prefix"]["media"][0]["uri"] = "/unavailable/history.png"
    Path(a["path"]).write_bytes(gold.encode(sample))
    a["sha256"] = gold.file_digest(Path(a["path"]))
    invpath.write_bytes(gold.encode(inv))
    rebind(manifest, "inventory")
    path.write_bytes(gold.encode(manifest))
    code, receipt = run(tmp_path, path)
    assert code == 0
    assert receipt["historical_rgb_coverage"] == "gap"
    assert receipt["counts"]["historical_gui_media_rgb_unavailable"] == 1


def test_unused_rgb_reencoding_does_not_bridge_history(tmp_path):
    path, manifest = fixture(tmp_path)
    # Unused frames in old episodes are source material, not historical exposure.
    index_path = Path(manifest["source_index"]["path"])
    index = json.loads(index_path.read_bytes())
    media = Path(manifest["media_root"]) / "screenshots"
    target = media / "0000000000000001_1.png"
    with Image.open(media / "0000000000000000_1.png") as image:
        image.save(target, compress_level=0)
    index["media"][target.name] = gold.file_digest(target)
    assert index["media"][target.name] != index["media"]["0000000000000000_1.png"]
    index_path.write_bytes(gold.encode(index))
    rebind(manifest, "source_index")
    path.write_bytes(gold.encode(manifest))
    code, receipt = run(tmp_path, path)
    assert code == 0
    assert receipt["counts"]["source_unused_steps_omitted"] == 4


def test_component_capacity_may_exceed_remaining_quota(tmp_path):
    path, _ = fixture(tmp_path)
    # Every new component has size two; a one-item holdout increment is legal
    # only if its unused original remains assigned to that same component pool.
    code, receipt = run(tmp_path, path, cal=6, hold=2)
    assert code == 0, receipt
    ledger = [
        json.loads(line) for line in (tmp_path / "output/allocation.jsonl").read_text().splitlines()
    ]
    by_component = {}
    for row in ledger:
        assert by_component.setdefault(row["component"], row["pool"]) == row["pool"]


def test_source_index_pin_cannot_be_rebound_in_real_mode(tmp_path):
    path, manifest = fixture(tmp_path)
    manifest["plan"]["sha256"] = gold.PLAN_SHA
    path.write_bytes(gold.encode(manifest))
    code, receipt = run(tmp_path, path, canary=False, cal=3000, hold=1000)
    assert code == 1 and receipt["category"] == "source_not_pinned"


def test_collect_worker_counts_preserve_candidates_graph_and_selection(tmp_path):
    _, manifest = fixture(tmp_path)
    m = gold.BuildManifest.model_validate(manifest)
    outcomes = []
    for workers in (1, 8):
        graph = gold.Components()
        receipt = gold.Receipt()
        _, assignments, existing = gold.inventory_state(m, receipt, graph)
        rows = gold.collect(m, gold.Components(), receipt, workers=workers)
        selected = gold.allocate(
            rows, existing, graph, assignments, gold.Quarantine(families=[], media=[]), 6, 3
        )
        outcomes.append((rows, graph.parents, selected, receipt.counts))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("workers", [1, 8])
def test_collect_worker_exception_preserves_category(tmp_path, workers):
    _, manifest = fixture(tmp_path)
    m = gold.BuildManifest.model_validate(manifest)
    path = Path(m.media_root) / "screenshots" / "0000000000000000_0.png"
    path.write_bytes(b"changed source")
    with pytest.raises(gold.BuildError, match="^media_source_mismatch$"):
        gold.collect(m, gold.Components(), workers=workers)


@pytest.mark.parametrize("workers", [1, 8])
def test_collect_decode_worker_exception_is_not_silently_dropped(tmp_path, workers):
    _, manifest = fixture(tmp_path)
    path = Path(manifest["media_root"]) / "screenshots" / "0000000000000000_0.png"
    path.write_bytes(b"not a PNG")
    index_path = Path(manifest["source_index"]["path"])
    index = json.loads(index_path.read_text())
    index["media"][path.name] = gold.file_digest(path)
    index_path.write_bytes(gold.encode(index))
    manifest["source_index"]["sha256"] = gold.file_digest(index_path)
    from PIL import UnidentifiedImageError

    with pytest.raises(UnidentifiedImageError):
        gold.collect(
            gold.BuildManifest.model_validate(manifest), gold.Components(), workers=workers
        )


def test_collect_deduplicates_raw_media_without_losing_aliases(tmp_path):
    _, manifest = fixture(tmp_path)
    root = Path(manifest["media_root"]) / "screenshots"
    original = root / "0000000000000000_0.png"
    alias = root / "0000000000000001_0.png"
    alias.write_bytes(original.read_bytes())
    index_path = Path(manifest["source_index"]["path"])
    index = json.loads(index_path.read_text())
    index["media"][alias.name] = gold.file_digest(alias)
    index_path.write_bytes(gold.encode(index))
    manifest["source_index"]["sha256"] = gold.file_digest(index_path)
    results = []
    for workers in (1, 8):
        graph, receipt = gold.Components(), gold.Receipt()
        rows = gold.collect(
            gold.BuildManifest.model_validate(manifest), graph, receipt, workers=workers
        )
        assert receipt.counts["source_unique_media_decoded"] == 15
        assert receipt.counts["source_steps_aliased"] == 16
        nodes = ["f:" + gold.canonical_family_hash(f"gui:{i:016d}") for i in (0, 1)]
        assert graph.find(nodes[0]) == graph.find(nodes[1])
        results.append((rows, graph.parents, receipt.counts))
    assert results[0] == results[1]


def cache_for(manifest):
    media = {}
    for path in (Path(manifest["media_root"]) / "screenshots").glob("*.png"):
        info = gui.image_info(path)
        info.pop("bytes")
        info["isolation_alias_sha256"] = gold.digest(
            f"rgb:{info['width']}:{info['height']}:{info['decoded_rgb_sha256']}".encode()
        )
        media[info["sha256"]] = info
    payload = dict(
        schema_version="observed-image-rgb-cache-v1",
        pillow_version="12.3.0",
        source_index_sha256="3b29b518d765aa963ee0b734d1c759ad716e6ed1fcd41a0162530204efe222cc",
        media=media,
    )
    path = Path(manifest["inventory"]["path"]).parent / "rgb-cache.json"
    path.write_bytes(gold.encode(payload))
    manifest["historical_rgb_cache"] = dict(path=str(path), sha256=gold.file_digest(path))
    return path, payload


@pytest.mark.parametrize("attack", ["mutation", "missing", "alias", "key", "conflict"])
def test_historical_rgb_cache_fails_closed(tmp_path, attack):
    _, manifest = fixture(tmp_path)
    path, payload = cache_for(manifest)
    root = Path(manifest["media_root"]) / "screenshots"
    keys = [gold.file_digest(root / f"{i:016d}_0.png") for i in range(3)]
    if attack == "missing":
        del payload["media"][keys[0]]
    elif attack == "alias":
        payload["media"][keys[0]]["isolation_alias_sha256"] = "a" * 64
    elif attack == "key":
        payload["media"][keys[0]]["sha256"] = "a" * 64
    elif attack == "conflict":
        # Actual consumed calibration and holdout images collide in decoded RGB.
        row = payload["media"][keys[1]]
        row["decoded_rgb_sha256"] = payload["media"][keys[0]]["decoded_rgb_sha256"]
        row["isolation_alias_sha256"] = gold.digest(
            f"rgb:{row['width']}:{row['height']}:{row['decoded_rgb_sha256']}".encode()
        )
    else:
        payload["pillow_version"] = "tampered"
    path.write_bytes(gold.encode(payload))
    if attack != "mutation":
        manifest["historical_rgb_cache"]["sha256"] = gold.file_digest(path)
    with pytest.raises(gold.BuildError):
        graph = gold.Components()
        _, assignments, _ = gold.inventory_state(
            gold.BuildManifest.model_validate(manifest), gold.Receipt(), graph
        )
        gold.historical_pools(graph, assignments)


def test_candidate_bridge_excluded_without_changing_history(tmp_path):
    _, manifest = fixture(tmp_path)
    m = gold.BuildManifest.model_validate(manifest)
    graph = gold.Components()
    _, assignments, existing = gold.inventory_state(m, gold.Receipt(), graph)
    rows = gold.collect(m, gold.Components(), assignments=assignments, existing=existing)
    old = dict(graph.parents)
    history_rgbs = [
        row["image"]["decoded_rgb_sha256"] for key, row in rows.items() if key in existing
    ]
    bridge = [row for row in rows.values() if row["episode_id"] == f"{3:016d}"]
    for row, rgb in zip(bridge, history_rgbs, strict=True):
        row["image"]["decoded_rgb_sha256"] = rgb
    selected = gold.allocate(
        rows, existing, graph, assignments, gold.Quarantine(families=[], media=[]), 4, 3
    )
    assert not any(
        item in ids for item, row in rows.items() if row in bridge for ids in selected.values()
    )
    assert gold.historical_pools(graph, assignments)
    assert all(graph.find(node) == parent for node, parent in old.items())


def test_bound_historical_cache_preserves_existing_samples(tmp_path):
    path, manifest = fixture(tmp_path)
    cache_for(manifest)
    path.write_bytes(gold.encode(manifest))
    originals = [
        json.loads(Path(artifact["path"]).read_bytes()) for artifact in manifest["current_reserves"]
    ]
    code, receipt = run(tmp_path, path)
    assert code == 0
    assert receipt["historical_rgb_coverage"] == "complete_supplied_gui_media"
    for original in originals:
        output = tmp_path / "output" / f"{original['meta']['pool']}.jsonl"
        preserved = next(
            json.loads(line)
            for line in output.read_text().splitlines()
            if json.loads(line)["id"] == original["id"]
        )
        assert preserved == original


def test_real_build_requires_bound_historical_cache(tmp_path, monkeypatch):
    path, manifest = fixture(tmp_path)
    monkeypatch.setattr(gold, "PLAN_SHA", manifest["plan"]["sha256"])
    monkeypatch.setattr(gold, "SOURCE_INDEX_SHA", manifest["source_index"]["sha256"])
    code, receipt = run(tmp_path, path, canary=False, cal=3000, hold=1000)
    assert code == 1
    assert receipt["category"] == "historical_rgb_cache_required"
