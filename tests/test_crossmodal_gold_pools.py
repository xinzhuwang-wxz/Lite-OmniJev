"""Public synthetic fixtures only."""

import json
from pathlib import Path
from typing import Any

import pytest
from test_gold_inventory import SECRET, manifest, sample, spec

from data.schema import Sample
from scripts.crossmodal_gold_pools import QUOTAS, build, compose, legal_target, main, media_uri
from scripts.gold_inventory import audit_manifest, digest


def fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    artifacts, sources = [], []
    for pool in QUOTAS:
        for dataset, modality, count in (
            ("VQA-v2", "image", 4),
            ("Clotho-AQA", "audio", 2),
            ("LibriSpeech", "audio", 2),
        ):
            rows = []
            for i in range(count):
                name = f"{pool}-{dataset}-{i}"
                row = sample(name, modality=modality, pool=pool)
                path = tmp_path / (name + ".media")
                # Repeated Clotho media/family within one pool is deliberate.
                payload = f"{pool}:{dataset}:{0 if dataset == 'Clotho-AQA' else i}".encode()
                path.write_bytes(payload)
                row["prefix"]["media"][0].update(uri=path.as_uri(), sha256=digest(payload))
                row["meta"]["source_dataset"] = dataset
                if dataset == "Clotho-AQA":
                    row["meta"]["family_id"] = pool + "-clotho-shared"
                else:
                    row["questions"][0].update(
                        type="choice",
                        options=["red", "blue"],
                        target={"kind": "hard", "value": "A", "source": SECRET},
                    )
                rows.append(row)
            artifact = spec(tmp_path, rows, cohort=modality, expected_pool=pool)
            artifacts.append(artifact)
            sources.append(
                {k: artifact[k] for k in ("path", "sha256")} | {"pool": pool, "dataset": dataset}
            )
    dev = sample("untouched-dev")
    artifacts.append(spec(tmp_path, [dev], cohort="text", expected_pool="dev"))
    inventory = manifest(tmp_path, artifacts)
    config = {
        "inventory": {"path": str(inventory), "sha256": digest(inventory.read_bytes())},
        "sources": sources,
    }
    path = tmp_path / "build.json"
    path.write_text(json.dumps(config))
    return path, config


def quotas() -> dict[str, dict[str, int]]:
    return {pool: {"Clotho-AQA": 2, "LibriSpeech": 2} for pool in QUOTAS}


def test_composition_preservation_rules_and_sharing(tmp_path: Path) -> None:
    path, config = fixture(tmp_path)
    originals = {s["path"]: Path(s["path"]).read_bytes() for s in config["sources"]}
    output = tmp_path / "output"
    receipt = build(path, tmp_path, output, canary=True, quotas=quotas())
    assert receipt["synthetic_canary"] and receipt["effective_sample_size"] == "not_estimated"
    for pool in QUOTAS:
        counts = receipt["counts"][pool]
        assert counts["question_items"] == 4 and counts["audio_families"] == 3
        assert counts["audio_media"] == 3
        assert counts["rules"] == {
            "both_true": 2,
            "image_false_audio_true": 1,
            "image_true_audio_false": 1,
        }
    combined = json.loads((output / "inventory-manifest.json").read_bytes())
    original = json.loads(Path(config["inventory"]["path"]).read_bytes())
    assert combined["samples"][: len(original["samples"])] == original["samples"]
    assert audit_manifest(output / "inventory-manifest.json").outcome == "isolated"
    assert all(Path(p).read_bytes() == content for p, content in originals.items())
    second = build(path, tmp_path, tmp_path / "second", canary=True, quotas=quotas())
    assert receipt["output_sha256"]["components"] == second["output_sha256"]["components"]
    with pytest.raises(ValueError):
        build(path, tmp_path, output, canary=True, quotas=quotas())


@pytest.mark.parametrize("audio_type", ["choice", "noul"])
@pytest.mark.parametrize(
    "rule,truth",
    [("both_true", "yes"), ("image_false_audio_true", "no"), ("image_true_audio_false", "no")],
)
def test_and_truth_and_legal_false_proposals(
    tmp_path: Path, audio_type: str, rule: str, truth: str
) -> None:
    path, config = fixture(tmp_path)
    image = Sample.model_validate_json(
        Path(config["sources"][0]["path"]).read_text().splitlines()[0]
    )
    source = config["sources"][1 if audio_type == "noul" else 2]
    audio = Sample.model_validate_json(Path(source["path"]).read_text().splitlines()[0])
    result, ledger = compose(image, audio, tmp_path, "calibration", rule, True)
    assert result.questions[0].target.value == truth
    assert ledger["audio"]["proposal"] in (["yes", "no"] if audio_type == "noul" else ["A", "B"])


def test_source_mutation_real_quota_and_redaction(tmp_path: Path, capsys: Any) -> None:
    path, config = fixture(tmp_path)
    with pytest.raises(ValueError):
        build(path, tmp_path, tmp_path / "real", quotas=quotas())
    Path(config["sources"][0]["path"]).write_text(SECRET)
    args = [
        "--manifest",
        str(path),
        "--media-root",
        str(tmp_path),
        "--output-dir",
        str(tmp_path / "bad"),
        "--report",
        str(tmp_path / "report"),
        "--synthetic-canary",
        "--canary-per-source",
        "2",
    ]
    assert main(args) == 1
    output = capsys.readouterr()
    assert SECRET not in output.out and str(tmp_path) not in output.out and not output.err


def test_path_and_target_rejections(tmp_path: Path) -> None:
    for uri in (
        "../escape",
        "https://example.com/image",
        "file://host/image",
        "/outside",
        "image?query=1",
    ):
        with pytest.raises(ValueError):
            media_uri(tmp_path, uri, "0" * 64)
    raw = sample("invalid", modality="image")
    with pytest.raises(ValueError):
        legal_target(Sample.model_validate(raw), "image")
    raw["questions"][0].update(
        type="choice", options=["a", "b"], target={"kind": "hard", "value": "U", "source": SECRET}
    )
    with pytest.raises(ValueError):
        legal_target(Sample.model_validate(raw), "image")


def test_cross_pool_alias_rejected_and_source_must_be_current(tmp_path: Path) -> None:
    path, config = fixture(tmp_path)
    inventory_path = Path(config["inventory"]["path"])
    inventory = json.loads(inventory_path.read_bytes())
    source = config["sources"][3]
    source_path = Path(source["path"])
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    rows[0]["meta"]["family_id"] = "family-calibration-VQA-v2-0"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source["sha256"] = digest(source_path.read_bytes())
    inventory["samples"][3]["sha256"] = source["sha256"]
    inventory_path.write_text(json.dumps(inventory))
    config["inventory"]["sha256"] = digest(inventory_path.read_bytes())
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="inventory_isolation"):
        build(path, tmp_path, tmp_path / "bad", canary=True, quotas=quotas())
