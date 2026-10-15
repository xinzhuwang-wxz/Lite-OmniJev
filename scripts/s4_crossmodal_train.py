"""Build a bounded TRAIN-only AND pilot from SHA-bound registered components.

Manifest: inventory {path,sha256}, sources [{path,sha256,modality,media_root}].
Exactly one image and one audio artifact; explicit absolute roots are required.
No gold quality, admission, IID, or model-training claim is made.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample
from scripts.crossmodal_gold_pools import compose, legal_target, rank
from scripts.gold_inventory import Manifest, PrivateParser, audit_manifest, digest
from scripts.gold_media_bundle import copy_media, media_path, private_write, real_path
from scripts.s1_text_pools import allocation_rows, canonical_sample_sha
from scripts.vqa2_gold_capacity import bound

VERSION = "crossmodal-train-pilot-v1"
SOURCES = {
    "image": {"aokvqa_correct_choice_idx", "vqa2_unanimous_10_human_answers"},
    "audio": {"clotho_original_unanimous_three_answers"},
}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> str:
    private_write(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return digest(path.read_bytes())


def build(manifest_path: Path, output: Path, *, count: int = 200) -> dict[str, Any]:
    if type(count) is not int or count < 4 or count > 200 or count % 4:
        raise ValueError("quota")
    for path in (manifest_path, output):
        real_path(path)
    if output.exists():
        raise ValueError("output_exists")
    config = json.loads(manifest_path.read_bytes())
    if set(config) != {"inventory", "sources"}:
        raise ValueError("manifest")
    real_path(Path(config["inventory"]["path"]))
    inventory_bytes = bound(config["inventory"])
    inventory = Manifest.model_validate_json(inventory_bytes)
    if audit_manifest(Path(config["inventory"]["path"])).outcome != "isolated":
        raise ValueError("inventory_isolation")
    if len(config["sources"]) != 2:
        raise ValueError("sources")
    selected: dict[str, list[tuple[Sample, Path]]] = {}
    counts: dict[str, Any] = {}
    hashes = {"manifest": digest(manifest_path.read_bytes()), "inventory": digest(inventory_bytes)}
    for source in config["sources"]:
        if set(source) != {"path", "sha256", "modality", "media_root"}:
            raise ValueError("source_declaration")
        modality = source["modality"]
        if modality not in SOURCES or modality in selected:
            raise ValueError("source_modality")
        registered = [
            a
            for a in inventory.samples
            if a.path == source["path"]
            and a.sha256 == source["sha256"]
            and a.expected_pool == "train"
        ]
        if len(registered) != 1:
            raise ValueError("source_unregistered")
        artifact = registered[0]
        real_path(Path(source["path"]))
        root = Path(source["media_root"])
        real_path(root)
        payload = bound({k: source[k] for k in ("path", "sha256")})
        hashes[modality] = digest(payload)
        candidates = []
        for line in payload.splitlines():
            original = Sample.model_validate_json(line)
            cohort = artifact.cohort or (artifact.cohorts_by_data_version or {}).get(
                original.meta.data_version
            )
            if (
                cohort != modality
                or original.meta.pool != "train"
                or original.meta.usage != "commercial"
                or original.meta.modality != [modality]
                or len(original.prefix.media) != 1
                or original.prefix.media[0].type != modality
            ):
                raise ValueError("source_contract")
            legal_target(original, modality)
            q = original.questions[0]
            if (
                q.type != ("choice" if modality == "image" else "noul")
                or not isinstance(q.target, HardTarget)
                or q.target.source not in SOURCES[modality]
            ):
                raise ValueError("source_supervision")
            media = original.prefix.media[0]
            path = media_path([root], media.uri)
            if digest(path.read_bytes()) != media.sha256:
                raise ValueError("media_hash")
            candidates.append((original, path))
        families: set[str] = set()
        media_hashes: set[str] = set()
        chosen = []
        for original, path in sorted(candidates, key=lambda p: rank(modality + ":" + p[0].id)):
            if (
                original.meta.family_id in families
                or original.prefix.media[0].sha256 in media_hashes
            ):
                continue
            families.add(original.meta.family_id)
            media_hashes.add(original.prefix.media[0].sha256)
            chosen.append((original, path))
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError("unique_capacity")
        selected[modality] = chosen
        counts[modality] = {
            "input": len(candidates),
            "selected": count,
            "excluded": len(candidates) - count,
            "target_sources": dict(
                Counter(
                    str(s.questions[0].target.model_dump().get("source"))
                    for s, _ in chosen
                    if s.questions[0].target is not None
                )
            ),
        }
    output.mkdir(mode=0o700)
    media_root = output / "media"
    media_root.mkdir(mode=0o700)
    normalized: dict[str, list[Sample]] = {}
    for modality, chosen in selected.items():
        normalized[modality] = []
        for original, path in chosen:
            sha = original.prefix.media[0].sha256
            target = media_root / (sha + path.suffix)
            if not target.exists():
                copy_media(path, target, sha)
            row = original.model_dump(mode="json")
            row["prefix"]["media"][0]["uri"] = target.relative_to(output).as_posix()
            normalized[modality].append(Sample.model_validate(row))
    pairs = sorted(
        zip(normalized["image"], normalized["audio"], strict=True),
        key=lambda p: rank("rule:" + p[0].id + ":" + p[1].id),
    )
    samples, components = [], []
    for index, (image, audio) in enumerate(pairs):
        rule = (
            "both_true"
            if index < count // 2
            else "image_false_audio_true"
            if index < 3 * count // 4
            else "image_true_audio_false"
        )
        sample, component = compose(image, audio, output, "train", rule, False)
        row = sample.model_dump(mode="json")
        row["id"] = "crossmodal-train-" + sample.meta.source_sha256[:24]
        row["meta"]["data_version"] = VERSION
        sample = Sample.model_validate(row)
        component["item_id"] = sample.id
        samples.append(sample)
        components.append(component)
    allocations = allocation_rows({"train": samples})
    original_hashes = {sample.id: canonical_sample_sha(sample) for sample in samples}
    for allocation in allocations:
        allocation["original_sample_sha256"] = original_hashes[allocation["item_id"]]
    outputs = {
        "train": write_rows(output / "train.jsonl", [s.model_dump(mode="json") for s in samples]),
        "components": write_rows(output / "components.jsonl", components),
        "allocation": write_rows(output / "allocation.jsonl", allocations),
    }
    combined = inventory.model_dump(mode="json", exclude_none=True)
    combined["samples"].append(
        {
            "path": str(output / "train.jsonl"),
            "sha256": outputs["train"],
            "cohort": "crossmodal",
            "exposure": "current",
            "expected_pool": "train",
        }
    )
    combined["component_ledgers"].append(
        {"path": str(output / "components.jsonl"), "sha256": outputs["components"]}
    )
    private_write(output / "inventory-manifest.json", json.dumps(combined, sort_keys=True))
    if audit_manifest(output / "inventory-manifest.json").outcome != "isolated":
        raise ValueError("composed_isolation")
    outputs["inventory"] = digest((output / "inventory-manifest.json").read_bytes())
    result = {
        "schema_version": VERSION,
        "count": count,
        "seed": 42,
        "counts": counts,
        "rules": dict(Counter(c["proposal_rule"] for c in components)),
        "input_sha256": hashes,
        "output_sha256": outputs,
        "code_sha256": digest(Path(__file__).read_bytes()),
        "independent_observations": "not_claimed",
        "quality_admission": "not_claimed",
    }
    private_write(
        output / "provenance.json",
        json.dumps({**result, "input_bindings": config}, sort_keys=True),
    )
    outputs["provenance"] = digest((output / "provenance.json").read_bytes())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = PrivateParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        real_path(args.report)
        if args.report.exists():
            raise ValueError("report_exists")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = build(args.manifest, args.output_dir, count=args.count)
        private_write(args.report, json.dumps({"outcome": "built", **result}, sort_keys=True))
        return 0
    except Exception:
        if args.report.is_absolute() and not args.report.exists():
            with contextlib.suppress(Exception):
                real_path(args.report)
                private_write(
                    args.report,
                    json.dumps({"outcome": "rejected", "categories": ["build_rejected"]}),
                )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
