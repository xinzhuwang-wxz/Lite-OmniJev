"""Prepare three controlled TRAIN claim variants per existing component pair.

No inventory admission or training is performed. Full instructions change.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import HardTarget, Sample
from scripts.crossmodal_gold_pools import compose, legal_target
from scripts.gold_inventory import Manifest, PrivateParser, digest
from scripts.gold_media_bundle import copy_media, media_path, private_write, real_path
from scripts.s1_text_pools import allocation_rows, canonical_sample_sha
from scripts.s4_crossmodal_train import SOURCES, write_rows
from scripts.s4_crossmodal_train import VERSION as ORIGINAL_VERSION
from scripts.vqa2_gold_capacity import bound

VERSION = "crossmodal-controlled-train-pairs-v1"
RULES = ("both_true", "image_false_audio_true", "image_true_audio_false")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def keyed(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), "duplicate_identity")
    return result


def build(original: Path, provenance_sha256: str, output: Path) -> dict[str, Any]:
    for path in (original, output):
        real_path(path)
    require(not output.exists(), "output_exists")
    provenance_bytes = bound(
        {"path": str(original / "provenance.json"), "sha256": provenance_sha256}
    )
    provenance = json.loads(provenance_bytes)
    require(provenance["schema_version"] == ORIGINAL_VERSION, "original_version")
    bindings = provenance["input_bindings"]
    inventory = Manifest.model_validate_json(bound(bindings["inventory"]))
    require(len(bindings["sources"]) == 2, "source_count")
    originals: dict[str, dict[str, Sample]] = {}
    roots = {}
    for source in bindings["sources"]:
        modality = source["modality"]
        require(modality in SOURCES and modality not in originals, "source_modality")
        registered = [
            a
            for a in inventory.samples
            if a.path == source["path"]
            and a.sha256 == source["sha256"]
            and a.expected_pool == "train"
        ]
        require(len(registered) == 1, "source_not_registered_train")
        rows = [
            Sample.model_validate_json(line)
            for line in bound({k: source[k] for k in ("path", "sha256")}).splitlines()
            if line.strip()
        ]
        require(len({s.id for s in rows}) == len(rows), "duplicate_source_id")
        for sample in rows:
            require(
                sample.meta.pool == "train"
                and sample.meta.usage == "commercial"
                and sample.meta.modality == [modality],
                "source_contract",
            )
            require(
                len(sample.prefix.media) == 1 and sample.prefix.media[0].type == modality,
                "source_media",
            )
            legal_target(sample, modality)
            target = sample.questions[0].target
            if not isinstance(target, HardTarget):
                raise ValueError("source_hard_gold")
            require(target.source in SOURCES[modality], "source_gold")
        originals[modality] = {s.id: s for s in rows}
        roots[modality] = Path(source["media_root"])
        real_path(roots[modality])
    loaded = {}
    for name in ("train", "components", "allocation"):
        raw = bound(
            {"path": str(original / (name + ".jsonl")), "sha256": provenance["output_sha256"][name]}
        )
        loaded[name] = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ss, cc, aa = (
        keyed(loaded[n], k)
        for n, k in (("train", "id"), ("components", "item_id"), ("allocation", "item_id"))
    )
    require(0 < len(ss) <= 200 and ss.keys() == cc.keys() == aa.keys(), "original_coverage")
    require(len(ss) == provenance["count"], "original_count")
    pairs = []
    used: dict[str, set[str]] = {m: set() for m in SOURCES}
    for ident in sorted(ss):
        sample = Sample.model_validate(ss[ident])
        c, a = cc[ident], aa[ident]
        require(sample.meta.pool == c["pool"] == a["pool"] == "train", "original_pool")
        require(
            canonical_sample_sha(sample)
            == a["derived_sample_sha256"]
            == a["original_sample_sha256"],
            "original_allocation",
        )
        normalized = []
        for modality in ("image", "audio"):
            require(c[modality]["item_id"] in originals[modality], "component_missing")
            component = originals[modality][c[modality]["item_id"]]
            require(component.id not in used[modality], "component_reuse")
            used[modality].add(component.id)
            # Normalize only URI, exactly as the original builder did; keep gold intact.
            row = component.model_dump(mode="json")
            expected = component.prefix.media[0].sha256
            candidates = [m for m in sample.prefix.media if m.type == modality]
            require(len(candidates) == 1 and candidates[0].sha256 == expected, "component_media")
            source_media = media_path([roots[modality]], component.prefix.media[0].uri)
            require(digest(source_media.read_bytes()) == expected, "source_media_sha")
            row["prefix"]["media"][0]["uri"] = candidates[0].uri
            normalized.append(Sample.model_validate(row))
        require(c["proposal_rule"] in RULES, "original_rule")
        rebuilt, ledger = compose(
            normalized[0], normalized[1], original, "train", c["proposal_rule"], False
        )
        row = rebuilt.model_dump(mode="json")
        row["id"], row["meta"]["data_version"] = ident, ORIGINAL_VERSION
        ledger["item_id"] = ident
        require(row == sample.model_dump(mode="json") and ledger == c, "original_reconstruction")
        require(sample.meta.family_id == a["family_id"], "original_family")
        pairs.append((ident, normalized))
    # All original bindings verified before creating the output directory.
    output.mkdir(mode=0o700)
    (output / "media").mkdir(mode=0o700)
    samples, components, groups = [], [], []
    for ident, normalized in pairs:
        for component in normalized:
            media = component.prefix.media[0]
            src = original / media.uri
            dst = output / media.uri
            require(dst.parent == output / "media", "normalized_media_path")
            if not dst.exists():
                copy_media(src, dst, media.sha256)
        group: dict[str, Any] = {
            "group_id": "crossmodal-pair-"
            + digest(
                normalized[0].meta.family_id.encode() + b":" + normalized[1].meta.family_id.encode()
            ),
            "original_item_id": ident,
            "variants": {},
            "component_gold": {
                m: legal_target(s, m) for m, s in zip(("image", "audio"), normalized, strict=True)
            },
            "source_component_sha256": {
                m: canonical_sample_sha(originals[m][s.id])
                for m, s in zip(("image", "audio"), normalized, strict=True)
            },
        }
        for rule in RULES:
            sample, c = compose(normalized[0], normalized[1], output, "train", rule, False)
            row = sample.model_dump(mode="json")
            row["id"] = "crossmodal-pair-train-" + sample.meta.source_sha256[:24]
            row["meta"]["data_version"] = VERSION
            sample = Sample.model_validate(row)
            c["item_id"] = sample.id
            samples.append(sample)
            components.append(c)
            group["variants"][rule] = sample.id
        groups.append(group)
    require(len({s.id for s in samples}) == len(samples), "new_id_collision")
    allocations = allocation_rows({"train": samples})
    shas = {s.id: canonical_sample_sha(s) for s in samples}
    for a in allocations:
        a["original_sample_sha256"] = shas[a["item_id"]]
    outputs = {
        "train": write_rows(output / "train.jsonl", [s.model_dump(mode="json") for s in samples]),
        "components": write_rows(output / "components.jsonl", components),
        "allocation": write_rows(output / "allocation.jsonl", allocations),
        "groups": write_rows(output / "groups.jsonl", groups),
    }
    result = {
        "schema_version": VERSION,
        "status": "new_prepared_not_admitted",
        "count": len(samples),
        "group_count": len(groups),
        "rules": dict(Counter(c["proposal_rule"] for c in components)),
        "input_root": str(original),
        "provenance_sha256": provenance_sha256,
        "input_sha256": provenance["output_sha256"],
        "source_bindings": bindings,
        "output_sha256": outputs,
        "code_sha256": digest(Path(__file__).read_bytes()),
        "full_inventory_isolation": "pending_root_exposure_003",
        "semantic_boundary": (
            "Full instruction changes; component gold does not. "
            "Three correlated variants per TRAIN family."
        ),
        "training": "not_run",
    }
    private_write(output / "provenance.json", json.dumps(result, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = PrivateParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--provenance-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        real_path(args.report)
        require(not args.report.exists(), "report_exists")
        result = build(args.original_dir, args.provenance_sha256, args.output_dir)
        private_write(args.report, json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        if not args.report.exists():
            private_write(
                args.report,
                json.dumps(
                    {
                        "status": "rejected",
                        "error_type": type(exc).__name__,
                        "output_may_be_partial": args.output_dir.exists(),
                    }
                ),
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
