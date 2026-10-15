"""Compose private same-pool gold question identities; never attest IID or admission.

CLI: --manifest ABS --media-root ABS --output-dir ABS --report ABS.
Manifest: inventory {path,sha256}, sources [{path,sha256,pool,dataset}], where
pool is calibration/holdout and dataset is VQA-v2, Clotho-AQA, or LibriSpeech.
Sources must exactly match current inventory artifacts. --mode supplemental-dev
uses only registered dev artifacts (current or historical), with 250 audio items
per source and 500 VQA items. Existing dev artifacts are preserved. Canary quotas
are explicit and never establish real admission.
"""

from __future__ import annotations

import contextlib
import io
import json
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from data.schema import Sample, answer_tokens
from scripts.clevrer_gold_pools import encode, write_private
from scripts.gold_inventory import ComponentRecord, Manifest, PrivateParser, audit_manifest, digest
from scripts.s0_crossmodal_dev import family_id_for, option_text, source_hash_for, target_value
from scripts.s1_text_pools import canonical_family_hash
from scripts.vqa2_gold_capacity import bound
from scripts.vqa2_gold_pools import safe_path, write_rows

VERSION = "crossmodal-gold-pools-v1"
DATASETS = {"VQA-v2": "image", "Clotho-AQA": "audio", "LibriSpeech": "audio"}
QUOTAS = {
    "calibration": {"Clotho-AQA": 1400, "LibriSpeech": 1600},
    "holdout": {"Clotho-AQA": 500, "LibriSpeech": 500},
}


DEV_QUOTAS = {"dev": {"Clotho-AQA": 250, "LibriSpeech": 250}}


def rank(value: str) -> str:
    return digest(("42:" + value).encode())


def media_uri(root: Path, uri: str, expected: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file") or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("unsafe_media")
    decoded = unquote(parsed.path)
    candidate = Path(decoded)
    if ".." in candidate.parts or "\\" in decoded or "\x00" in decoded:
        raise ValueError("unsafe_media")
    path = candidate if candidate.is_absolute() else root / candidate
    safe_path(path)
    if not path.is_relative_to(root) or not path.is_file() or digest(path.read_bytes()) != expected:
        raise ValueError("media_binding")
    return path.relative_to(root).as_posix()


def legal_target(sample: Sample, modality: str) -> str:
    if len(sample.questions) != 1:
        raise ValueError("question_count")
    question = sample.questions[0]
    if question.type not in ("choice", "noul") or (
        modality == "image" and question.type != "choice"
    ):
        raise ValueError("question_type")
    gold = target_value(sample)
    if gold == "U" or gold not in answer_tokens(question):
        raise ValueError("target")
    return gold


def proposal(sample: Sample, truth: bool) -> str:
    modality = "image" if sample.meta.modality == ["image"] else "audio"
    gold = legal_target(sample, modality)
    if truth:
        return gold
    options = [token for token in answer_tokens(sample.questions[0]) if token not in (gold, "U")]
    if not options:
        raise ValueError("no_false_proposal")
    return min(options, key=lambda token: rank(sample.id + ":" + token))


def claim(sample: Sample, token: str) -> str:
    question = sample.questions[0]
    if question.type == "choice":
        options = ", ".join(
            f"{t}: {text}"
            for t, text in zip(answer_tokens(question), question.options or [], strict=True)
        )
        return (
            f"the answer to '{question.instruction}' with options [{options}] "
            f"is {token}: {option_text(sample, token)}"
        )
    return f"the answer to '{question.instruction}' is {token}"


def compose(
    image: Sample, audio: Sample, root: Path, pool: str, rule: str, canary: bool
) -> tuple[Sample, dict[str, Any]]:
    image_proposal = proposal(image, rule != "image_false_audio_true")
    audio_proposal = proposal(audio, rule != "image_true_audio_false")
    gold = "yes" if rule == "both_true" else "no"
    source_hash = source_hash_for(
        image=image,
        audio=audio,
        image_proposal=image_proposal,
        audio_proposal=audio_proposal,
        rule=rule,
    )
    sample = Sample.model_validate(
        {
            "id": "crossmodal-gold-" + source_hash[:24],
            "prefix": {
                "state": "Answer using both the attached image and audio.",
                "media": [
                    {"type": m.type, "uri": media_uri(root, m.uri, m.sha256), "sha256": m.sha256}
                    for original in (image, audio)
                    for m in original.prefix.media
                ],
            },
            "questions": [
                {
                    "qid": "noul",
                    "type": "noul",
                    "instruction": (
                        f"Are BOTH claims true? (1) In the image, {claim(image, image_proposal)}. "
                        f"(2) For the audio, {claim(audio, audio_proposal)}."
                    ),
                    "target": {
                        "kind": "hard",
                        "value": gold,
                        "source": "programmatic_component_answer_and",
                    },
                }
            ],
            "meta": {
                "modality": ["image", "audio"],
                "source_dataset": image.meta.source_dataset + "+" + audio.meta.source_dataset,
                "source_revision": image.meta.source_revision + "+" + audio.meta.source_revision,
                "source_sha256": source_hash,
                "license": "image:" + image.meta.license + ";audio:" + audio.meta.license,
                "usage": "research" if canary else "commercial",
                "pool": pool,
                "family_id": family_id_for(image, audio),
                "data_version": VERSION + ("-canary" if canary else ""),
            },
        }
    )
    component: dict[str, Any] = {
        "item_id": sample.id,
        "crossmodal_family_id": sample.meta.family_id,
        "crossmodal_family_hash": canonical_family_hash(sample.meta.family_id),
        "pool": pool,
        "proposal_rule": rule,
        "program_truth": gold,
        "component_family_separate_ledger": True,
    }
    for modality, original, token in (
        ("image", image, image_proposal),
        ("audio", audio, audio_proposal),
    ):
        component[modality] = {
            "item_id": original.id,
            "family_id": original.meta.family_id,
            "family_hash": canonical_family_hash(original.meta.family_id),
            "source_hash": original.meta.source_sha256,
            "media_sha256": original.prefix.media[0].sha256,
            "proposal": token,
            "proposal_option_text_sha256": digest(option_text(original, token).encode())
            if original.questions[0].type == "choice"
            else None,
        }
    ComponentRecord.model_validate(component)
    return sample, component


def dependency_count(pairs: list[tuple[Sample, Sample]]) -> int:
    parents: dict[str, str] = {}

    def find(key: str) -> str:
        parents.setdefault(key, key)
        root = key
        while parents[root] != root:
            root = parents[root]
        while parents[key] != key:
            parent = parents[key]
            parents[key] = root
            key = parent
        return root

    for image, audio in pairs:
        keys = [
            key
            for sample in (image, audio)
            for key in (
                "family:" + canonical_family_hash(sample.meta.family_id),
                "media:" + sample.prefix.media[0].sha256,
            )
        ]
        for key in keys[1:]:
            parents[find(key)] = find(keys[0])
    return len({find(key) for key in parents})


def build(
    manifest_path: Path,
    root: Path,
    output: Path,
    *,
    canary: bool = False,
    quotas: dict[str, dict[str, int]] | None = None,
    mode: str = "reserves",
) -> dict[str, Any]:
    for path in (manifest_path, root, output):
        safe_path(path)
    if output.exists() or not root.is_dir():
        raise ValueError("output_path")
    if mode not in {"reserves", "supplemental-dev"}:
        raise ValueError("mode")
    expected_quotas = DEV_QUOTAS if mode == "supplemental-dev" else QUOTAS
    quotas = expected_quotas if quotas is None else quotas
    if (
        set(quotas) != set(expected_quotas)
        or any(
            set(q) != {"Clotho-AQA", "LibriSpeech"}
            or any(type(n) is not int or n < 1 for n in q.values())
            or sum(q.values()) % 4
            for q in quotas.values()
        )
        or (not canary and quotas != expected_quotas)
    ):
        raise ValueError("quota")
    config = json.loads(manifest_path.read_bytes())
    if set(config) != {"inventory", "sources"}:
        raise ValueError("manifest")
    safe_path(Path(config["inventory"]["path"]))
    inventory = Manifest.model_validate_json(bound(config["inventory"]))
    audit = audit_manifest(Path(config["inventory"]["path"]))
    if audit.outcome != "isolated":
        raise ValueError("inventory_isolation")
    current = {
        (a.path, a.sha256, a.expected_pool, a.cohort)
        for a in inventory.samples
        if a.exposure == "current" or (mode == "supplemental-dev" and a.expected_pool == "dev")
    }
    sources: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    seen_ids: set[str] = set()
    input_hashes = {
        "manifest": digest(manifest_path.read_bytes()),
        "inventory": config["inventory"]["sha256"],
    }
    for index, spec in enumerate(config["sources"]):
        if (
            set(spec) != {"path", "sha256", "pool", "dataset"}
            or spec["dataset"] not in DATASETS
            or spec["pool"] not in expected_quotas
        ):
            raise ValueError("source_declaration")
        modality = DATASETS[spec["dataset"]]
        if (spec["path"], spec["sha256"], spec["pool"], modality) not in current:
            raise ValueError("source_not_current")
        safe_path(Path(spec["path"]))
        payload = bound({k: spec[k] for k in ("path", "sha256")})
        input_hashes[f"source_{index}"] = spec["sha256"]
        for line in payload.splitlines():
            sample = Sample.model_validate_json(line)
            if (
                sample.id in seen_ids
                or sample.meta.pool != spec["pool"]
                or sample.meta.source_dataset != spec["dataset"]
                or sample.meta.modality != [modality]
                or len(sample.prefix.media) != 1
                or sample.prefix.media[0].type != modality
                or (not canary and sample.meta.usage != "commercial")
            ):
                raise ValueError("source_identity")
            legal_target(sample, modality)
            if mode == "supplemental-dev" and sample.questions[0].type != (
                "noul" if spec["dataset"] == "Clotho-AQA" else "choice"
            ):
                raise ValueError("source_task")
            media_uri(root, sample.prefix.media[0].uri, sample.prefix.media[0].sha256)
            seen_ids.add(sample.id)
            sources[(spec["pool"], spec["dataset"])].append(sample)
    selections = {}
    for pool in expected_quotas:
        total = sum(quotas[pool].values())
        images = sorted(sources[(pool, "VQA-v2")], key=lambda s: rank("image:" + s.id))
        audios = []
        for dataset, count in quotas[pool].items():
            candidates = sorted(sources[(pool, dataset)], key=lambda s: rank("audio:" + s.id))
            if len(candidates) < count:
                raise ValueError("source_capacity")
            audios.extend(candidates[:count])
        if len(images) < total:
            raise ValueError("source_capacity")
        audios.sort(key=lambda s: rank("pair-audio:" + s.id))
        pairs = list(zip(images[:total], audios, strict=True))
        pairs.sort(key=lambda pair: rank("rule:" + pair[0].id + ":" + pair[1].id))
        selections[pool] = pairs
    output.mkdir(mode=0o700)
    combined = json.loads(bound(config["inventory"]))
    components = []
    counts: dict[str, Any] = {}
    outputs = {}
    for pool, pairs in selections.items():
        total = len(pairs)
        samples = []
        for index, (image, audio) in enumerate(pairs):
            rule = (
                "both_true"
                if index < total // 2
                else "image_false_audio_true"
                if index < 3 * total // 4
                else "image_true_audio_false"
            )
            sample, ledger = compose(image, audio, root, pool, rule, canary)
            samples.append(sample.model_dump(mode="json"))
            components.append(ledger)
        path = output / f"{pool}.jsonl"
        outputs[pool] = write_rows(path, samples)
        combined["samples"].append(
            {
                "path": str(path),
                "sha256": outputs[pool],
                "cohort": "crossmodal",
                "exposure": "current",
                "expected_pool": pool,
            }
        )
        counts[pool] = {
            "question_items": total,
            "component_dependency_groups": dependency_count(pairs),
            "maximum_audio_family_reuse": max(Counter(a.meta.family_id for _, a in pairs).values()),
            "image_families": len({i.meta.family_id for i, _ in pairs}),
            "audio_families": len({a.meta.family_id for _, a in pairs}),
            "image_media": len({i.prefix.media[0].sha256 for i, _ in pairs}),
            "audio_media": len({a.prefix.media[0].sha256 for _, a in pairs}),
            "audio_sources": dict(Counter(a.meta.source_dataset for _, a in pairs)),
            "rules": dict(Counter(c["proposal_rule"] for c in components if c["pool"] == pool)),
        }
    component_path = output / "components.jsonl"
    outputs["components"] = write_rows(component_path, components)
    combined["component_ledgers"].append(
        {"path": str(component_path), "sha256": outputs["components"]}
    )
    inventory_path = output / "inventory-manifest.json"
    write_private(inventory_path, encode(combined))
    final_audit = audit_manifest(inventory_path)
    if final_audit.outcome != "isolated":
        raise ValueError("composed_isolation")
    outputs["inventory"] = digest(inventory_path.read_bytes())
    return {
        "mode": mode,
        "counts": counts,
        "input_sha256": input_hashes,
        "output_sha256": outputs,
        "dependency_components_global": final_audit.isolation.connected_components
        if final_audit.isolation
        else None,
        "synthetic_canary": canary,
        "effective_sample_size": "not_estimated",
        "independent_observations": "not_claimed",
    }


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "rejected",
        "category": "job_failed",
        "exit_code": 1,
        "schema_version": VERSION,
        "full_gold_gate": "not_validated",
        "private_job_permissions": "not_validated",
        "inventory_completeness": "not_validated",
        "seed": 42,
        "cost_usd": None,
        "code_sha256": digest(Path(__file__).read_bytes()),
        "runtime": f"Python {platform.python_version()} / {platform.system()}",
        "network_bytes": 0,
    }
    args = None
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            parser = PrivateParser(add_help=False)
            for name in ("manifest", "media-root", "output-dir", "report"):
                parser.add_argument("--" + name, type=Path, required=True)
            parser.add_argument(
                "--mode", choices=("reserves", "supplemental-dev"), default="reserves"
            )
            parser.add_argument("--synthetic-canary", action="store_true")
            parser.add_argument("--canary-per-source", type=int)
            args = parser.parse_args(argv)
            safe_path(args.report)
            if args.report.exists():
                raise ValueError("report_exists")
            if (args.canary_per_source is not None) != args.synthetic_canary:
                raise ValueError("canary_quota")
            quotas = (
                {
                    pool: dict.fromkeys(("Clotho-AQA", "LibriSpeech"), args.canary_per_source)
                    for pool in (DEV_QUOTAS if args.mode == "supplemental-dev" else QUOTAS)
                }
                if args.synthetic_canary
                else None
            )
            result.update(
                build(
                    args.manifest,
                    args.media_root,
                    args.output_dir,
                    canary=args.synthetic_canary,
                    quotas=quotas,
                    mode=args.mode,
                )
            )
            result.update(status="built", category="none", exit_code=0)
    except Exception:
        pass
    result["elapsed_seconds"] = round(time.monotonic() - started, 6)
    if args is not None:
        try:
            write_private(args.report, encode(result))
        except Exception:
            result.update(status="rejected", category="report_io", exit_code=1)
    print(json.dumps(result, sort_keys=True))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
