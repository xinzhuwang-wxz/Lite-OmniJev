"""Private GUI original-CLICK gold builder; aggregate receipts only.

Manifest binds plan, source_index, inventory, quarantine, current_reserves and
absolute annotation_root/media_root (the latter contains screenshots/).
The source index binds every annotation and screenshot to the approved revision.
No network, GPU, model scores, or generated source labels are used.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from pydantic import Field

from data.schema import Contract, Sample, Sha256
from scripts import s0_gui_dev as gui
from scripts.clevrer_gold_pools import (
    BuildError,
    Quarantine,
    bound_json,
    encode,
    file_digest,
    rank,
    verify,
    write_private,
)
from scripts.clotho_gold_pools import subset_sum
from scripts.gold_inventory import (
    Artifact,
    ComponentRecord,
    Manifest,
    PrivateParser,
    Rejected,
    SampleArtifact,
    audit_manifest,
    bound_lines,
    digest,
    extract,
)
from scripts.gui_odyssey_media import REVISION
from scripts.s1_text_pools import canonical_family_hash

PLAN_SHA = "661a56214ee442ae0c694b21c41b374fc02dad0ea1a962bb8d9638b3d393c0c4"
SOURCE_INDEX_SHA = "35398d33d0121f4f6cc13cbb5c492cca5d263b27901d69303c198e9ebed8d658"


class BuildManifest(Contract):
    schema_version: Literal["gui-gold-build-v1"]
    plan: Artifact
    source_index: Artifact
    inventory: Artifact
    quarantine: Artifact
    current_reserves: list[Artifact] = Field(min_length=1)
    annotation_root: str
    media_root: str


class SourceIndex(Contract):
    schema_version: Literal["gui-gold-source-index-v1"]
    revision: Literal["61632d0f3f4d51d7e9561ce4f84347dd06b2019d"]
    license: Literal["CC-BY-4.0"]
    annotations: dict[str, Sha256]
    media: dict[str, Sha256]


class Receipt(Contract):
    schema_version: Literal["gui-gold-receipt-v1"] = "gui-gold-receipt-v1"
    outcome: Literal["built", "rejected"] = "rejected"
    mode: Literal["real", "canary"] = "real"
    real_source_verified: bool = False
    full_gold_gate: Literal["not_validated"] = "not_validated"
    private_job_permissions: Literal["not_validated"] = "not_validated"
    inventory_completeness: Literal["not_validated"] = "not_validated"
    historical_rgb_coverage: Literal["complete_supplied_gui_media", "gap"] = "gap"
    source_revision: str = REVISION
    seed: Literal[42] = 42
    category: str = "job_failed"
    counts: dict[str, int] = Field(default_factory=dict)
    input_sha256: dict[str, str] = Field(default_factory=dict)
    output_sha256: dict[str, str] = Field(default_factory=dict)
    code_sha256: str = ""
    code_commit: str = "unavailable"
    runtime: str = ""
    elapsed_seconds: float = 0
    cost_usd: None = None
    exit_code: int = 1


class Components:
    def __init__(self) -> None:
        self.parents: dict[str, str] = {}

    def find(self, node: str) -> str:
        self.parents.setdefault(node, node)
        if self.parents[node] != node:
            self.parents[node] = self.find(self.parents[node])
        return self.parents[node]

    def union(self, a: str, b: str) -> None:
        left, right = sorted((self.find(a), self.find(b)))
        self.parents[right] = left


def safe_path(path: Path) -> Path:
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise BuildError("unsafe_path")
    return path


def verified_media(item: tuple[Path, str | None]) -> tuple[Path, str]:
    path, expected = item
    actual = file_digest(safe_path(path))
    if actual != expected:
        raise BuildError("media_source_mismatch")
    return path, actual


def decoded_media(item: tuple[Path, str]) -> tuple[str, dict[str, Any]]:
    path, expected = item
    verified_media((path, expected))
    info = gui.image_info(path)
    if info["sha256"] != expected:
        raise BuildError("media_changed")
    return expected, info


def collect(
    manifest: BuildManifest,
    graph: Components,
    receipt: Receipt | None = None,
    *,
    workers: int = 8,
) -> dict[str, dict[str, Any]]:
    index = SourceIndex.model_validate(bound_json(manifest.source_index))
    annotation_root = safe_path(Path(manifest.annotation_root))
    media_root = safe_path(Path(manifest.media_root))
    result: dict[str, dict[str, Any]] = {}
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    source_steps: list[tuple[str, dict[str, Any], Path]] = []
    scanned_steps = 0
    scanned_episodes = 0
    for entry in gui.load_plan(Path(manifest.plan.path)):
        scanned_episodes += 1
        episode = entry["episode_id"]
        path = safe_path(annotation_root / f"{episode}.json")
        if index.annotations.get(episode) != file_digest(path):
            raise BuildError("annotation_source_mismatch")
        annotation = gui.load_annotation(annotation_root, episode)
        gui.validate_episode_steps(annotation, episode, media_root)
        for step in annotation["steps"]:
            path = safe_path(media_root / "screenshots" / step["screenshot"])
            source_steps.append((episode, step, path))
            scanned_steps += 1
    # Only paths, step records and decoded metadata cross the process boundary.
    # Verify every alias before deduplicating by its pinned raw content hash.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        verified = pool.map(
            verified_media,
            ((path, index.media.get(step["screenshot"])) for _, step, path in source_steps),
            chunksize=16,
        )
        unique: dict[str, Path] = {}
        for path, media_sha in verified:
            unique.setdefault(media_sha, path)
        decoded_cache = dict(
            pool.map(decoded_media, ((path, sha) for sha, path in unique.items()), chunksize=1)
        )
    for episode, step, path in source_steps:
        family_node = "f:" + canonical_family_hash(f"gui:{episode}")
        info = decoded_cache[index.media[step["screenshot"]]]
        # Every original step contributes aliases, even if it cannot be a gold item.
        graph.union(family_node, "m:" + info["sha256"])
        graph.union(family_node, "r:" + info["decoded_rgb_sha256"])
        if step.get("action") != "CLICK":
            continue
        instruction = str(step.get("low_level_instruction", "")).strip()
        if not instruction:
            continue
        try:
            box = gui.parse_box(step.get("sam2_bbox"))
            point = gui.parse_click_point(step.get("info"))
        except ValueError:
            continue
        if not gui.box_is_legal(
            box, width=info["width"], height=info["height"]
        ) or not gui.contains(box, point):
            continue
        negatives = gui.candidate_negative_boxes(
            box, point, width=info["width"], height=info["height"]
        )
        if len(negatives) != 3:
            continue
        row = dict(
            episode_id=episode,
            step=step["step"],
            instruction=instruction,
            screenshot=step["screenshot"],
            media_path=path,
            image=info,
            true_box=box,
            point=point,
            negative_boxes=negatives,
            raw_step_sha256=digest(gui.canonical_json(step).encode()),
            annotation_sha256=index.annotations[episode],
        )
        sample, _, _ = gui.sample_for(row)
        result[sample.id] = row
    if receipt is not None:
        receipt.counts.update(
            source_episodes_scanned=scanned_episodes,
            source_steps_aliased=scanned_steps,
            source_unique_media_decoded=len(decoded_cache),
            eligible_original_steps=len(result),
        )
    return result


def inventory_state(
    manifest: BuildManifest, receipt: Receipt, graph: Components
) -> tuple[Manifest, dict[str, set[str]], dict[str, Sample]]:
    if audit_manifest(Path(manifest.inventory.path)).outcome != "isolated":
        raise BuildError("historical_pool_conflict")
    inventory = Manifest.model_validate(bound_json(manifest.inventory))
    replaced = {(a.path, a.sha256) for a in manifest.current_reserves}
    if len(replaced) != len(manifest.current_reserves):
        raise BuildError("invalid_reserve_binding")
    found: set[tuple[str, str]] = set()
    assignments: dict[str, set[str]] = defaultdict(set)
    existing: dict[str, Sample] = {}
    decoded: set[str] = set()
    missing: set[str] = set()
    for i, artifact in enumerate(inventory.samples):
        replace = (artifact.path, artifact.sha256) in replaced
        if replace:
            found.add((artifact.path, artifact.sha256))
        for line in bound_lines(artifact):
            sample = Sample.model_validate_json(line)
            extracted = extract(sample, artifact)
            current_gui = (
                artifact.exposure == "current"
                and extracted.cohort == "screenshot"
                and sample.meta.pool in {"calibration", "holdout"}
            )
            if current_gui != replace:
                raise BuildError("invalid_reserve_binding")
            node = "f:" + canonical_family_hash(sample.meta.family_id)
            assignments[node].add(sample.meta.pool)
            for media in sample.prefix.media:
                graph.union(node, "m:" + media.sha256)
                if extracted.cohort != "screenshot":
                    continue
                if media.sha256 in decoded:
                    continue
                uri = urlparse(media.uri)
                local = Path(unquote(uri.path)) if uri.scheme == "file" else Path(media.uri)
                if not local.is_absolute() or not local.is_file():
                    missing.add(media.sha256)
                    continue
                safe_path(local)
                info = gui.image_info(local)
                if info["sha256"] != media.sha256:
                    raise BuildError("historical_media_mismatch")
                graph.union("m:" + media.sha256, "r:" + info["decoded_rgb_sha256"])
                decoded.add(media.sha256)
            if replace:
                if sample.id in existing:
                    raise BuildError("duplicate_existing_identity")
                existing[sample.id] = sample
        receipt.input_sha256[f"inventory_sample_{i}"] = artifact.sha256
    if found != replaced:
        raise BuildError("invalid_reserve_binding")
    for i, component_artifact in enumerate(inventory.component_ledgers):
        for line in bound_lines(component_artifact):
            record = ComponentRecord.model_validate_json(line)
            for component in (record.image, record.audio):
                node = "f:" + canonical_family_hash(component.family_id)
                assignments[node].add(record.pool)
                graph.union(node, "m:" + component.media_sha256)
            assignments["f:" + record.crossmodal_family_hash].add(record.pool)
        receipt.input_sha256[f"inventory_component_{i}"] = component_artifact.sha256
    receipt.counts["historical_gui_media_rgb_verified"] = len(decoded)
    receipt.counts["historical_gui_media_rgb_unavailable"] = len(missing - decoded)
    receipt.historical_rgb_coverage = "gap" if missing - decoded else "complete_supplied_gui_media"
    return inventory, assignments, existing


def allocate(
    rows: dict[str, dict[str, Any]],
    existing: dict[str, Sample],
    graph: Components,
    assignments: dict[str, set[str]],
    quarantine: Quarantine,
    calibration_count: int,
    holdout_count: int,
) -> dict[str, list[str]]:
    pools: dict[str, set[str]] = defaultdict(set)
    for node, assigned in assignments.items():
        pools[graph.find(node)].update(assigned)
    if any(len(p) > 1 for p in pools.values()):
        raise BuildError("historical_pool_conflict")
    blocked = {graph.find("f:" + f) for f in quarantine.families} | {
        graph.find("m:" + m) for m in quarantine.media
    }
    output: dict[str, list[str]] = {"calibration": [], "holdout": []}
    seen: set[str] = set()
    for item_id, sample in existing.items():
        row = rows.get(item_id)
        if row is None:
            raise BuildError("invalid_existing_gold")
        expected, _, _ = gui.sample_for(row)
        component = graph.find("f:" + canonical_family_hash(sample.meta.family_id))
        if (
            sample.questions != expected.questions
            or sample.meta.source_sha256 != expected.meta.source_sha256
            or sample.meta.source_dataset != gui.DATASET
            or sample.meta.source_revision != REVISION
            or sample.meta.license != gui.ANNOTATION_LICENSE
            or sample.meta.family_id != expected.meta.family_id
            or len(sample.prefix.media) != 1
            or sample.prefix.media[0].sha256 != row["image"]["sha256"]
            or component in blocked
            or pools[component] != {sample.meta.pool}
            or row["image"]["decoded_rgb_sha256"] in seen
        ):
            raise BuildError("invalid_existing_gold")
        seen.add(row["image"]["decoded_rgb_sha256"])
        output[sample.meta.pool].append(item_id)
    candidates: dict[str, list[str]] = defaultdict(list)
    for item_id, row in sorted(rows.items()):
        component = graph.find("f:" + canonical_family_hash(f"gui:{row['episode_id']}"))
        rgb = row["image"]["decoded_rgb_sha256"]
        # Historical reserve components retain their assignments but are not expanded.
        if component in blocked or pools.get(component) or rgb in seen:
            continue
        seen.add(rgb)
        candidates[component].append(item_id)
    need_cal = calibration_count - len(output["calibration"])
    need_hold = holdout_count - len(output["holdout"])
    sizes = {key: len(ids) for key, ids in candidates.items()}
    if min(need_cal, need_hold) < 0 or sum(sizes.values()) < need_cal + need_hold:
        raise BuildError("insufficient_capacity")
    # Exact whole-component capacity; truncation is only within its assigned pool.
    reachable = 1
    for size in sizes.values():
        reachable |= reachable << size
    capacity = next(
        (n for n in range(need_hold, sum(sizes.values()) - need_cal + 1) if (reachable >> n) & 1),
        None,
    )
    if capacity is None:
        raise BuildError("insufficient_capacity")
    selected = subset_sum(sizes, capacity)
    for component, ids in candidates.items():
        if component in selected:
            output["holdout"].extend(ids)
    mandatory_hold = [i for i in output["holdout"] if i in existing]
    new_hold = sorted(
        (i for i in output["holdout"] if i not in existing),
        key=lambda i: (rank("gui-holdout", i), i),
    )
    output["holdout"] = mandatory_hold + new_hold[:need_hold]
    cal = [i for c, ids in candidates.items() if c not in selected for i in ids]
    output["calibration"].extend(
        sorted(cal, key=lambda i: (rank("gui-calibration", i), i))[:need_cal]
    )
    if len(output["calibration"]) != calibration_count or len(output["holdout"]) != holdout_count:
        raise BuildError("insufficient_capacity")
    return output


def build(args: argparse.Namespace, receipt: Receipt) -> None:
    started = time.monotonic()
    payload = args.manifest.read_bytes()
    receipt.input_sha256["build_manifest"] = digest(payload)
    manifest = BuildManifest.model_validate_json(payload)
    real = not args.canary
    if min(args.calibration_count, args.holdout_count) < 1 or (
        real and (args.calibration_count, args.holdout_count) != (3000, 1000)
    ):
        raise BuildError("invalid_quota")
    if real and (
        manifest.plan.sha256 != PLAN_SHA or manifest.source_index.sha256 != SOURCE_INDEX_SHA
    ):
        raise BuildError("source_not_pinned")
    for name in ("plan", "source_index", "inventory", "quarantine"):
        artifact = getattr(manifest, name)
        safe_path(Path(artifact.path))
        verify(artifact)
        receipt.input_sha256[name] = artifact.sha256
    graph = Components()
    inventory, assignments, existing = inventory_state(manifest, receipt, graph)
    rows = collect(manifest, graph, receipt)
    if real and {
        p: sum(s.meta.pool == p for s in existing.values()) for p in ("calibration", "holdout")
    } != {"calibration": 119, "holdout": 59}:
        raise BuildError("invalid_existing_count")
    selected = allocate(
        rows,
        existing,
        graph,
        assignments,
        Quarantine.model_validate(bound_json(manifest.quarantine)),
        args.calibration_count,
        args.holdout_count,
    )
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    media_dir = args.output_dir / "media"
    media_dir.mkdir(mode=0o700)
    artifacts = []
    ledger = []
    for pool, ids in selected.items():
        samples = []
        for item_id in sorted(ids):
            row = rows[item_id]
            if file_digest(row["media_path"]) != row["image"]["sha256"]:
                raise BuildError("media_changed")
            if item_id in existing:
                sample = existing[item_id]
            else:
                sample, _, _ = gui.sample_for(row)
                destination = media_dir / (row["image"]["sha256"] + ".png")
                write_private(destination, row["media_path"].read_bytes())
                raw = sample.model_dump(mode="json")
                raw["prefix"]["media"][0]["uri"] = destination.relative_to(
                    args.output_dir
                ).as_posix()
                raw["meta"].update(
                    pool=pool,
                    data_version="gui-original-gold-v1" if real else "gui-canary-not-real-gold-v1",
                    usage="commercial" if real else "research",
                )
                sample = Sample.model_validate(raw)
            samples.append(sample)
            ledger.append(
                dict(
                    item_id=item_id,
                    family_id=sample.meta.family_id,
                    family_hash=canonical_family_hash(sample.meta.family_id),
                    pool=pool,
                    preserved=item_id in existing,
                    episode_id=row["episode_id"],
                    step=row["step"],
                    annotation_sha256=row["annotation_sha256"],
                    raw_step_sha256=row["raw_step_sha256"],
                    source_sha256=sample.meta.source_sha256,
                    media_sha256=row["image"]["sha256"],
                    decoded_rgb_sha256=row["image"]["decoded_rgb_sha256"],
                    component=graph.find("f:" + canonical_family_hash(sample.meta.family_id)),
                )
            )
        path = args.output_dir / f"{pool}.jsonl"
        write_private(path, b"".join((s.model_dump_json() + "\n").encode() for s in samples))
        artifacts.append(
            SampleArtifact(
                path=str(path),
                sha256=file_digest(path),
                cohort="screenshot",
                exposure="current",
                expected_pool="calibration" if pool == "calibration" else "holdout",
            )
        )
        receipt.counts[pool] = len(samples)
    replaced = {(a.path, a.sha256) for a in manifest.current_reserves}
    combined = Manifest(
        schema_version="gold-inventory-manifest-v1",
        samples=[
            a.model_copy(update={"exposure": "historical"}) if (a.path, a.sha256) in replaced else a
            for a in inventory.samples
        ]
        + artifacts,
        component_ledgers=inventory.component_ledgers,
    )
    write_private(args.output_dir / "inventory-manifest.json", combined.model_dump_json().encode())
    write_private(args.output_dir / "allocation.jsonl", b"".join(encode(r) for r in ledger))
    write_private(args.output_dir / "build-manifest.json", payload)
    for name in (
        "calibration.jsonl",
        "holdout.jsonl",
        "inventory-manifest.json",
        "allocation.jsonl",
        "build-manifest.json",
    ):
        receipt.output_sha256[name] = file_digest(args.output_dir / name)
    if audit_manifest(args.output_dir / "inventory-manifest.json").outcome != "isolated":
        raise BuildError("output_isolation_failed")
    receipt.counts["preserved_items"] = len(existing)
    receipt.counts["new_original_steps"] = sum(map(len, selected.values())) - len(existing)
    receipt.real_source_verified = real
    provenance = dict(
        schema_version="gui-gold-provenance-v1",
        mode=receipt.mode,
        inputs=manifest.model_dump(mode="json"),
        outputs=receipt.output_sha256,
        code_sha256=receipt.code_sha256,
        code_commit=receipt.code_commit,
        runtime=receipt.runtime,
        seed=42,
        cost_usd=None,
        exit_code=0,
        elapsed_seconds=round(time.monotonic() - started, 6),
        new_sample_media_root=str(args.output_dir),
        source_attestation="sha_bound_observed_cache_not_independent_upstream_attestation",
        historical_rgb_coverage=receipt.historical_rgb_coverage,
    )
    write_private(args.output_dir / "provenance.json", encode(provenance))
    receipt.output_sha256["provenance.json"] = file_digest(args.output_dir / "provenance.json")
    receipt.outcome, receipt.category, receipt.exit_code = "built", "none", 0


def main(argv: list[str] | None = None) -> int:
    start = time.monotonic()
    receipt = Receipt(runtime=f"Python {platform.python_version()} / {platform.system()}")
    args = None
    try:
        receipt.code_sha256 = file_digest(Path(__file__))
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode == 0:
            receipt.code_commit = commit.stdout.strip()
        parser = PrivateParser(description=__doc__)
        for name in ("manifest", "output-dir", "report"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument("--canary", action="store_true")
        parser.add_argument("--calibration-count", type=int, default=3000)
        parser.add_argument("--holdout-count", type=int, default=1000)
        args = parser.parse_args(argv)
        receipt.mode = "canary" if args.canary else "real"
        if args.canary:
            receipt.source_revision = "synthetic-canary"
        paths = [args.manifest, args.output_dir, args.report]
        if any(not path.is_absolute() for path in paths) or len({p.resolve() for p in paths}) != 3:
            raise BuildError("invalid_arguments")
        if any(p.exists() or p.is_symlink() for p in [args.output_dir, args.report]):
            raise BuildError("output_exists")
        for path in paths:
            safe_path(path)
        build(args, receipt)
    except BuildError as error:
        receipt.category = str(error)
    except Rejected as error:
        receipt.category = error.category
    except (ValueError, KeyError, TypeError):
        receipt.category = "invalid_metadata"
    except OSError:
        receipt.category = "io_error"
    except Exception:
        receipt.category = "job_failed"
    receipt.elapsed_seconds = round(time.monotonic() - start, 6)
    if args is not None and args.report.is_absolute():
        try:
            write_private(args.report, receipt.model_dump_json().encode() + b"\n")
        except Exception:
            receipt.outcome, receipt.category, receipt.exit_code = "rejected", "report_io", 1
    print(receipt.model_dump_json())
    return receipt.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
