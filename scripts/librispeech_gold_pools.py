"""Private CPU speech-transcription Choice builder; stdout contains aggregate receipts only.

Manifest schema librispeech-gold-build-v1 binds dev_clean, test_clean, inventory,
quarantine as absolute {path, sha256} artifacts. Optional alias_ledgers bind prior
allocation ledgers, required when extending a previous LibriSpeech release.
Real quotas are calibration=1600, holdout=500, supplemental speech dev=500.
Original Clotho dev is unchanged. This is not an audio quality/security gate.
"""

from __future__ import annotations

import argparse
import io
import platform
import re
import subprocess
import tarfile
import time
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import Field

from data.schema import Contract, Sample, Sha256
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
from scripts.gold_inventory import (
    Artifact,
    ComponentRecord,
    Manifest,
    PrivateParser,
    Rejected,
    SampleArtifact,
    bound_lines,
    digest,
    extract,
)
from scripts.s1_text_pools import canonical_family_hash

PINS = {
    "dev_clean": "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
    "test_clean": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
}
GoldPool = Literal["calibration", "holdout", "dev"]
POOLS: tuple[GoldPool, ...] = ("calibration", "holdout", "dev")


class BuildManifest(Contract):
    schema_version: Literal["librispeech-gold-build-v1"]
    dev_clean: Artifact
    test_clean: Artifact
    inventory: Artifact
    quarantine: Artifact
    alias_ledgers: list[Artifact] = Field(default_factory=list)


class Receipt(Contract):
    schema_version: Literal["librispeech-gold-receipt-v1"] = "librispeech-gold-receipt-v1"
    outcome: Literal["built", "rejected"] = "rejected"
    mode: Literal["real", "canary"] = "real"
    real_source_verified: bool = False
    full_gold_gate: Literal["not_validated"] = "not_validated"
    inventory_completeness: Literal["not_validated"] = "not_validated"
    private_job_permissions: Literal["not_validated"] = "not_validated"
    category: str = "job_failed"
    seed: Literal[42] = 42
    source_license: Literal["CC-BY-4.0"] = "CC-BY-4.0"
    source_url: Literal["https://www.openslr.org/12/"] = "https://www.openslr.org/12/"
    input_sha256: dict[str, Sha256] = Field(default_factory=dict)
    output_sha256: dict[str, Sha256] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    code_sha256: str = ""
    code_commit: str = "unavailable"
    runtime: str = ""
    elapsed_seconds: float = 0.0
    cost_usd: None = None
    exit_code: int = 1


class PriorComponent(Contract):
    """Stable identities survive changes in connected-component family hashes."""

    family_id: str
    family_hash: Sha256
    aliases: list[str] = Field(min_length=1)
    pool: Literal["calibration", "holdout", "dev"]
    members: list[dict[str, Any]]


@dataclass
class Utterance:
    identity: str
    archive: str
    member: str
    transcript: str
    transcript_sha256: str
    media_sha256: str
    aliases: set[str]
    eligible: bool


def exclusions(manifest: BuildManifest, receipt: Receipt) -> tuple[set[str], set[str], set[str]]:
    inventory = Manifest.model_validate(bound_json(manifest.inventory))
    families: set[str] = set()
    media: set[str] = set()
    aliases: set[str] = set()
    prior_families: set[str] = set()
    for index, sample_artifact in enumerate(inventory.samples):
        for line in bound_lines(sample_artifact):
            sample = Sample.model_validate_json(line)
            extract(sample, sample_artifact)
            if sample.meta.source_dataset == "LibriSpeech":
                prior_families.add(canonical_family_hash(sample.meta.family_id))
            families.add(canonical_family_hash(sample.meta.family_id))
            media.update(item.sha256 for item in sample.prefix.media)
        receipt.input_sha256[f"inventory_sample_{index}"] = sample_artifact.sha256
    for index, artifact in enumerate(inventory.component_ledgers):
        for line in bound_lines(artifact):
            record = ComponentRecord.model_validate_json(line)
            for component in (record.image, record.audio):
                if component.family_hash != canonical_family_hash(component.family_id):
                    raise BuildError("invalid_component")
                families.add(component.family_hash)
                media.add(component.media_sha256)
            if record.crossmodal_family_hash != canonical_family_hash(record.crossmodal_family_id):
                raise BuildError("invalid_component")
            families.add(record.crossmodal_family_hash)
        receipt.input_sha256[f"inventory_components_{index}"] = artifact.sha256
    quarantine = Quarantine.model_validate(bound_json(manifest.quarantine))
    families.update(quarantine.families)
    media.update(quarantine.media)
    for index, artifact in enumerate(manifest.alias_ledgers):
        for line in bound_lines(artifact):
            prior = PriorComponent.model_validate_json(line)
            if prior.family_hash != canonical_family_hash(prior.family_id):
                raise BuildError("invalid_component")
            if prior.family_id != "librispeech:component:" + digest(
                encode(sorted(set(prior.aliases)))
            ):
                raise BuildError("invalid_component")
            prior_families.discard(prior.family_hash)
            aliases.update(prior.aliases)
            families.add(prior.family_hash)
        receipt.input_sha256[f"alias_ledger_{index}"] = artifact.sha256
    if prior_families:
        raise BuildError("missing_alias_history")
    return families, media, aliases


def members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    result = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in member.name
            or not path.parts
            or member.name.rstrip("/") != str(path)
            or not (member.isdir() or member.isfile())
        ):
            raise BuildError("unsafe_archive")
        name = str(path)
        if name in result:
            raise BuildError("duplicate_archive_member")
        result[name] = member
    return result


def read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    stream = archive.extractfile(member)
    if stream is None:
        raise BuildError("invalid_archive")
    with stream:
        return stream.read()


def flac_eligible(payload: bytes) -> bool:
    # STREAMINFO is mandatory first metadata block; rate/channels/total samples
    # are packed into its last ten bytes before the MD5.
    if len(payload) < 42 or payload[:4] != b"fLaC" or payload[4] & 127 != 0:
        raise BuildError("invalid_audio")
    if int.from_bytes(payload[5:8], "big") != 34:
        raise BuildError("invalid_audio")
    packed = int.from_bytes(payload[18:26], "big")
    rate = packed >> 44
    channels = ((packed >> 41) & 7) + 1
    samples = packed & ((1 << 36) - 1)
    return rate == 16000 and channels == 1 and 0 < samples <= 480000


def source_rows(path: Path, split: str, receipt: Receipt | None = None) -> list[Utterance]:
    result = []
    with tarfile.open(path, "r:gz") as archive:
        index = members(archive)
        chapter_member = index.get("LibriSpeech/CHAPTERS.TXT")
        if chapter_member is None:
            raise BuildError("missing_metadata")
        if receipt is not None:
            metadata_hashes = {
                name: digest(read_member(archive, member))
                for name, member in index.items()
                if member.isfile() and name.endswith((".TXT", ".trans.txt"))
            }
            receipt.input_sha256[split.replace("-", "_") + "_metadata"] = digest(
                encode(metadata_hashes)
            )
        chapters: dict[str, tuple[str, str, str]] = {}
        for line in read_member(archive, chapter_member).decode("utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith(";"):
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 7 or not all(parts[i].isdigit() for i in (0, 1, 5)):
                raise BuildError("invalid_chapters")
            if parts[0] in chapters:
                raise BuildError("duplicate_chapter")
            chapters[parts[0]] = (parts[1], parts[3], parts[5])
        texts: dict[str, str] = {}
        for name, member in index.items():
            if not name.endswith(".trans.txt"):
                continue
            for line in read_member(archive, member).decode("utf-8").splitlines():
                match = re.fullmatch(r"(\d+-\d+-\d+)\s+(.+)", line)
                if match is None:
                    raise BuildError("invalid_transcript")
                identity, text = match.groups()
                speaker, chapter, _ = identity.split("-")
                expected = f"LibriSpeech/{split}/{speaker}/{chapter}/{speaker}-{chapter}.trans.txt"
                if name != expected or identity in texts or not text.strip():
                    raise BuildError("invalid_transcript_identity")
                texts[identity] = text.strip()
        seen = set()
        for name, member in index.items():
            if not name.endswith(".flac"):
                continue
            identity = PurePosixPath(name).stem
            if re.fullmatch(r"\d+-\d+-\d+", identity) is None or identity in seen:
                raise BuildError("invalid_audio_identity")
            seen.add(identity)
            speaker, chapter, _ = identity.split("-")
            if name != f"LibriSpeech/{split}/{speaker}/{chapter}/{identity}.flac":
                raise BuildError("invalid_audio_identity")
            if chapter not in chapters or chapters[chapter][:2] != (speaker, split):
                raise BuildError("invalid_chapter_identity")
            if identity not in texts:
                raise BuildError("missing_transcript")
            payload = read_member(archive, member)
            sha = digest(payload)
            text = texts[identity]
            normalized = digest(" ".join(text.casefold().split()).encode())
            result.append(
                Utterance(
                    identity,
                    split,
                    name,
                    text,
                    digest(text.encode()),
                    sha,
                    {
                        f"librispeech:speaker:{speaker}",
                        f"librispeech:book:{chapters[chapter][2]}",
                        f"librispeech:transcript:{normalized}",
                        f"librispeech:media:{sha}",
                    },
                    flac_eligible(payload),
                )
            )
        if set(texts) != seen or not result:
            raise BuildError("unpaired_transcripts")
    return result


def components(rows: list[Utterance]) -> list[tuple[str, set[str], list[Utterance]]]:
    # Union BEFORE exclusions, duration filtering or deduplication: even an
    # unusable utterance can bridge a speaker, a book and a repeated transcript.
    parent = list(range(len(rows)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    aliases: dict[str, int] = {}
    identities: set[str] = set()
    for i, row in enumerate(rows):
        if row.identity in identities:
            raise BuildError("duplicate_utterance")
        identities.add(row.identity)
        for alias in row.aliases:
            if alias in aliases:
                parent[root(i)] = root(aliases[alias])
            else:
                aliases[alias] = i
    groups: dict[int, list[Utterance]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    result = []
    for group in groups.values():
        original_aliases = set().union(*(r.aliases for r in group))
        family = "librispeech:component:" + digest(encode(sorted(original_aliases)))
        result.append(
            (family, original_aliases, sorted(group, key=lambda r: rank("item", r.identity)))
        )
    return sorted(result, key=lambda group: rank("component", group[0]))


def allocate(capacities: list[int], quotas: list[int]) -> dict[int, GoldPool]:
    """Minimal component capacity per pool, reserving every unused member."""
    remaining = list(range(len(capacities)))
    assignment = {}
    for position, (pool, quota) in enumerate(zip(POOLS, quotas, strict=True)):
        if sum(capacities[i] for i in remaining) < sum(quotas[position:]):
            raise BuildError("insufficient_capacity")
        limit = quota + max((capacities[i] for i in remaining), default=0)
        reachable: dict[int, tuple[int, ...]] = {0: ()}
        for i in remaining:
            for total, picked in list(reachable.items()):
                new = total + capacities[i]
                if new <= limit and new not in reachable:
                    reachable[new] = (*picked, i)
        available = sum(capacities[i] for i in remaining)
        feasible = [n for n in reachable if quota <= n <= available - sum(quotas[position + 1 :])]
        if not feasible:
            raise BuildError("insufficient_component_capacity")
        chosen = reachable[min(feasible)]
        for i in chosen:
            assignment[i] = pool
        remaining = [i for i in remaining if i not in chosen]
    return assignment


def decode(payload: bytes) -> bytes:
    process = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            "pipe:1",
        ],
        input=payload,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if process.returncode:
        raise BuildError("decode_failed")
    return process.stdout


def validate_wav(payload: bytes) -> None:
    with wave.open(io.BytesIO(payload), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise BuildError("invalid_decoded_audio")
        # ffmpeg pipe WAV has an unknown-size header; validate actual decoded data.
        frames = audio.readframes(480001)
        if not frames or len(frames) % 2 or len(frames) > 960000:
            raise BuildError("invalid_decoded_duration")


def build(args: argparse.Namespace, receipt: Receipt) -> None:
    content = args.manifest.read_bytes()
    receipt.input_sha256["build_manifest"] = digest(content)
    manifest = BuildManifest.model_validate_json(content)
    real = not args.canary
    quotas = [args.calibration_count, args.holdout_count, args.dev_count]
    if (
        any(n < 4 for n in quotas)
        or (real and quotas != [1600, 500, 500])
        or (not real and any(n > 100 for n in quotas))
    ):
        raise BuildError("invalid_quota")
    for name in (*PINS, "inventory", "quarantine"):
        artifact = getattr(manifest, name)
        if real and name in PINS and artifact.sha256 != PINS[name]:
            raise BuildError("source_not_pinned")
        verify(artifact)
        receipt.input_sha256[name] = artifact.sha256
    receipt.real_source_verified = real
    families, media, prior_aliases = exclusions(manifest, receipt)
    rows = source_rows(Path(manifest.dev_clean.path), "dev-clean", receipt) + source_rows(
        Path(manifest.test_clean.path), "test-clean", receipt
    )
    groups = components(rows)
    counts: Counter[str] = Counter(source_utterances=len(rows), source_components=len(groups))
    eligible_groups = []
    for family, aliases, group in groups:
        if (
            canonical_family_hash(family) in families
            or aliases & prior_aliases
            or any(canonical_family_hash(alias) in families for alias in aliases)
            or any(row.media_sha256 in media for row in group)
        ):
            counts["excluded_components"] += 1
            continue
        unique: list[Utterance] = []
        used_texts: set[str] = set()
        used_media: set[str] = set()
        for row in group:
            normalized = " ".join(row.transcript.casefold().split())
            if not row.eligible:
                counts["ineligible_audio"] += 1
            elif normalized in used_texts or row.media_sha256 in used_media:
                counts["duplicate_utterances"] += 1
            else:
                unique.append(row)
                used_texts.add(normalized)
                used_media.add(row.media_sha256)
        if unique:
            eligible_groups.append((family, aliases, group, unique))
    receipt.counts = dict(counts)
    receipt.counts["eligible_distinct_utterances"] = sum(len(g[3]) for g in eligible_groups)
    assignment = allocate([len(g[3]) for g in eligible_groups], quotas)
    args.output_dir.mkdir(mode=0o700, exist_ok=False)
    media_dir = args.output_dir / "media"
    media_dir.mkdir(mode=0o700)
    selected: dict[GoldPool, list[tuple[Utterance, str]]] = {p: [] for p in POOLS}
    for i, pool in assignment.items():
        family, _, _, eligible = eligible_groups[i]
        selected[pool].extend((row, family) for row in eligible)
    for pool, quota in zip(POOLS, quotas, strict=True):
        selected[pool] = sorted(selected[pool], key=lambda item: rank("select", item[0].identity))[
            :quota
        ]
    selected_ids = {row.identity for values in selected.values() for row, _ in values}
    # Read only selected audio a second time, never extract arbitrary archive paths.
    payloads: dict[str, bytes] = {}
    for name, split in (("dev_clean", "dev-clean"), ("test_clean", "test-clean")):
        with tarfile.open(getattr(manifest, name).path, "r:gz") as archive:
            index = members(archive)
            for row in rows:
                if row.archive == split and row.identity in selected_ids:
                    payload = read_member(archive, index[row.member])
                    if digest(payload) != row.media_sha256:
                        raise BuildError("media_changed")
                    payloads[row.identity] = payload
    derived: dict[str, str] = {}
    artifacts = []
    for pool, values in selected.items():
        samples = []
        for row, family in values:
            distractors = sorted(
                (r for r, _ in values if r.identity != row.identity),
                key=lambda r: rank("distractor:" + row.identity, r.identity),
            )[:3]
            options = sorted(
                [row, *distractors], key=lambda r: rank("option:" + row.identity, r.identity)
            )
            wav = decode(payloads.pop(row.identity))
            validate_wav(wav)
            sha = digest(wav)
            if sha in media or sha in derived.values():
                raise BuildError("derived_media_overlap")
            derived[row.identity] = sha
            path = media_dir / (row.identity + ".wav")
            write_private(path, wav)
            sample = Sample.model_validate(
                {
                    "id": "librispeech-" + row.identity,
                    "prefix": {
                        "state": "Listen to the attached audio.",
                        "media": [{"type": "audio", "uri": path.as_uri(), "sha256": sha}],
                    },
                    "questions": [
                        {
                            "qid": "transcript",
                            "type": "choice",
                            "instruction": "Which option matches the words spoken in the audio?",
                            "options": [r.transcript for r in options],
                            "target": {
                                "kind": "hard",
                                "value": chr(65 + options.index(row)),
                                "source": "librispeech_original_transcript",
                            },
                        }
                    ],
                    "meta": {
                        "modality": ["audio"],
                        "source_dataset": "LibriSpeech",
                        "source_revision": ("official-sha256:" if real else "synthetic-canary:")
                        + getattr(manifest, row.archive.replace("-", "_")).sha256,
                        "source_sha256": digest(
                            encode(
                                {
                                    "transcript": row.transcript_sha256,
                                    "original_audio": row.media_sha256,
                                }
                            )
                        ),
                        "license": "CC-BY-4.0",
                        "usage": "commercial" if real else "research",
                        "pool": pool,
                        "family_id": family,
                        "data_version": "librispeech-speech-gold-v1"
                        if real
                        else "librispeech-canary-not-real-gold-v1",
                    },
                }
            )
            samples.append(sample)
        path = args.output_dir / f"{pool}.jsonl"
        write_private(path, b"".join((s.model_dump_json() + "\n").encode() for s in samples))
        sha = file_digest(path)
        receipt.output_sha256[pool] = sha
        receipt.counts[pool] = len(samples)
        artifacts.append(
            SampleArtifact.model_validate(
                {
                    "path": str(path),
                    "sha256": sha,
                    "cohort": "audio",
                    "exposure": "current",
                    "expected_pool": pool,
                }
            )
        )
    ledger = []
    for i, pool in assignment.items():
        family, aliases, group, _ = eligible_groups[i]
        ledger.append(
            PriorComponent(
                family_id=family,
                family_hash=canonical_family_hash(family),
                aliases=sorted(aliases),
                pool=pool,
                members=[
                    {
                        "identity": r.identity,
                        "archive": r.archive,
                        "member": r.member,
                        "original_media_sha256": r.media_sha256,
                        "derived_media_sha256": derived.get(r.identity),
                        "source_transcript_sha256": r.transcript_sha256,
                        "selected": r.identity in selected_ids,
                        "attribution": (
                            "LibriSpeech — Vassil Panayotov, Guoguo Chen, "
                            "Daniel Povey, Sanjeev Khudanpur; https://www.openslr.org/12/; "
                            "CC-BY-4.0; FLAC converted to WAV, transcript options derived"
                        ),
                    }
                    for r in group
                ],
            )
        )
    write_private(
        args.output_dir / "allocation.jsonl",
        b"".join((record.model_dump_json() + "\n").encode() for record in ledger),
    )
    output_manifest = Manifest(
        schema_version="gold-inventory-manifest-v1", samples=artifacts, component_ledgers=[]
    )
    write_private(
        args.output_dir / "inventory-manifest.json", output_manifest.model_dump_json().encode()
    )
    for name in ("allocation.jsonl", "inventory-manifest.json"):
        receipt.output_sha256[name] = file_digest(args.output_dir / name)
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
        for pool, count in zip(POOLS, (1600, 500, 500), strict=True):
            parser.add_argument(f"--{pool}-count", type=int, default=count)
        args = parser.parse_args(argv)
        receipt.mode = "canary" if args.canary else "real"
        if any(not p.is_absolute() for p in (args.manifest, args.output_dir, args.report)):
            raise BuildError("invalid_arguments")
        if any(p.exists() or p.is_symlink() for p in (args.output_dir, args.report)):
            raise BuildError("output_exists")
        build(args, receipt)
    except BuildError as error:
        receipt.category = str(error)
    except Rejected as error:
        receipt.category = error.category
    except tarfile.TarError:
        receipt.category = "invalid_archive"
    except (ValueError, KeyError, TypeError, UnicodeError, wave.Error):
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
