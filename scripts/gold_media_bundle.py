"""Private media relocation only; not gold, provenance, or permission admission.

Input gold-media-bundle-manifest-v1 lists SampleArtifacts with media_roots
(or one legacy media_root) each.
Output inventory-manifest.json uses output-dir as the sole trusted media root;
private mapping.json records original/new artifact paths and SHA256 values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Literal, Never, Self
from urllib.parse import unquote, urlsplit

from pydantic import Field, model_validator

from data.schema import Contract, Sample
from scripts.gold_inventory import SampleArtifact, digest


class InputArtifact(SampleArtifact):
    media_root: str | None = Field(default=None, min_length=1)
    media_roots: list[str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def root_absolute(self) -> Self:
        if (self.media_root is None) == (self.media_roots is None):
            raise ValueError("exactly one root declaration required")
        if any(not Path(root).is_absolute() for root in self.roots()):
            raise ValueError("absolute roots required")
        return self

    def roots(self) -> list[str]:
        return self.media_roots if self.media_roots is not None else [self.media_root or ""]


class Manifest(Contract):
    schema_version: Literal["gold-media-bundle-manifest-v1"]
    samples: list[InputArtifact] = Field(min_length=1)


class Rejected(Exception):
    """Only fixed categories escape the private process."""


class Receipt(Contract):
    schema_version: Literal["gold-media-bundle-receipt-v1"] = "gold-media-bundle-receipt-v1"
    outcome: Literal["bundled", "rejected"]
    categories: list[str] = Field(default_factory=list)
    manifest_sha256: str | None = None
    artifacts: int = 0
    samples: int = 0
    media_references: int = 0
    unique_media: int = 0
    copied_bytes: int = 0
    output_artifact_sha256: list[str] = Field(default_factory=list)
    mapping_sha256: str | None = None
    inventory_manifest_sha256: str | None = None
    media_root_sha256: str | None = None
    gold_provenance: Literal["not_validated"] = "not_validated"
    private_job_permissions: Literal["not_validated"] = "not_validated"


def real_path(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise Rejected("unsafe_path")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise Rejected("unsafe_path")


def private_write(path: Path, content: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(content)


def media_path(roots: list[Path], uri: str) -> Path:
    for root in roots:
        real_path(root)
        if not root.is_dir():
            raise Rejected("unsafe_path")
    if uri.startswith("file://"):
        parsed = urlsplit(uri)
        if parsed.netloc or "?" in uri or "#" in uri:
            raise Rejected("unsafe_path")
        local = unquote(parsed.path)
        value = Path(local)
        if not value.is_absolute() or not any(value.is_relative_to(root) for root in roots):
            raise Rejected("unsafe_path")
        candidates = [value]
    else:
        # Relative URIs are literal local filenames, as in the trainer resolver.
        # Percent escapes, question marks and hashes have no URL meaning here.
        local = uri
        value = Path(local)
        if value.is_absolute() or ":" in local:
            raise Rejected("unsafe_path")
        candidates = [root / value for root in roots]
    if ".." in value.parts or "\\" in local or "\x00" in local:
        raise Rejected("unsafe_path")
    matches: set[Path] = set()
    for path in candidates:
        real_path(path)
        if path.exists():
            matches.add(path.resolve())
    if not matches:
        raise Rejected("missing_media")
    if len(matches) != 1:
        raise Rejected("ambiguous_media")
    return next(iter(matches))


def copy_media(source: Path, target: Path, expected: str) -> int:
    # Directory descriptors with NOFOLLOW also reject traversal during concurrent renames.
    fd = os.open(source.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in source.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        media_fd = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)
    with os.fdopen(media_fd, "rb") as media:
        if not stat.S_ISREG(os.fstat(media.fileno()).st_mode):
            raise Rejected("invalid_media")
        out_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        total = 0
        hasher = hashlib.sha256()
        with os.fdopen(out_fd, "wb") as output:
            while block := media.read(1024 * 1024):
                hasher.update(block)
                total += len(block)
                output.write(block)
        if not total or hasher.hexdigest() != expected:
            target.unlink()
            raise Rejected("hash_mismatch")
    return total


def bundle(manifest_path: Path, output: Path) -> Receipt:
    receipt = Receipt(outcome="rejected")
    try:
        real_path(manifest_path)
        real_path(output)
        content = manifest_path.read_bytes()
        receipt.manifest_sha256 = digest(content)
        try:
            manifest = Manifest.model_validate_json(content)
        except ValueError:
            raise Rejected("invalid_manifest") from None
        output.mkdir(mode=0o700)
        (output / "media").mkdir(mode=0o700)
        mapping = []
        rewritten = []
        seen: dict[str, tuple[str, str]] = {}
        for index, artifact in enumerate(manifest.samples):
            source = Path(artifact.path)
            real_path(source)
            with source.open("rb") as bound:
                hasher = hashlib.sha256()
                while block := bound.read(1024 * 1024):
                    hasher.update(block)
                if hasher.hexdigest() != artifact.sha256:
                    raise Rejected("hash_mismatch")
            target = output / f"samples-{index:04d}.jsonl"
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            input_hash = hashlib.sha256()
            with source.open("rb") as inputs, os.fdopen(fd, "w") as destination:
                for line in inputs:
                    input_hash.update(line)
                    try:
                        Sample.model_validate_json(line)
                        row: dict[str, Any] = json.loads(line)
                    except ValueError:
                        raise Rejected("invalid_sample") from None
                    for media in row["prefix"].get("media", []):
                        path = media_path([Path(root) for root in artifact.roots()], media["uri"])
                        sha, kind = media["sha256"], media["type"]
                        if sha in seen and seen[sha][0] != kind:
                            raise Rejected("type_conflict")
                        # Preserve recognized container suffixes; otherwise use the media type.
                        suffix = path.suffix.lower()
                        if suffix not in {
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".webp",
                            ".wav",
                            ".flac",
                            ".mp3",
                            ".ogg",
                            ".mp4",
                            ".webm",
                            ".mov",
                            ".avi",
                        }:
                            suffix = "." + kind
                        relative = seen.get(sha, (kind, f"media/{sha}{suffix}"))[1]
                        if sha in seen:
                            # Verify every declared reference, even when bytes were deduplicated.
                            temporary = output / "media" / f"verify-{receipt.media_references}"
                            copy_media(path, temporary, sha)
                            temporary.unlink()
                        else:
                            receipt.copied_bytes += copy_media(path, output / relative, sha)
                            seen[sha] = (kind, relative)
                        media["uri"] = relative
                        receipt.media_references += 1
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                    receipt.samples += 1
            if input_hash.hexdigest() != artifact.sha256:
                raise Rejected("hash_mismatch")
            with target.open("rb") as generated:
                generated_hash = hashlib.sha256()
                while block := generated.read(1024 * 1024):
                    generated_hash.update(block)
            new_hash = generated_hash.hexdigest()
            rewritten.append(
                {
                    **artifact.model_dump(exclude={"media_root", "media_roots"}),
                    "path": str(target),
                    "sha256": new_hash,
                }
            )
            mapping.append(
                {
                    "original": {"path": artifact.path, "sha256": artifact.sha256},
                    "rewritten": {"path": str(target), "sha256": new_hash},
                }
            )
            receipt.output_artifact_sha256.append(new_hash)
            receipt.artifacts += 1
        private_write(output / "mapping.json", json.dumps(mapping) + "\n")
        private_write(
            output / "inventory-manifest.json",
            json.dumps(
                {
                    "schema_version": "gold-inventory-manifest-v1",
                    "samples": rewritten,
                    "component_ledgers": [],
                }
            )
            + "\n",
        )
        private_write(output / "media-root.json", json.dumps({"media_root": str(output)}) + "\n")
        receipt.mapping_sha256 = digest((output / "mapping.json").read_bytes())
        receipt.inventory_manifest_sha256 = digest(
            (output / "inventory-manifest.json").read_bytes()
        )
        receipt.media_root_sha256 = digest((output / "media-root.json").read_bytes())
        receipt.unique_media = len(seen)
        receipt.outcome = "bundled"
    except Rejected as error:
        receipt.categories = [str(error)]
    except FileExistsError:
        receipt.categories = ["output_exists"]
    except OSError:
        receipt.categories = ["io_failure"]
    except Exception:
        receipt.categories = ["job_failed"]
    return receipt


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        raise Rejected("invalid_arguments")


def main(argv: list[str] | None = None) -> int:
    try:
        parser = Parser(description=__doc__)
        for argument in ("manifest", "output-dir", "report"):
            parser.add_argument("--" + argument, required=True, type=Path)
        args = parser.parse_args(argv)
        if any(not path.is_absolute() for path in (args.manifest, args.output_dir, args.report)):
            raise Rejected("invalid_arguments")
        real_path(args.report)
        if args.report.exists():
            raise Rejected("output_exists")
        receipt = bundle(args.manifest, args.output_dir)
        private_write(args.report, receipt.model_dump_json() + "\n")
    except Rejected as error:
        receipt = Receipt(outcome="rejected", categories=[str(error)])
    except Exception:
        receipt = Receipt(outcome="rejected", categories=["report_io"])
    print(receipt.model_dump_json())
    return 0 if receipt.outcome == "bundled" else 1


if __name__ == "__main__":
    raise SystemExit(main())
