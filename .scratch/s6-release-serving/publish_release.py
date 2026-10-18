"""Publish an explicit release allowlist using existing operator credentials.

Never prints credentials or uploads scratch scripts, logs, private pools, or raw data.
Run on the existing release box. Upload commits may be partial until final verification.
"""

import hashlib
import json
import os
import re
import shlex
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path("/root/autodl-tmp/s6-hf-repo")
BASE = Path(
    "/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
QUANT = Path("/root/autodl-tmp/s6-quant")
REPORT = Path("/root/autodl-tmp/s6-release-serving-001/artifacts/hf-publish-final-001.json")


def link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_ino != destination.stat().st_ino:
            raise ValueError(f"destination already exists: {destination.name}")
        return
    os.link(source, destination)


def main():
    started = time.time()
    # Reuse the already-authorized token from the prior operator startup file.
    match = re.search(r"^export HF_TOKEN=(.+)$", (ROOT / "hfup.sh").read_text(), re.M)
    if match is None:
        raise ValueError("existing operator token unavailable")
    token = shlex.split(match.group(1))[0]
    api = HfApi(token=token)
    api.whoami()  # Fail before scheduling any upload if revoked.
    metadata = [
        "chat_template.jinja",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
    ]
    for name in metadata:
        if (BASE / name).is_file():
            shutil.copy2(BASE / name, ROOT / name)
    link(
        QUANT / "gguf/mmproj-lite-omnijev-e4b-d3p600-f16.gguf",
        ROOT / "gguf/mmproj-lite-omnijev-e4b-d3p600-f16.gguf",
    )
    mlx = QUANT / "mlx/lite-omnijev-e4b-d3p600-4bit"
    for path in mlx.iterdir():
        if path.is_file() and path.suffix in (".json", ".safetensors", ".jinja"):
            link(path, ROOT / "mlx-4bit" / path.name)
    allowed = [
        "README.md",
        "LICENSE",
        "TRAINING.md",
        "DATA_CARD.md",
        "RELEASE_STATUS.md",
        "SHA256SUMS",
        "config.json",
        "generation_config.json",
        "MERGED_FROM.json",
        "model.safetensors.index.json",
        "model-flat-*.safetensors",
        *metadata,
        "adapter-d3p600/*.json",
        "adapter-d3p600/*.safetensors",
        "gguf/*.gguf",
        "mlx-4bit/*.json",
        "mlx-4bit/*.safetensors",
        "mlx-4bit/*.jinja",
        "gateway/*.sh",
        "gateway/*.py",
        "gateway/*.md",
        "gateway/receipts/*.json",
        "gateway/receipts/*.jsonl",
        "gateway/receipts/*.txt",
        "gateway/receipts/*.png",
        "docs/charts/*.svg",
        "docs/demo/*.gif",
        "docs/demo/*.webm",
    ]
    # One complete manifest is created before network writes. No private artifacts.
    from fnmatch import fnmatch

    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(ROOT).parts)
        and any(fnmatch(str(p.relative_to(ROOT)), pattern) for pattern in allowed)
    ]
    checksums = []
    old_manifest = ROOT / "SHA256SUMS"
    cached = {}
    if old_manifest.exists():
        cached = dict(line.split("  ", 1)[::-1] for line in old_manifest.read_text().splitlines())
    for path in sorted(files):
        if path.name == "SHA256SUMS":
            continue
        name = str(path.relative_to(ROOT))
        if (
            path.stat().st_size > 100_000_000
            and name in cached
            and path.stat().st_mtime <= old_manifest.stat().st_mtime
        ):
            digest = cached[name]
        else:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        checksums.append(f"{digest}  {path.relative_to(ROOT)}")
    (ROOT / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(
        f"PUBLISH_START files={len(files)} bytes={sum(p.stat().st_size for p in files)}", flush=True
    )
    # Use the current resumable large-folder uploader rather than git LFS copies.
    api.upload_folder(
        repo_id="Chinzhu/Lite-OmniJev",
        repo_type="model",
        folder_path=ROOT,
        allow_patterns=allowed,
    )
    info = api.repo_info("Chinzhu/Lite-OmniJev", repo_type="model", files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings}
    errors = []
    for path in files:
        name = str(path.relative_to(ROOT))
        item = remote.get(name)
        if item is None or item.size != path.stat().st_size:
            errors.append(name)
    receipt = {
        "status": "published" if not errors else "verification_failed",
        "exit_code": int(bool(errors)),
        "revision": info.sha,
        "expected_files": len(files),
        "remote_files": len(remote),
        "missing_or_wrong_size": errors,
        "elapsed_seconds": time.time() - started,
        "repository": "Chinzhu/Lite-OmniJev",
    }
    REPORT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
