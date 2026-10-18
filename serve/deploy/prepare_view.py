"""Prepare identity-bound local vLLM views without deleting operator directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

METADATA = (
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
MARKER = ".source-binding.json"


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def prepare(model: Path, base: Path, view: Path, config_dir: Path) -> None:
    model, base = model.resolve(), base.resolve()
    if view.resolve() == config_dir.resolve():
        raise ValueError("view and config directories must differ")
    index = model / "model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text())
        names = sorted(set(payload["weight_map"].values()))
        if not names:
            raise ValueError("empty weight map")
    else:
        names = ["model.safetensors"]
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in names):
        raise ValueError("weight index must reference local safetensors basenames")
    weights = {name: model / name for name in names}
    weight_identity = {name: identity(path) for name, path in weights.items()}
    files = {name: base / name for name in METADATA if (base / name).is_file()}
    files["config.json"] = model / "config.json"
    generation = model / "generation_config.json"
    if not generation.exists():
        generation = base / "generation_config.json"
    if generation.exists():
        files["generation_config.json"] = generation
    if index.exists():
        files[index.name] = index
    config = json.loads((base / "config.json").read_text())
    for section in ("", "audio_config", "text_config", "vision_config"):
        node = config if not section else config.get(section)
        if not isinstance(node, dict) or node.get("dtype") not in ("bfloat16", "float32"):
            raise ValueError(f"unsupported dtype section: {section or 'root'}")
        node["dtype"] = "float32"
    source = {
        "model_dir": str(model),
        "base_dir": str(base),
        "weights": weight_identity,
        "metadata_sha256": {name: checksum(path) for name, path in files.items()},
        "base_config_sha256": checksum(base / "config.json"),
    }
    derived = json.dumps(config, indent=2).encode() + b"\n"
    # Validate both existing directories before creating either. Reuse is allowed
    # only for the same immutable source and intact files, never by config presence.
    for target, kind in ((view, "view"), (config_dir, "config")):
        marker = target / MARKER
        if not target.exists():
            continue
        if not marker.is_file():
            raise ValueError(f"unbound directory; select a fresh path: {target}")
        binding = json.loads(marker.read_text())
        if binding.get("source") != source or binding.get("kind") != kind:
            raise ValueError(f"source identity differs; select a fresh path: {target}")
        if kind == "view":
            for name, path in files.items():
                if checksum(target / name) != checksum(path):
                    raise ValueError(f"derived metadata changed: {target / name}")
            for name, expected in weight_identity.items():
                if identity(target / name) != expected:
                    raise ValueError(f"derived weight changed: {target / name}")
        elif (target / "config.json").read_bytes() != derived:
            raise ValueError(f"derived config changed: {target}")
    for target, kind in ((view, "view"), (config_dir, "config")):
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        try:
            if kind == "view":
                for name, path in files.items():
                    shutil.copy2(path, temporary / name)
                for name, path in weights.items():
                    os.link(path, temporary / name)
            else:
                (temporary / "config.json").write_bytes(derived)
            (temporary / MARKER).write_text(
                json.dumps({"source": source, "kind": kind}, indent=2) + "\n"
            )
            if target.exists():
                raise ValueError(f"destination appeared during preparation: {target}")
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)  # Only the temporary directory created above.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "base-dir", "view", "config-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.model_dir, args.base_dir, args.view, args.config_dir)


if __name__ == "__main__":
    main()
