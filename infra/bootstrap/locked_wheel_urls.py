#!/usr/bin/env python3
"""Turn a hashed uv export into wheel URLs for CPython 3.11 / Ubuntu 22.04.

Run with a Python environment containing packaging. No network requests are made.
Example:
  python infra/bootstrap/locked_wheel_urls.py --lock uv.lock \
    --requirements requirements-training.lock > requirements-training-urls.lock
Install with: uv pip sync --no-index --require-hashes OUTPUT
Only exported dependencies active on the target are selected, never the whole lock.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def convert(lock: Path, requirements: Path, mirror: str | None = None) -> str:
    """Select hash-matched binary artifacts or fail before emitting any output."""
    environment = default_environment()
    environment.update(
        implementation_name="cpython",
        implementation_version="3.11.14",
        os_name="posix",
        platform_machine="x86_64",
        platform_python_implementation="CPython",
        platform_system="Linux",
        python_full_version="3.11.14",
        python_version="3.11",
        sys_platform="linux",
        extra="",
    )
    platforms = []
    for minor in range(35, 4, -1):
        platforms.append(f"manylinux_2_{minor}_x86_64")
        if minor in {17, 12, 5}:
            platforms.append(
                {17: "manylinux2014", 12: "manylinux2010", 5: "manylinux1"}[minor] + "_x86_64"
            )
    platforms.append("linux_x86_64")
    tags = list(cpython_tags((3, 11), abis=["cp311"], platforms=platforms))
    tags.extend(compatible_tags((3, 11), interpreter="cp311", platforms=platforms))
    priority = {tag: index for index, tag in reversed(list(enumerate(tags)))}
    packages = tomllib.loads(lock.read_text())["package"]
    if mirror:
        parsed = urlsplit(mirror)
        if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("mirror must be an HTTPS base URL without query or fragment")
        if parsed.username or parsed.password:
            raise ValueError("mirror must not contain credentials")
    lines = requirements.read_text().replace("\\\n", " ").splitlines()
    output = ["# CPython 3.11.14 / Linux x86_64 / glibc 2.35; generated from hashed uv export"]
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        hashes = set(re.findall(r"--hash=(sha256:[a-fA-F0-9]{64})(?=\s|$)", line))
        requirement = Requirement(re.sub(r"\s+--hash=\S+", "", line))
        if requirement.marker:
            if any(k in str(requirement.marker) for k in ("platform_release", "platform_version")):
                raise ValueError(f"kernel-dependent marker cannot be resolved: {requirement}")
            if not requirement.marker.evaluate(environment):
                continue
        pins = list(requirement.specifier)
        if requirement.url or len(pins) != 1 or pins[0].operator != "==" or "*" in pins[0].version:
            raise ValueError(f"expected an exact registry version: {requirement}")
        if not hashes:
            raise ValueError(f"missing exported SHA256 hashes: {requirement}")
        name = canonicalize_name(requirement.name)
        version = Version(pins[0].version)
        candidates = []
        for package in packages:
            if canonicalize_name(package["name"]) != name or Version(package["version"]) != version:
                continue
            for wheel in package.get("wheels", []):
                url = wheel["url"]
                wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(
                    unquote(urlsplit(url).path.rsplit("/", 1)[-1])
                )
                ranks = [priority[t] for t in wheel_tags if t in priority]
                if (
                    wheel_name == name
                    and wheel_version == version
                    and ranks
                    and wheel["hash"] in hashes
                ):
                    candidates.append((min(ranks), url, wheel["hash"]))
        if not candidates:
            raise ValueError(f"no compatible hash-matched wheel: {requirement}")
        _, url, digest = min(candidates)
        if mirror and url.startswith("https://files.pythonhosted.org/"):
            url = mirror.rstrip("/") + "/" + url.removeprefix("https://files.pythonhosted.org/")
        extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
        marker = f" ; {requirement.marker}" if requirement.marker else ""
        output.append(f"{name}{extras} @ {url}{marker} --hash={digest}")
    if len(output) == 1:
        raise ValueError("export contains no dependencies active for the target")
    return "\n".join(output) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument(
        "--pypi-mirror-base",
        help="Rewrite files.pythonhosted.org origin only; mirror must preserve /packages/... paths",
    )
    args = parser.parse_args()
    try:
        result = convert(args.lock, args.requirements, args.pypi_mirror_base)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    sys.stdout.write(result)


if __name__ == "__main__":
    main()
