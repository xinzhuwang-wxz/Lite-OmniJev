"""Deterministic rebuild verification: compare rebuilt outputs to pinned hashes.

Any mismatch means the data source changed (contamination or drift) and the
provision chain must stop with evidence. CPU only; self-tested with fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_expected(manifest_path: Path) -> dict[str, str]:
    """Expected {relative_path: sha256} from the committed expected-hash file."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "files" in payload:
        return {entry["path"]: entry["sha256"] for entry in payload["files"]}
    return {k: v for k, v in payload.items() if isinstance(v, str)}


def verify(root: Path, expected: dict[str, str]) -> dict:
    report = {"matched": 0, "missing": [], "mismatched": [], "extra": [], "ok": True}
    for relative, expected_sha in sorted(expected.items()):
        path = root / relative
        if not path.is_file():
            report["missing"].append(relative)
            report["ok"] = False
            continue
        actual = sha256_file(path)
        if actual != expected_sha:
            report["mismatched"].append({"path": relative, "expected": expected_sha, "actual": actual})
            report["ok"] = False
        else:
            report["matched"] += 1
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    expected = load_expected(args.expected)
    report = verify(args.root, expected)
    report["expected_files"] = len(expected)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


def _self_test() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a").write_text("hello")
        (root / "sub").mkdir()
        (root / "sub/b").write_text("world")
        expected = {
            "a": hashlib.sha256(b"hello").hexdigest(),
            "sub/b": hashlib.sha256(b"world").hexdigest(),
        }
        good = verify(root, expected)
        assert good["ok"] and good["matched"] == 2
        expected["sub/b"] = "0" * 64
        expected["gone"] = "1" * 64
        bad = verify(root, expected)
        assert not bad["ok"] and len(bad["mismatched"]) == 1 and bad["missing"] == ["gone"]
    print(json.dumps({"self_test": "passed"}))
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
