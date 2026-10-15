"""Bind historical six-cohort development and crossmodal parent ledgers to audit."""

import hashlib
import json
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
out = root / "artifacts/s4-gold-inventory-001"
manifest = json.loads((out / "training-reserve-manifest.json").read_text())
seen = set()
for name in ["s0-development-bundle-001.json", "s0-development-bundle-v2-001.json"]:
    bundle = json.loads((root / "reports" / name).read_text())
    assert bundle["status"] == "passed"
    for entry in bundle["sources"]:
        key = (entry["name"], entry["dev_sha256"])
        if key in seen:
            continue
        seen.add(key)
        path = Path(entry["data"])
        report = Path(entry["report"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["dev_sha256"]
        assert hashlib.sha256(report.read_bytes()).hexdigest() == entry["report_sha256"]
        manifest["samples"].append(
            {
                "path": str(path),
                "sha256": entry["dev_sha256"],
                "cohort": entry["name"],
                "exposure": "historical",
                "expected_pool": "dev",
            }
        )
        if entry["name"] == "crossmodal":
            raw = json.loads(report.read_text())
            command = raw["command"]
            component = Path(command[command.index("--component-ledger-out") + 1])
            assert component.is_absolute()
            sha = hashlib.sha256(component.read_bytes()).hexdigest()
            assert sha == raw["outputs"]["component_ledger_sha256"]
            manifest["component_ledgers"].append({"path": str(component), "sha256": sha})
p = out / "with-development-manifest.json"
assert not p.exists()
p.write_text(json.dumps(manifest, indent=2) + "\n")
print(
    json.dumps(
        {
            "status": "prepared",
            "sample_artifacts": len(manifest["samples"]),
            "development_sources": len(seen),
            "component_ledgers": len(manifest["component_ledgers"]),
            "manifest_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
    )
)
