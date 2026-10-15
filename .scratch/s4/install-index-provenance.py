"""Install the hash-reviewed provenance patch in a new derived source copy."""

import ast
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
bundle = root / "artifacts/s4-index-provenance-bundle-001"
original = root / "artifacts/s4-index-work-001/source"
derived = root / "artifacts/s4-index-derived-source-001"
report_path = root / "reports/decision-index-provenance-install-001.json"
assert not derived.exists() and not report_path.exists()
start = time.monotonic()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


receipt = json.loads((bundle / "receipt.json").read_text())
assert receipt["exit_code"] == 0 and receipt["status"] == "patch_bundle_prepared"
layout = Path("decision_index/suite/build/layout.py")
assert sha(original / layout) == receipt["original_source_sha256"]
assert sha(bundle / "layout.py") == receipt["derived_source_sha256"]
assert sha(bundle / "revision-map.json") == receipt["revision_map_sha256"]
shutil.copytree(original, derived)
shutil.copyfile(bundle / "layout.py", derived / layout)
assert sha(derived / layout) == receipt["derived_source_sha256"]
assert sha(original / layout) == receipt["original_source_sha256"]
# Run only our generated function, never import/execute the upstream module.
functions = [
    node
    for node in ast.parse((derived / layout).read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "git_revision"
]
assert len(functions) == 1
namespace = {}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(derived / layout), "exec"), namespace)
os.environ["DECISION_INDEX_PROVENANCE_MAP"] = str(bundle / "revision-map.json")
mapping = json.loads((bundle / "revision-map.json").read_text())
verified = {}
for name, item in mapping["repositories"].items():
    actual = namespace["git_revision"](name)
    assert actual == item["revision"]
    verified[name] = actual
try:
    namespace["git_revision"](str(root))
except ValueError:
    unknown_rejected = True
else:
    raise AssertionError("unknown parent repo accepted")
report = {
    "status": "installed_and_lookup_verified",
    "exit_code": 0,
    "ready": False,
    "upstream_rebuild_executed": False,
    "source_patch_is_original_upstream": False,
    "original_source": str(original),
    "derived_source": str(derived),
    "bundle_receipt_sha256": sha(bundle / "receipt.json"),
    "original_layout_sha256": sha(original / layout),
    "derived_layout_sha256": sha(derived / layout),
    "revision_map_sha256": sha(bundle / "revision-map.json"),
    "verified_repositories": verified,
    "unknown_parent_rejected": unknown_rejected,
    "script_sha256": sha(Path(__file__)),
    "wall_seconds": time.monotonic() - start,
    "remaining": [
        "evaluation admissions",
        "dependency/network closure",
        "full rebuild and independent verify",
    ],
}
with report_path.open("x") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report))
