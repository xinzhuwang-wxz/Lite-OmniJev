"""Bounded metadata-only rejection checks; accepts same CLI inputs as proposal."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "proposal", Path(__file__).with_name("decision-index-additional-admission.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ap = argparse.ArgumentParser()
for name in ["observed", "publishers", "source", "clinc-receipt"]:
    ap.add_argument("--" + name, required=True, type=Path)
a = ap.parse_args()
observed = json.loads(a.observed.read_bytes())
publishers = json.loads(a.publishers.read_bytes())


def run(o, p):
    return module.build(o, p, a.source, a.clinc_receipt)


result = run(observed, publishers)
assert [x["catalog_id"] for x in result["admissions"]] == [4, 5, 9]
assert result["applied"] is False and result["remaining_of44"] == 28
cases = []
o = copy.deepcopy(observed)
o["manifest_sha256"] = "0" * 64
cases.append((o, publishers, "approved_base_identity"))
o = copy.deepcopy(observed)
o["manifest"]["admissions"][0]["evidence"] = "changed"
cases.append((o, publishers, "base_manifest_bytes"))
o = copy.deepcopy(observed)
o["observed_files"][0]["sha256"] = "0" * 64
cases.append((o, publishers, "data_sha_identity"))
o = copy.deepcopy(observed)
o["observed_files"][0]["git_blob_sha1"] = "0" * 40
cases.append((o, publishers, "publisher_blob_identity"))
p = [
    x
    for x in publishers
    if x.get("sha256") != "7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661"
]
cases.append((observed, p, "publisher_license_identity"))
for o, p, expected in cases:
    try:
        run(o, p)
    except ValueError as e:
        assert str(e) == expected, (expected, str(e))
    else:
        raise AssertionError("accepted invalid binding:" + expected)
print(json.dumps({"positive": 1, "rejection_checks": len(cases), "status": "passed"}))
