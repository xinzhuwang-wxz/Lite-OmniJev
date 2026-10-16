"""Build the gateway registry manifest from a completed calibration fit report.

Metadata-only: reads the fit report's exported artifact list (path/sha/binding)
and the runtime recipe, and emits a schema-2 manifest with the explicit
fitted/unfitted partition. Never opens temperature artifact bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FITTED_GROUPS = [
    ("text", "choice"),
    ("image", "choice"),
    ("screenshot", "choice"),
    ("audio", "choice"),
    ("audio", "noul"),
    ("video", "noul"),
    ("crossmodal", "noul"),
]
ALL_COHORTS = ("text", "image", "screenshot", "audio", "video", "crossmodal")
ALL_PRIMITIVES = ("choice", "score", "noul")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-version", required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.fit_report.read_text())
    require(report["status"] == "fitted" and report["exit_code"] == 0, "not_fitted")
    artifacts = report["artifacts"]
    require(len(artifacts) == 7, "expected_seven_artifacts")
    seen = set()
    groups = []
    fitted = []
    for entry in artifacts:
        binding = entry["binding"]
        key = (binding["cohort"], binding["primitive"])
        require(key not in seen, "duplicate_group")
        seen.add(key)
        require(key in FITTED_GROUPS, "unexpected_fitted_group")
        relative = Path(entry["path"])
        require(not relative.is_absolute(), "artifact_path_must_be_registry_relative")
        resolved = (args.registry_root / relative).resolve()
        require(resolved.is_relative_to(args.registry_root.resolve()), "artifact_escapes_root")
        require(
            resolved.is_file() or entry.get("sha256"), "artifact_missing_metadata"
        )  # existence checked again by the registry at load time
        groups.append({"path": str(relative), "sha256": entry["sha256"], "binding": binding})
        fitted.append({"cohort": key[0], "primitive": key[1]})
    require(seen == set(FITTED_GROUPS), "fitted_groups_mismatch")
    adapter_sha = artifacts[0]["binding"]["adapter_weights_sha256"]
    unfitted = [
        {"cohort": c, "primitive": p}
        for c in ALL_COHORTS
        for p in ALL_PRIMITIVES
        if (c, p) not in seen
    ]
    require(len(unfitted) == 11, "unfitted_partition_size")
    recipe = report.get("runtime_recipe")
    manifest = {
        "schema_version": 2,
        "model_version": adapter_sha[:16],
        "calibration_version": args.calibration_version,
        "groups": groups,
        "fitted": fitted,
        "unfitted": unfitted,
        "runtime_provenance": [
            {"path": str(args.fit_report), "sha256": report["report_sha256"]}
        ]
        if report.get("report_sha256")
        else [],
    }
    if recipe is not None:
        manifest["runtime_recipe"] = recipe
    args.output.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {args.output}: {len(groups)} fitted, {len(unfitted)} unfitted")
    return 0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"registry manifest rejected: {message}")


if __name__ == "__main__":
    raise SystemExit(main())
