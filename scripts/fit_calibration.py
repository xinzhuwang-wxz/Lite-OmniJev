"""Explicit CPU calibration fit/strict-load CLI; never discovers private datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from train.calibration import (
    Binding,
    CalibrationInput,
    fit_temperature,
    load_temperature,
    save_temperature,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--expected-binding", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "fit":
            payload = args.input.read_bytes()
            data = CalibrationInput.model_validate(json.loads(payload))
            artifact = fit_temperature(data, hashlib.sha256(payload).hexdigest())
            save_temperature(artifact, args.output)
        else:
            expected = Binding.model_validate(json.loads(args.expected_binding.read_text()))
            artifact = load_temperature(args.artifact, expected)
        print(
            json.dumps(
                {
                    "status": "fitted" if args.action == "fit" else "binding_verified",
                    "temperature": artifact.temperature,
                    "nll_before": artifact.nll_before,
                    "nll_after": artifact.nll_after,
                    "stage_acceptance": "not_evaluated",
                }
            )
        )
        return 0
    except (ValueError, OSError) as exc:
        # Validation details can contain private values; print the exception class only.
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
