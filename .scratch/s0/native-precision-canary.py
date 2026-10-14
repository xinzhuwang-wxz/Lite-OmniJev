"""Six-case real HTTP precision diagnostic; the formal gate still requires 100 rows."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from scripts.s0_parity import load_cases, load_official_processor, row_from_prepared, rows_to_json
from serve.vllm_chat import VllmChatClient
from train.multimodal import prepare_case

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--asset-root", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--served-model", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--upstream-url", default="http://127.0.0.1:18001")
args = parser.parse_args()
if args.output.exists():
    parser.error("output exists")
started = time.monotonic()
report = {
    "status": "running",
    "exit_code": 1,
    "cases": [],
    "scope": "Six-case diagnostic only; not full100, quality or stage acceptance",
    "release_or_stage_acceptance": False,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
}


def save():
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


async def run():
    reference = json.loads(args.reference.read_text())
    assert reference["status"] == "passed" and reference["exact_repeatability"]
    rows = reference["passes"][0]
    assert len(rows) == 6
    cases = load_cases(Path("tests/fixtures/s0-prepare-public-001/manifest.json"))
    processor = load_official_processor(args.model_dir)
    async with VllmChatClient(
        args.upstream_url, args.served_model, raw_logprobs_confirmed=True
    ) as client:
        for i, index in enumerate((0, 20, 40, 60, 80, 90)):
            case = cases[index]
            prepared = prepare_case(case, processor, args.asset_root, profile="canonical_http")
            outputs = []
            for _ in range(2):
                result = await client.read_logprobs(prepared)
                outputs.append(
                    rows_to_json([row_from_prepared(case, prepared, result.raw_logprobs)])[0]
                )
            for key in ("id_sha256", "input_sha256", "media_sha256", "token_ids"):
                assert rows[i][key] == outputs[0][key]
            differences = {
                key: max(abs(a - b) for a, b in zip(rows[i][key], outputs[0][key], strict=True))
                for key in ("raw_probabilities", "joint_k_plus_u", "conditional_without_u")
            }
            differences["p_U"] = abs(rows[i]["p_U"] - outputs[0]["p_U"])
            report["cases"].append(
                {
                    "id": case.id,
                    "exact_repeatability": outputs[0] == outputs[1],
                    "max_differences": differences,
                    "outputs": outputs,
                }
            )
            save()
            print(case.category, differences, flush=True)


save()
try:
    asyncio.run(run())
    passed = all(
        case["exact_repeatability"] and max(case["max_differences"].values()) < 0.001
        for case in report["cases"]
    )
    report.update(status="passed" if passed else "failed", exit_code=0 if passed else 2)
except Exception as error:
    import traceback

    report.update(
        status="failed",
        exit_code=1,
        error_type=type(error).__name__,
        stack=[
            {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)
        ],
    )
save()
print(json.dumps({k: report[k] for k in ("status", "exit_code", "elapsed_seconds")}))
raise SystemExit(report["exit_code"])
