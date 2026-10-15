"""Read-only same-input arm comparison; descriptive evidence, not stage acceptance."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.s0_development_baselines import answer_tokens_for_record, score_records

ROOT = Path("/root/autodl-tmp/s1-text-lora-001")
parser = argparse.ArgumentParser()
parser.add_argument("--source", choices=("original", "supplemental"), default="original")
args = parser.parse_args()
count = 100 if args.source == "original" else 50
paths = {
    "coverage": ROOT
    / (
        "artifacts/s4-cross-batch100-original-coverage-001"
        if args.source == "original"
        else "artifacts/s4-cross-batch50-supplemental-coverage-001"
    ),
    **{
        arm: ROOT / f"artifacts/s4-claim-pair-{arm}-{args.source}{count}-001"
        for arm in ("ce0", "pair01")
    },
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prediction(r):
    labels = answer_tokens_for_record(r)
    return max(range(len(labels)), key=lambda i: r["conditional_without_u"][labels[i]])


reports = {k: json.loads((p / "receipt.json").read_text()) for k, p in paths.items()}
rows = {}
for key, path in paths.items():
    report = reports[key]
    assert report["status"] == "completed" and report["exit_code"] == 0
    assert sha(path / "records.jsonl") == report["records_sha256"]
    rows[key] = [json.loads(x) for x in (path / "records.jsonl").read_text().splitlines()]
    assert len(rows[key]) == count and len({r["id"] for r in rows[key]}) == count
for a, b, c in zip(rows["coverage"], rows["ce0"], rows["pair01"], strict=True):
    for field in ["id", "input_sha256", "input_summary_sha256", "answer_token_ids", "target_index"]:
        assert a[field] == b[field] == c[field]
train = {
    k: json.loads((ROOT / f"reports/s4-claim-pair-{k}-training-001.json").read_text())
    for k in ("ce0", "pair01")
}
assert all(t["status"] == "completed" and t["steps_completed"] == 10 for t in train.values())
assert train["ce0"]["init_adapter"] == train["pair01"]["init_adapter"]
assert train["ce0"]["data_sha256"] == train["pair01"]["data_sha256"]
configs = {
    k: {
        n: v
        for n, v in t["config"].items()
        if n
        not in {
            "pair_loss_weight",
            "pair_sidecar",
            "pair_sidecar_sha256",
            "pair_ledger",
            "pair_ledger_sha256",
        }
    }
    for k, t in train.items()
}
assert configs["ce0"] == configs["pair01"]
result = {
    "status": "compared",
    "scope": f"fixed {args.source} dev{count}; no stage gate or significance claim",
    "same_inputs_verified": count,
    "same_initialization_data_config_except_pair": True,
    "models": {},
    "paired": {},
    "bindings": {
        k: {"receipt_sha256": sha(p / "receipt.json"), "records_sha256": sha(p / "records.jsonl")}
        for k, p in paths.items()
    },
}
for key, records in rows.items():
    nll = []
    for r in records:
        target = answer_tokens_for_record(r)[r["target_index"]]
        logits = r["raw_logprobs"]
        peak = max(logits.values())
        nll.append(
            peak + math.log(math.fsum(math.exp(v - peak) for v in logits.values())) - logits[target]
        )
    result["models"][key] = {
        **score_records(records),
        "joint_k_plus_u_nll": math.fsum(nll) / count,
        "correct": sum(prediction(r) == r["target_index"] for r in records),
        "evaluation_seconds": reports[key]["duration_seconds"],
    }
for before, after in [("coverage", "ce0"), ("coverage", "pair01"), ("ce0", "pair01")]:
    fixed = regressed = changed = 0
    for a, b in zip(rows[before], rows[after], strict=True):
        x = prediction(a) == a["target_index"]
        y = prediction(b) == b["target_index"]
        fixed += not x and y
        regressed += x and not y
        changed += prediction(a) != prediction(b)
    result["paired"][before + "->" + after] = {
        "corrected": fixed,
        "regressed": regressed,
        "changed_predictions": changed,
    }
with (ROOT / f"reports/s4-claim-pair-{args.source}{count}-comparison-001.json").open("x") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
