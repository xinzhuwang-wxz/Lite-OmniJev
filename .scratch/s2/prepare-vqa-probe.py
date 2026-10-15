import hashlib
import json
from pathlib import Path

from scripts.train_s1 import verify_data_is_train_pool

root = Path("/root/autodl-tmp/s1-text-lora-001")
out = root / "artifacts/s2-vqa-probe-001"
out.mkdir(exist_ok=False)
new = root / "artifacts/s2-vqa2-gold-001"
report = json.loads((new / "report.json").read_text())
assert report["sample_count"] == 1811
assert (
    hashlib.sha256((new / "train.jsonl").read_bytes()).hexdigest()
    == report["output_sha256"]["train.jsonl"]
)
provenance = {
    r["id"]: r for r in map(json.loads, (new / "provenance.jsonl").read_text().splitlines())
}
rows = list(map(json.loads, (new / "train.jsonl").read_text().splitlines()))


def select(items, n):
    return sorted(
        items, key=lambda r: hashlib.sha256(("vqa-probe:42:" + r["id"]).encode()).hexdigest()
    )[:n]


selected = []
for category, n in [("color", 160), ("integer", 160), ("yes_no", 80)]:
    part = select([r for r in rows if provenance[r["id"]]["category"] == category], n)
    assert len(part) == n
    selected += part
ledgers = [new / "allocation-ledger.jsonl"]
for directory, n in [
    ("s2-media-pools-image-001", 400),
    ("s1-text-pools-v1-001", 100),
    ("s2-media-pools-screenshot-001", 100),
]:
    path = root / "artifacts" / directory / "run-a"
    selected += select(list(map(json.loads, (path / "train.jsonl").read_text().splitlines())), n)
    ledgers.append(path / "allocation-ledger.jsonl")
assert len(selected) == 1000
p = out / "train.jsonl"
p.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in selected))
check = verify_data_is_train_pool(p, ledgers)
config = json.loads((root / "artifacts/s2-image-focus-001/config.json").read_text())
config["expected_data_versions"].append("vqa2-existing-train-choice-v1")
for name, weight in [("ce", 0.0), ("brier", 1.0)]:
    (out / f"{name}-config.json").write_text(
        json.dumps({**config, "brier_weight": weight}, indent=2) + "\n"
    )
result = {
    "status": "passed",
    "exit_code": 0,
    "rows": 1000,
    "new_gold": 400,
    "old_image": 400,
    "text": 100,
    "screenshot": 100,
    "ledger_check": check,
    "data_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "new_gold_report_sha256": hashlib.sha256((new / "report.json").read_bytes()).hexdigest(),
    "config_sha256": {
        n: hashlib.sha256((out / f"{n}-config.json").read_bytes()).hexdigest()
        for n in ["ce", "brier"]
    },
}
(out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
