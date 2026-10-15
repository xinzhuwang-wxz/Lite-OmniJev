"""One gold-preserving image-label smoothing candidate from the frozen focus pool."""

import hashlib
import json
from pathlib import Path

from data.schema import Sample
from scripts.train_s1 import verify_data_is_train_pool

root = Path("/root/autodl-tmp/s1-text-lora-001")
source = root / "artifacts/s2-image-focus-001"
output = root / "artifacts/s2-image-smooth-001"
output.mkdir(exist_ok=False)
original = source / "train.jsonl"
rows = [json.loads(line) for line in original.read_text().splitlines()]
changed = 0
old_version = "aokvqa-train-image-choice-s2-pools-v1"
new_version = "aokvqa-train-image-choice-s2-smooth-v1"
for row in rows:
    if row["meta"]["data_version"] == old_version:
        question = row["questions"][0]
        gold = question["target"]
        assert gold["kind"] == "hard"
        count = len(question["options"])
        index = ord(gold["value"]) - ord("A")
        probabilities = [0.1 / count] * count
        probabilities[index] += 0.9
        assert probabilities.index(max(probabilities)) == index
        question["target"] = {
            "kind": "soft",
            "probs": probabilities,
            "source": "gold_label_smoothing:epsilon=0.1;parent_source=" + gold["source"],
        }
        row["meta"]["data_version"] = new_version
        changed += 1
    Sample.model_validate(row)
assert changed == 600
path = output / "train.jsonl"
path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
config = json.loads((source / "config.json").read_text())
config["expected_data_versions"] = [
    new_version if v == old_version else v for v in config["expected_data_versions"]
]
(output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
checks = verify_data_is_train_pool(
    path,
    [
        root / "artifacts" / d / "run-a/allocation-ledger.jsonl"
        for d in [
            "s1-text-pools-v1-001",
            "s2-media-pools-image-001",
            "s2-media-pools-screenshot-001",
        ]
    ],
)
report = {
    "status": "passed",
    "exit_code": 0,
    "scope": "gold-derived training targets, not teacher labels",
    "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "config_sha256": hashlib.sha256((output / "config.json").read_bytes()).hexdigest(),
    "changed_rows": changed,
    "epsilon": 0.1,
    "gold_argmax_preserved": True,
    "ledger_check": checks,
    "holdout_content_accessed": False,
}
(output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
