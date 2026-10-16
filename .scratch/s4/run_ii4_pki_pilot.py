"""II-4 PKI zero-GPU pilot: lexical program-key kNN over TRAIN, DEV-logit interpolation.

Datastore: TRAIN video pool (8299) + TRAIN crossmodal AND rows (600), keys built
from word/char-ngram lexical features plus modality/type/operator signature;
values are gold one-hot answer distributions. Queries: cached candidate
distributions on the original DEV video500 and crossmodal500 records. Lambda is
fit on a DEV half-split (diagnostic only; production lambda belongs to the
private calibration fit). TRAIN/DEV only; calibration/holdout never touched.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "artifacts/s4-ensemble-oracle-001"
K = 10
LAMBDA_GRID = [round(0.1 * i, 1) for i in range(11)]
TOKEN = re.compile(r"[a-z0-9']+")
OPERATORS = (
    "before",
    "after",
    "enter",
    "enter",
    "leave",
    "leaves",
    "collision",
    "collide",
    "hit",
    "exist",
    "both",
    "how many",
    "what",
    "is there",
    "bell",
)
DIM = 4096


def hash_token(text: str) -> int:
    return int.from_bytes(text.encode(), "big") % DIM


def features(instruction: str, modality: list[str], qtype: str) -> dict[int, float]:
    words = TOKEN.findall(instruction.lower())
    bag: dict[int, float] = {}
    for word in words:
        key = hash_token("w:" + word)
        bag[key] = bag.get(key, 0.0) + 1.0
    lowered = instruction.lower()
    for operator in OPERATORS:
        if operator in lowered:
            key = hash_token("op:" + operator)
            bag[key] = bag.get(key, 0.0) + 3.0
    bag[hash_token("mod:" + "+".join(sorted(modality)))] = 4.0
    bag[hash_token("typ:" + qtype)] = 4.0
    norm = math.sqrt(math.fsum(v * v for v in bag.values()))
    return {k: v / norm for k, v in bag.items()} if norm else bag


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return math.fsum(v * b.get(k, 0.0) for k, v in a.items())


def ece15(probabilities: list[float], corrects: list[bool]) -> float:
    bins = 10
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [(p, c) for p, c in zip(probabilities, corrects) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not members:
            continue
        conf = math.fsum(p for p, _ in members) / len(members)
        acc = math.fsum(1.0 if c else 0.0 for _, c in members) / len(members)
        ece += len(members) / len(probabilities) * abs(conf - acc)
    return ece


def gold_distribution(row: dict) -> dict[str, float]:
    target = row["questions"][0]["target"] if "questions" in row else row.get("target")
    if isinstance(target, dict) and target.get("kind") == "hard":
        value = str(target["value"]).lower()
        if value in ("yes", "no"):
            return {"yes": 1.0 if value == "yes" else 0.0, "no": 1.0 if value == "no" else 0.0}
    return {}


def load_datastore() -> tuple[list[dict], list[dict]]:
    entries = []
    for name, modality in [
        ("clevrer-train-pool.jsonl", ["video"]),
        ("crossmodal-pairs-train.jsonl", ["image", "audio"]),
    ]:
        for line in (ROOT / name).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question = row["questions"][0]
            dist = gold_distribution(row)
            if not dist:
                continue
            entries.append(
                {
                    "vector": features(question["instruction"], modality, question["type"]),
                    "dist": dist,
                    "id": row["id"],
                }
            )
    return entries, []


def load_queries(cohort: str) -> list[dict]:
    texts = {
        json.loads(line)["id"]: json.loads(line)
        for line in (ROOT / f"dev-{cohort}-text.jsonl").read_text().splitlines()
        if line.strip()
    }
    records = [
        json.loads(line)
        for line in (ROOT / f"{cohort}500-candidate-e465.jsonl").read_text().splitlines()
        if line.strip()
    ]
    queries = []
    for record in records:
        text = texts.get(record["id"])
        if text is None:
            continue
        probs = record.get("probabilities") or record.get("conditional_without_u")
        gold = ["yes", "no"][record["target_index"]] if "target_index" in record else record["target"]
        queries.append(
            {
                "id": record["id"],
                "vector": features(text["instruction"], text["modality"], text["type"]),
                "p_yes": probs["yes"],
                "gold_yes": gold == "yes",
            }
        )
    return queries


def neighbors(entries: list[dict], query: dict) -> tuple[list[tuple[float, dict[str, float]]], dict[str, float]]:
    scored = [(cosine(query["vector"], entry["vector"]), entry) for entry in entries]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:K]
    distances = sorted(1.0 - score for score, _ in top)
    tau = max(distances[len(distances) // 2], 1e-6)
    weights = [math.exp(-(1.0 - score) / tau) for score, _ in top]
    total = math.fsum(weights)
    mixed = {"yes": 0.0, "no": 0.0}
    for (score, entry), weight in zip(top, weights, strict=True):
        mixed["yes"] += weight / total * entry["dist"]["yes"]
        mixed["no"] += weight / total * entry["dist"]["no"]
    return [(score, entry["dist"]) for score, entry in top], mixed


def evaluate(queries: list[dict], mixed_by_id: dict[str, dict[str, float]], lam: float) -> dict:
    probs, corrects = [], []
    for query in queries:
        mixed = mixed_by_id[query["id"]]
        p_yes = (1 - lam) * query["p_yes"] + lam * mixed["yes"]
        probs.append(p_yes)
        corrects.append((p_yes >= 0.5) == query["gold_yes"])
    accuracy = math.fsum(1.0 if c else 0.0 for c in corrects) / len(corrects)
    return {"accuracy": accuracy, "ece15": ece15(probs, corrects)}


def nll(queries: list[dict], mixed_by_id: dict[str, float], lam: float) -> float:
    total = 0.0
    for query in queries:
        mixed = mixed_by_id[query["id"]]
        p_yes = min(max((1 - lam) * query["p_yes"] + lam * mixed["yes"], 1e-9), 1 - 1e-9)
        total += -math.log(p_yes if query["gold_yes"] else 1 - p_yes)
    return total / len(queries)


def main() -> int:
    entries, _ = load_datastore()
    report = {"datastore": len(entries), "k": K, "cohorts": {}}
    for cohort in ("video", "crossmodal"):
        queries = load_queries(cohort)
        neighbor_info, mixed_by_id = {}, {}
        gold_hit = 0
        for query in queries:
            top, mixed = neighbors(entries, query)
            mixed_by_id[query["id"]] = mixed
            neighbor_info[query["id"]] = [score for score, _ in top]
            gold_hit += any(dist["yes"] == (1.0 if query["gold_yes"] else 0.0) for _, dist in top)
        report["cohorts"][cohort] = {
            "queries": len(queries),
            "neighbor_recall_at10_gold": gold_hit / len(queries),
            "baseline": evaluate(queries, mixed_by_id, 0.0),
            "lambda_grid": {
                f"{lam:.1f}": evaluate(queries, mixed_by_id, lam) for lam in LAMBDA_GRID
            },
        }
        # two-fold lambda fit on DEV halves (diagnostic)
        queries.sort(key=lambda q: q["id"])
        half = len(queries) // 2
        folds = [queries[:half], queries[half:]]
        fitted = []
        for index, test in enumerate(folds):
            train_fold = folds[1 - index]
            best = min(
                LAMBDA_GRID,
                key=lambda lam: nll(train_fold, mixed_by_id, lam),
            )
            fitted.append({"fit_lambda": best, "test": evaluate(test, mixed_by_id, best)})
        report["cohorts"][cohort]["twofold_lambda_fit"] = fitted
        # conditional lambda by candidate confidence tercile
        ordered = sorted(queries, key=lambda q: abs(q["p_yes"] - 0.5))
        tercile = len(ordered) // 3
        groups = [ordered[:tercile], ordered[tercile : 2 * tercile], ordered[2 * tercile :]]
        conditional = []
        for index, test in enumerate(groups):
            train_groups = [g for j, g in enumerate(groups) if j != index]
            pool = [q for g in train_groups for q in g]
            best = min(LAMBDA_GRID, key=lambda lam: nll(pool, mixed_by_id, lam))
            conditional.append(
                {
                    "confidence_tercile": index,
                    "fit_lambda": best,
                    "test": evaluate(test, mixed_by_id, best),
                }
            )
        report["cohorts"][cohort]["conditional_lambda_terciles"] = conditional
    out = Path(__file__).resolve().parents[2] / "reports/explore/s4-ii4-pki-pilot-001.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["cohorts"]["video"]["baseline"], indent=2))
    print(json.dumps(report["cohorts"]["crossmodal"]["baseline"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
