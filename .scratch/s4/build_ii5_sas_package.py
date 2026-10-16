"""II-5 SAS package: modality sub-questions extracted from the TRAIN AND pool.

Each of the 600 AND rows keeps its main row and gains an image sub-question and
an audio sub-question whose golds are the TRUE component answers recorded in
the pool's groups file (never the claim inside the variant). All rows stay in
the TRAIN pool, inside the same crossmodal family, under the existing Sample
contract (HardTarget only). CPU only; no new data sources.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CLAIM_RE = re.compile(
    r"^Are BOTH claims true\? \(1\) In the image, the answer to '(?P<img_q>.*?)' "
    r"with options \[(?P<img_opts>.*?)\] is (?P<img_claim>[A-Z]): .*?\. "
    r"\(2\) For the audio, the answer to '(?P<aud_q>.*?)' is (?P<aud_claim>yes|no)\.$"
)
OPT_RE = re.compile(r"([A-Z]): (.+?)(?:, |$)")


def parse_instruction(text: str) -> dict:
    match = CLAIM_RE.match(text)
    if match is None:
        raise SystemExit(f"instruction does not match AND template: {text[:120]}")
    options = [payload for _, payload in OPT_RE.findall(match.group("img_opts"))]
    letters = [letter for letter, _ in OPT_RE.findall(match.group("img_opts"))]
    if not options or letters != [chr(65 + i) for i in range(len(options))]:
        raise SystemExit(f"option parse failed: {match.group('img_opts')}")
    return {
        "image_question": match.group("img_q"),
        "image_options": options,
        "image_claim": match.group("img_claim"),
        "audio_question": match.group("aud_q"),
        "audio_claim": match.group("aud_claim"),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2] / "artifacts/s4-ensemble-oracle-001"
    train_rows = [json.loads(l) for l in (root / "crossmodal-pairs-train.jsonl").read_text().splitlines() if l.strip()]
    groups = [json.loads(l) for l in (root / "crossmodal-pairs-groups.jsonl").read_text().splitlines() if l.strip()]
    item_to_gold = {}
    for group in groups:
        golds = group["component_gold"]
        for item_id in group["variants"].values():
            item_to_gold[item_id] = golds
    emitted = []
    audit = {"parsed": 0, "and_gold_consistent": 0, "sub_rows": 0}
    for row in train_rows:
        question = row["questions"][0]
        parsed = parse_instruction(question["instruction"])
        audit["parsed"] += 1
        golds = item_to_gold.get(row["id"])
        if golds is None:
            raise SystemExit(f"no group golds for {row['id']}")
        image_gold, audio_gold = golds["image"], golds["audio"]
        if image_gold not in [chr(65 + i) for i in range(len(parsed["image_options"]))]:
            raise SystemExit(f"true image gold {image_gold} not among options for {row['id']}")
        claim_ok = (parsed["image_claim"] == image_gold) and (parsed["audio_claim"] == audio_gold)
        if claim_ok == (question["target"]["value"] == "yes"):
            audit["and_gold_consistent"] += 1
        else:
            raise SystemExit(f"AND gold inconsistent with component truth for {row['id']}")
        emitted.append(row)  # main row unchanged
        image_media = [m for m in row["prefix"]["media"] if m["type"] == "image"]
        audio_media = [m for m in row["prefix"]["media"] if m["type"] == "audio"]
        emitted.append(
            {
                "id": f"{row['id']}-sasimg",
                "prefix": {
                    "state": "Answer the question using the attached image.",
                    "media": image_media,
                },
                "questions": [
                    {
                        "qid": "choice",
                        "type": "choice",
                        "instruction": parsed["image_question"],
                        "options": parsed["image_options"],
                        "score_levels": None,
                        "score_descriptions": None,
                        "target": {
                            "kind": "hard",
                            "value": image_gold,
                            "source": "sas_component_true_gold",
                        },
                        "pair_of": None,
                    }
                ],
                "meta": {
                    **row["meta"],
                    "modality": ["image"],
                    "data_version": row["meta"]["data_version"] + "+sas-sub-v1",
                },
            }
        )
        emitted.append(
            {
                "id": f"{row['id']}-sasaud",
                "prefix": {
                    "state": "Answer the yes/no question using the attached audio.",
                    "media": audio_media,
                },
                "questions": [
                    {
                        "qid": "noul",
                        "type": "noul",
                        "instruction": parsed["audio_question"],
                        "options": None,
                        "score_levels": None,
                        "score_descriptions": None,
                        "target": {
                            "kind": "hard",
                            "value": audio_gold,
                            "source": "sas_component_true_gold",
                        },
                        "pair_of": None,
                    }
                ],
                "meta": {
                    **row["meta"],
                    "modality": ["audio"],
                    "data_version": row["meta"]["data_version"] + "+sas-sub-v1",
                },
            }
        )
        audit["sub_rows"] += 2
    (root / "ii5-sas-package.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in emitted), encoding="utf-8"
    )
    (root / "ii5-sas-package-ledger.jsonl").write_text(
        "".join(json.dumps({"item_id": r["id"], "pool": "train"}) + "\n" for r in emitted),
        encoding="utf-8",
    )
    import hashlib

    print(
        json.dumps(
            {
                **audit,
                "rows": len(emitted),
                "package_sha256": hashlib.sha256((root / "ii5-sas-package.jsonl").read_bytes()).hexdigest(),
                "ledger_sha256": hashlib.sha256((root / "ii5-sas-package-ledger.jsonl").read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
