"""Validate the B1 grid against the Sample contract and structural invariants."""

from __future__ import annotations

import json
from pathlib import Path

from data.schema import Sample

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "artifacts/s4-ensemble-oracle-001/b1-grid.jsonl"
MANIFEST = REPO / "artifacts/s4-ensemble-oracle-001/b1-grid-manifest.json"


def main() -> int:
    rows = [json.loads(line) for line in GRID.read_text().splitlines() if line.strip()]
    manifest = json.loads(MANIFEST.read_text())
    samples = [Sample.model_validate(row) for row in rows]
    assert len(samples) == manifest["item_count"] == 126, "item count mismatch"
    assert len({s.id for s in samples}) == len(samples), "duplicate sample ids"
    by_qid = {}
    for sample, derivation in zip(samples, manifest["derivations"], strict=True):
        assert sample.questions[0].qid == derivation["qid"], "derivation order mismatch"
        by_qid[derivation["qid"]] = (sample, derivation)
    # twin gold complementarity: before/after twins of one group must have opposite gold
    groups: dict[str, list[tuple[Sample, dict]]] = {}
    for sample, derivation in zip(samples, manifest["derivations"], strict=True):
        groups.setdefault(f"{derivation['scene']}:{derivation['group']}", []).append((sample, derivation))
    twin_checked = 0
    for name, members in groups.items():
        cells = sorted(d["cell"] for _, d in members)
        for sample, derivation in members:
            target = sample.questions[0].target
            if derivation["cell"].startswith("u_premise"):
                assert target.kind == "insufficient", f"{name}: U cell must be insufficient"
                continue
            assert target.kind == "hard" and target.value in {"yes", "no"}, f"{name}: bad hard target"
        for a_index, (first, first_d) in enumerate(members):
            for second, second_d in members[a_index + 1 :]:
                if {first_d["cell"].rsplit("_", 1)[-1], second_d["cell"].rsplit("_", 1)[-1]} == {"before", "after"}:
                    assert (
                        first.questions[0].target.value != second.questions[0].target.value
                    ), f"{name}: twin gold not complementary"
                    twin_checked += 1
    # media binding matches the TRAIN pool row of the same family
    family = manifest["families"][0]
    for sample in samples:
        assert sample.prefix.media[0].sha256 == family["video_media_sha256"], "media sha drift"
        assert sample.prefix.media[0].uri == family["video_uri"], "media uri drift"
        assert sample.meta.pool == "train", "grid must stay in TRAIN pool"
        assert sample.meta.license == "CC0-1.0", "license drift"
    # direct order cells: gold must equal the numeric frame comparison; because
    # every pairwise gold comes from the same numeric entry frames, a strict
    # total order (hence transitivity) holds by construction over these cells.
    direct = [d for d in manifest["derivations"] if d["cell"].startswith("order_direct")]
    entries = {}
    for derivation in direct:
        sample, _ = by_qid[derivation["qid"]]
        gold_yes = sample.questions[0].target.value == "yes"
        if derivation["cell"].endswith("after"):
            gold_yes = not gold_yes
        numeric = derivation["entry_left"] < derivation["entry_right"]
        assert gold_yes == numeric, f"order gold disagrees with frames: {derivation['qid']}"
        entries[(derivation["left"], derivation["right"])] = numeric
    triples_checked = 0
    for (a, b), ab in entries.items():
        for (b2, c), bc in entries.items():
            if b != b2:
                continue
            ac = entries.get((a, c))
            if ac is None:
                continue
            if ab and bc:
                assert ac, "transitivity violation (a<b, b<c but not a<c)"
            if (not ab) and (not bc):
                assert not ac, "transitivity violation (a>b, b>c but a<c)"
            triples_checked += 1
    summary = {
        "rows": len(rows),
        "twin_pairs_checked": twin_checked,
        "transitivity_triples_seen": triples_checked,
        "u_items": sum(1 for d in manifest["derivations"] if d["cell"].startswith("u_premise")),
        "groups": len(groups),
        "scene": family["scene_index"],
        "video_uri": family["video_uri"],
    }
    print(json.dumps(summary, indent=2))
    (REPO / "reports/explore").mkdir(exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
