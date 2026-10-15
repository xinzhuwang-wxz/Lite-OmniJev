"""Read only SHA-bound TRAIN data; emit aggregates, never sample content."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from data.schema import Sample
from scripts.s1_text_pools import canonical_sample_sha

RULES = {"both_true", "image_false_audio_true", "image_true_audio_false"}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def audit(samples, components, allocations):
    ss = {s.id: s for s in samples}
    cc = {c["item_id"]: c for c in components}
    aa = {a["item_id"]: a for a in allocations}
    require(
        len(ss) == len(samples) == len(cc) == len(components) == len(aa) == len(allocations),
        "duplicate_or_count",
    )
    require(ss.keys() == cc.keys() == aa.keys(), "id_coverage")
    for ident, s in ss.items():
        c, a = cc[ident], aa[ident]
        require(s.meta.pool == c["pool"] == a["pool"] == "train", "train_only")
        require(c["proposal_rule"] in RULES, "rule")
        gold = "yes" if c["proposal_rule"] == "both_true" else "no"
        require(len(s.questions) == 1 and s.questions[0].type == "noul", "question_contract")
        require(s.questions[0].target.value == c["program_truth"] == gold, "gold_rule")
        require(s.questions[0].target.source == "programmatic_component_answer_and", "gold_source")
        require(
            canonical_sample_sha(s) == a["derived_sample_sha256"] == a["original_sample_sha256"],
            "sample_binding",
        )
        require(s.meta.family_id == c["crossmodal_family_id"] == a["family_id"], "family_binding")
        require(
            s.meta.modality == ["image", "audio"] and len(s.prefix.media) == 2, "media_contract"
        )
        for modality in ("image", "audio"):
            media = [m for m in s.prefix.media if m.type == modality]
            require(
                len(media) == 1 and media[0].sha256 == c[modality]["media_sha256"], "media_binding"
            )
    counts = {}
    for modality, negative_rule in [
        ("audio", "image_false_audio_true"),
        ("image", "image_true_audio_false"),
    ]:
        positives = [c for c in components if c["proposal_rule"] == "both_true"]
        negatives = [c for c in components if c["proposal_rule"] == negative_rule]

        def key(c, modality=modality):
            z = c[modality]
            return tuple(
                z[k]
                for k in (
                    "item_id",
                    "family_hash",
                    "source_hash",
                    "media_sha256",
                    "proposal",
                    "proposal_option_text_sha256",
                )
            )

        candidates = [(p, n) for p in positives for n in negatives if key(p) == key(n)]
        # Identical instruction is stronger than shared component. Count separately;
        # different claim wording must never be silently called invariant semantics.
        counts[modality] = {
            "shared_component_and_proposal_pairs": len(candidates),
            "identical_full_instruction_pairs": sum(
                ss[p["item_id"]].questions[0].instruction
                == ss[n["item_id"]].questions[0].instruction
                for p, n in candidates
            ),
        }
    uniqueness = {}
    for modality in ("image", "audio"):
        uniqueness[modality] = {
            k: len({c[modality][k] for c in components})
            for k in ("item_id", "family_hash", "source_hash", "media_sha256")
        }
        uniqueness[modality]["max_reuse_by_item"] = max(
            Counter(c[modality]["item_id"] for c in components).values(), default=0
        )
    return {
        "samples": len(samples),
        "rules": dict(Counter(c["proposal_rule"] for c in components)),
        "pair_capacity": counts,
        "unique_components": uniqueness,
        "gold_consistency": (
            "rule/ledger/sample agreement only; component human truth not re-annotated"
        ),
        "semantic_boundary": (
            "Shared component and proposal is necessary, not sufficient for full "
            "question equivalence. No new compositions created."
        ),
        "adoption": "not_decided",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--train-sha256", required=True)
    p.add_argument("--components-sha256", required=True)
    p.add_argument("--allocation-sha256", required=True)
    p.add_argument("--check-failures", action="store_true")
    args = p.parse_args()
    payloads, hashes = {}, {}
    for name in ("train", "components", "allocation"):
        path = args.root / (name + ".jsonl")
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), "input_path")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        require(actual == getattr(args, name + "_sha256"), name + "_sha256")
        hashes[name] = actual
        payloads[name] = [json.loads(line) for line in raw.splitlines() if line.strip()]
    samples = [Sample.model_validate(s) for s in payloads["train"]]
    result = audit(samples, payloads["components"], payloads["allocation"])
    failures = []
    if args.check_failures:
        for case in (
            "non_train",
            "gold_mismatch",
            "allocation_mismatch",
            "missing_ledger",
            "duplicate_ledger",
        ):
            cs, als = copy.deepcopy(payloads["components"]), copy.deepcopy(payloads["allocation"])
            if case == "non_train":
                cs[0]["pool"] = "dev"
            if case == "gold_mismatch":
                cs[0]["program_truth"] = "U"
            if case == "allocation_mismatch":
                als[0]["derived_sample_sha256"] = "0" * 64
            if case == "missing_ledger":
                cs.pop()
            if case == "duplicate_ledger":
                cs.append(copy.deepcopy(cs[0]))
            try:
                audit(samples, cs, als)
            except ValueError:
                failures.append(case)
            else:
                raise AssertionError("failure_not_rejected:" + case)
    result.update(
        {
            "input_root": str(args.root),
            "input_sha256": hashes,
            "failure_checks": failures,
            "read_scope": "TRAIN only; no media bytes or other pools",
            "schema_version": 1,
        }
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
