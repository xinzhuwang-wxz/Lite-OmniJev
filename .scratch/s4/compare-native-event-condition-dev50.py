"""CPU-only native-course DEV50 paired diagnostics with strict saved-input bindings."""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import time
import zipfile
from pathlib import Path

from data.schema import Sample, answer_tokens
from scripts.s0_development_baselines import sha256_json
from scripts.s0_media_dev import source_hash_for_video
from scripts.s1_text_pools import canonical_sample_sha

path = Path(__file__).with_name("diagnose-temporal-observability.py")
spec = importlib.util.spec_from_file_location("saved_temporal_diagnosis", path)
assert spec and spec.loader
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
inspector = prior.inspector
diagnose = prior.diagnose
PREP_SHA = prior.PREP_SHA
DATA_SHA = prior.DATA_SHA
BASELINE = "f2dbb83b9dc1761041d6f497e777ddbc32f76ee8e36ad0b09162f15b98d43af3"
CANDIDATE = "f5feb5ab5b1f56092b46b1d0f3891bfb36b3fec82d28f96c2db85e08e615d0b8"


def verify_pair(left, right):
    for key in [
        "id",
        "family_id",
        "target_index",
        "answer_token_ids",
        "input_sha256",
        "input_summary_sha256",
        "input_summary",
        "prompt_sha256",
        "payload_sha256",
        "frame_indices",
        "pass_index",
        "sampling_policy",
    ]:
        if key not in left or key not in right or left[key] != right[key]:
            raise ValueError("paired identity/input mismatch: " + key)
    for row in [left, right]:
        if sha256_json(row["input_summary"]) != row["input_summary_sha256"]:
            raise ValueError("input summary digest")


def row_metrics(row, gold):
    p = row["conditional_without_u"]
    if (
        set(p) != {"yes", "no"}
        or not all(math.isfinite(v) and 0 < v < 1 for v in p.values())
        or not math.isclose(sum(p.values()), 1.0)
    ):
        raise ValueError("invalid conditional probabilities")
    raw = row["raw_logprobs"]
    maximum = max(raw[k] for k in p)
    denom = sum(math.exp(raw[k] - maximum) for k in p)
    if not all(math.isclose(p[k], math.exp(raw[k] - maximum) / denom, abs_tol=1e-9) for k in p):
        raise ValueError("raw probabilities disagree")
    if row["prediction"] not in p or p[row["prediction"]] != max(p.values()):
        raise ValueError("prediction readout mismatch")
    return {
        "correct": row["prediction"] == gold,
        "yes": row["prediction"] == "yes",
        "yes_fp": gold == "no" and row["prediction"] == "yes",
        "brier": sum((v - int(k == gold)) ** 2 for k, v in p.items()),
        "nll": -math.log(p[gold]),
    }


def summarize(rows):
    n = len(rows)
    result = {"count": n, "gold_yes": sum(r["gold"] == "yes" for r in rows)}
    for arm in ["baseline", "candidate"]:
        values = [r[arm + "_metrics"] for r in rows]
        result[arm] = {k: sum(v[k] for v in values) for k in ["correct", "yes", "yes_fp"]}
        result[arm].update(
            {k: sum(v[k] for v in values) / n if n else None for k in ["brier", "nll"]}
        )
    result["corrected"] = sum(
        not r["baseline_metrics"]["correct"] and r["candidate_metrics"]["correct"] for r in rows
    )
    result["regressed"] = sum(
        r["baseline_metrics"]["correct"] and not r["candidate_metrics"]["correct"] for r in rows
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    started = time.monotonic()
    m = inspector()
    motion = a.root / "artifacts/s4-motion-coverage-pilot-001"
    candidate = a.root / "artifacts/s4-native-event-condition-dev50-001/uniform"
    baseline = a.root / "artifacts/s4-temporal-coverage-dev50-001/uniform"

    def require(ok, msg):
        if not ok:
            raise ValueError(msg)

    require(m.sha(motion / "preparation.json") == PREP_SHA, "fixed50preparation SHA")
    prepared = json.loads((motion / "preparation.json").read_bytes())
    source = json.loads((a.root / "reports/s3-dev-eval-video-stable-001.json").read_bytes())
    data = Path(source["command"][source["command"].index("--data") + 1])
    require(m.sha(data) == DATA_SHA, "public dev500 SHA")
    all_samples = [
        Sample.model_validate_json(x) for x in data.read_text().splitlines() if x.strip()
    ]
    require(
        len(all_samples) == 500 and all(s.meta.pool == "dev" for s in all_samples),
        "public dev only",
    )
    by_id = {s.id: s for s in all_samples}
    samples = [by_id[r["id"]] for r in prepared["selected"]]
    require(len(samples) == 50, "fixed50count")
    for sample, row in zip(samples, prepared["selected"], strict=True):
        require(canonical_sample_sha(sample) == row["sample_sha256"], "fixed sample SHA")
    records = []
    receipts = []
    for directory, adapter in [(baseline, BASELINE), (candidate, CANDIDATE)]:
        saved = json.loads((directory / "receipt.json").read_bytes())
        require(saved["status"] == "completed" and saved["exit_code"] == 0, "incomplete readout")
        require(saved["identity"]["adapter_sha256"] == adapter, "adapter identity")
        require(saved["policy"] == "uniform" and saved["samples"] == 50, "single uniform50")
        require(m.sha(directory / "records.jsonl") == saved["records_sha256"], "records SHA")
        preparation = directory.parent / "preparation.json"
        require(m.sha(preparation) == saved["preparation_sha256"], "arm preparation SHA")
        receipts.append(saved)
    identities = [
        {
            k: v
            for k, v in r["identity"].items()
            if k not in {"adapter_sha256", "adapter_config_sha256"}
        }
        for r in receipts
    ]
    require(identities[0] == identities[1], "evaluation protocol differs")
    arms = [
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        for path in [baseline / "records.jsonl", candidate / "records.jsonl"]
    ]
    require(all(len(rows) == 50 for rows in arms), "readout count50")
    for path, expected in [
        (a.official / "questions/train.json", m.QUESTIONS_SHA),
        (a.official / "annotations/train/annotation_train.zip", m.ANNOTATIONS_SHA),
    ]:
        require(m.sha(path) == expected, "official source SHA")
    scenes = m.selected_scenes(
        a.official / "questions/train.json", {int(s.meta.family_id.split(":")[1]) for s in samples}
    )
    manifests = {}
    for video, binding in prepared["manifests"].items():
        require(m.sha(binding["path"]) == binding["sha256"], "frame manifest SHA")
        manifests[video] = json.loads(Path(binding["path"]).read_bytes())
    program_counts = collections.Counter()
    with zipfile.ZipFile(a.official / "annotations/train/annotation_train.zip") as zipped:
        for sample, left, right in zip(samples, *arms, strict=True):
            require(left["id"] == right["id"] == sample.id, "paired IDs/order")
            verify_pair(left, right)
            require(
                sha256_json(left["input_summary"]) == left["input_summary_sha256"],
                "input summary SHA",
            )
            for row in [left, right]:
                require(
                    sha256_json(row["input_summary"]) == row["input_summary_sha256"],
                    "each input summary SHA",
                )
                require(
                    answer_tokens(sample.questions[0])[row["target_index"]]
                    == sample.questions[0].target.value,
                    "source gold",
                )
            index = int(sample.meta.family_id.split(":")[1])
            scene = scenes[index]
            question = next(
                q
                for q in scene["questions"]
                if q["question_id"] == int(sample.id.rsplit("-", 1)[-1])
            )
            require(
                source_hash_for_video(
                    {"scene_index": index}, question, sample.prefix.media[0].sha256
                )
                == sample.meta.source_sha256,
                "question/program/media binding",
            )
            require(
                question["answer"].casefold() == sample.questions[0].target.value, "official answer"
            )
            manifest = manifests[sample.prefix.media[0].sha256]
            indices = manifest["uniform_indices"]
            require(
                manifest["total_num_frames"] == 128
                and left["frame_indices"] == right["frame_indices"] == indices,
                "actual uniform indices",
            )
            require(indices == [round(i * 127 / 15) for i in range(16)], "uniform policy")
            names = [n for n in zipped.namelist() if n.endswith(f"/annotation_{index:05d}.json")]
            require(
                len(names) == 1 and zipped.getinfo(names[0]).file_size < 5_000_000,
                "bounded annotation",
            )
            annotation = json.loads(zipped.read(names[0]))
            require(annotation["scene_index"] == index, "annotation family")
            program = question["program"]
            program_counts.update(t for t in program if t.startswith("filter_") or t == "get_frame")
            observed = diagnose(program, annotation, indices, question["answer"].casefold())
            gold = sample.questions[0].target.value
            observed.update(
                baseline_correct=left["prediction"] == gold,
                candidate_correct=right["prediction"] == gold,
                gold=gold,
                baseline_prediction=left["prediction"],
                candidate_prediction=right["prediction"],
            )
            observed["baseline_metrics"] = row_metrics(left, gold)
            observed["candidate_metrics"] = row_metrics(right, gold)
            records.append(observed)
    relation = [r for r in records if r["category"] == "event_relation"]
    absent = [r for r in relation if r["critical_order_proxy"]["target_events"] == 0]
    present = [r for r in relation if r["critical_order_proxy"]["target_events"] > 0]
    motion_rows = [r for r in records if r["category"] != "event_relation"]
    require(
        (len(relation), len(absent), len(motion_rows)) == (22, 8, 28), "frozen program grouping"
    )
    groups = {
        "all": records,
        "native_relation": relation,
        "target_transition_absent_proxy": absent,
        "target_transition_present_proxy": present,
        "motion": motion_rows,
    }
    for category in ["event_conditioned_motion", "endpoint_motion"]:
        groups[category] = [r for r in records if r["category"] == category]
    result = {
        "status": "completed",
        "exit_code": 0,
        "gpu_used": False,
        "cost": None,
        "paired_complete_input_hashes": 50,
        "groups": {k: summarize(v) for k, v in groups.items()},
        "proxy_limitation": (
            "camera transitions are metadata, not executor or pixel visibility proof"
        ),
        "bindings": {
            "helper_sha256": m.sha(Path(__file__)),
            "classification_helper_sha256": m.sha(
                Path(__file__).with_name("diagnose-temporal-observability.py")
            ),
            "public_data_sha256": DATA_SHA,
            "questions_sha256": m.QUESTIONS_SHA,
            "annotations_sha256": m.ANNOTATIONS_SHA,
            "baseline_adapter_sha256": BASELINE,
            "candidate_adapter_sha256": CANDIDATE,
            "baseline_receipt_sha256": m.sha(baseline / "receipt.json"),
            "candidate_receipt_sha256": m.sha(candidate / "receipt.json"),
            "baseline_records_sha256": receipts[0]["records_sha256"],
            "candidate_records_sha256": receipts[1]["records_sha256"],
        },
        "duration_seconds": time.monotonic() - started,
    }
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
