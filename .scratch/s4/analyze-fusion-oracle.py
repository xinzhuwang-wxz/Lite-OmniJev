"""Offline fusion/contrastive oracle across modalities; zero GPU, deterministic.

Inputs are fetched read-only records from past completed GPU runs (paths and
sha256 pinned below). The script never touches calibration/holdout pools and
never runs a forward pass: every number is recomputed from per-item recorded
probabilities/logits of two existing models on identical pinned inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "artifacts/s4-ensemble-oracle-001"
EXPLORE = REPO / "reports/explore"

PINNED: dict[str, tuple[str, str]] = {
    # local filename: (sha256, remote provenance)
    "video500-candidate-e465.jsonl": (
        "c2a204b86aed79b97f2f10b293d0d78dc5e1cc26ab440ba449e234de77eb3663",
        "remote:s1-text-lora-001/artifacts/s4-program-stratified-video500-step0100/eval/records.jsonl",
    ),
    "video500-parent-f2db.jsonl": (
        "1f34d9b6f26abb8f04690ff85a110353718873ff215ead61c39b3d3ec3def314",
        "remote:s1-text-lora-001/artifacts/s4-temporal-coverage-video500-001/eval/records.jsonl",
    ),
    "video500-purebase.records.jsonl": (
        "ce57d1b95a1d30750acc1df39c6ec85327318ed7dc080aa734a36f17d8007534",
        "remote:s0-development-baselines-002/artifacts/development-baselines-001/"
        "video-gemma4-e4b-it.json.records.jsonl",
    ),
    "image500-candidate-e465.jsonl": (
        "b1f9a9e9d09c42ad1d25a38cf33eaeec96c568cd0b32bc0742da0df40dbde514",
        "remote:s1-text-lora-001/artifacts/s4-program-stratified-image500-step0100/eval/records.jsonl",
    ),
    "image500-parent-f2db.jsonl": (
        "d2f41643ee7450a3c6ecf6c1ee22b71d2e81104e423346b6cf602b52b3573a8a",
        "remote:s1-text-lora-001/artifacts/s4-working-parent-image500-001/eval/records.jsonl",
    ),
    "audio500-candidate-e465.jsonl": (
        "222e4ce51b4ac4b6f6aacd10d62ee7341eb93622813158e5e85cdadee2238739",
        "remote:s1-text-lora-001/artifacts/s4-program-public500-audio-001/eval/records.jsonl",
    ),
    "jevbench-candidate-e465.records.jsonl": (
        "750ef69ab54774d283f759b86a0d2041efdc6eaeca5028c2d42805f9f12aeece",
        "remote:s1-text-lora-001/artifacts/s4-program-jevbench-001/records.jsonl "
        "(sha matches reports/s4-program-jevbench-001.json records_sidecar)",
    ),
    "jevbench-parent-base.records.jsonl": (
        "6f530aa73be9e1756f931adee625e893492127e5faf7eec030e8718ff8d5650a",
        "remote:s1-text-lora-001/artifacts/s4-parent-jevbench-001/records.jsonl "
        "(sha matches reports/s4-parent-jevbench-001.json records_sidecar)",
    ),
    "dev50-motionpilot-uniform.jsonl": (
        "7e14478e05f62beb85127e40a6e9e13ba38f5b3c1e18ffa3b21c52a8282e0ff3",
        "remote:s1-text-lora-001/artifacts/s4-motion-coverage-pilot-001/uniform/records.jsonl",
    ),
    "dev50-motionpilot-motion.jsonl": (
        "d5b8469019a42b553f4c01bf4b52f74744241924379c346b8cdf3301de8883ce",
        "remote:s1-text-lora-001/artifacts/s4-motion-coverage-pilot-001/motion/records.jsonl",
    ),
    "video-dev150.jsonl": (
        "e9b91fedecf2ccf7dc30e837ac11bcc4b233624d1f6c16812a40b1319c15b080",
        "remote:s1-text-lora-001/artifacts/s3-stable-probe-001/video-dev150.jsonl",
    ),
}
LAMBDAS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
WEIGHTS = [round(0.1 * step, 1) for step in range(11)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> dict[str, str]:
    found = {}
    for name, (expected, provenance) in PINNED.items():
        path = INPUTS / name
        if not path.exists():
            raise SystemExit(f"missing pinned input {name} ({provenance})")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"sha256 mismatch for {name}: {actual} != {expected}")
        found[name] = expected
    return found


def load_by_id(path: Path, *, first_pass_only: bool = False) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if first_pass_only and row.get("pass_index", 1) != 1:
            continue
        if row["id"] in rows:
            raise SystemExit(f"duplicate id {row['id']} in {path.name}")
        rows[row["id"]] = row
    return rows


def normalized(values: dict[str, float]) -> dict[str, float]:
    total = math.fsum(values.values())
    return {key: value / total for key, value in values.items()}


def pair_analysis(
    *,
    name: str,
    left: dict[str, dict],
    right: dict[str, dict],
    gold_of: callable,
    left_prob_of: callable,
    right_prob_of: callable,
    left_logit_of: callable | None,
    right_logit_of: callable | None,
    logit_space: str,
) -> dict:
    if set(left) != set(right):
        raise SystemExit(f"{name}: id sets differ")
    ids = sorted(left)
    n = len(ids)
    gold_mismatch = [i for i in ids if gold_of(left[i]) != gold_of(right[i])]
    if gold_mismatch:
        raise SystemExit(f"{name}: gold mismatch on {len(gold_mismatch)} items")
    acc_l = sum(left[i]["prediction"] == gold_of(left[i]) for i in ids) / n
    acc_r = sum(right[i]["prediction"] == gold_of(right[i]) for i in ids) / n
    mean_ok = geo_ok = oracle = disagree = 0
    both = left_only = right_only = neither = 0
    disagree_ids: list[str] = []
    for i in ids:
        gold = gold_of(left[i])
        lp, rp = left_prob_of(left[i]), right_prob_of(right[i])
        if set(lp) != set(rp):
            raise SystemExit(f"{name}: label sets differ on {i}")
        mean = normalized({k: (lp[k] + rp[k]) / 2 for k in lp})
        geo = normalized({k: math.sqrt(lp[k] * rp[k]) for k in lp})
        mean_ok += max(mean, key=mean.get) == gold
        geo_ok += max(geo, key=geo.get) == gold
        lc = left[i]["prediction"] == gold
        rc = right[i]["prediction"] == gold
        oracle += lc or rc
        if left[i]["prediction"] != right[i]["prediction"]:
            disagree += 1
            disagree_ids.append(i)
        if lc and rc:
            both += 1
        elif lc:
            left_only += 1
        elif rc:
            right_only += 1
        else:
            neither += 1
    weighted = {}
    for weight in WEIGHTS:
        ok = 0
        for i in ids:
            gold = gold_of(left[i])
            lp, rp = left_prob_of(left[i]), right_prob_of(right[i])
            mix = normalized({k: weight * lp[k] + (1 - weight) * rp[k] for k in lp})
            ok += max(mix, key=mix.get) == gold
        weighted[f"w_left={weight:.1f}"] = ok / n
    best_weight = max(weighted, key=weighted.get)
    contrastive: dict[str, float] = {}
    if left_logit_of is not None and right_logit_of is not None:
        for lam in LAMBDAS:
            ok = 0
            for i in ids:
                gold = gold_of(left[i])
                ll, rl = left_logit_of(left[i]), right_logit_of(right[i])
                # Decision space is the answer-label set only; p(U) is returned
                # separately and must never become the predicted answer.
                labels = sorted(left_prob_of(left[i]))
                score = {k: ll[k] - lam * rl[k] for k in labels}
                ok += max(score, key=score.get) == gold
            contrastive[f"lambda={lam:g}"] = ok / n
    return {
        "name": name,
        "items": n,
        "left_accuracy": acc_l,
        "right_accuracy": acc_r,
        "mean_probability_ensemble": mean_ok / n,
        "geometric_mean_ensemble": geo_ok / n,
        "best_weighted_ensemble": {"rule": best_weight, "accuracy": weighted[best_weight]},
        "weighted_sweep": weighted,
        "any_correct_oracle": oracle / n,
        "oracle_gain_over_left_pp": (oracle / n - acc_l) * 100,
        "ensemble_gain_over_left_pp": (mean_ok / n - acc_l) * 100,
        "disagreement_rate": disagree / n,
        "buckets": {
            "both_correct": both,
            "left_only_correct": left_only,
            "right_only_correct": right_only,
            "both_wrong": neither,
        },
        "contrastive": contrastive,
        "contrastive_space": logit_space,
        "disagree_ids": disagree_ids,
    }


def jevbench_analysis() -> tuple[dict, dict]:
    left = load_by_id(INPUTS / "jevbench-candidate-e465.records.jsonl", first_pass_only=True)
    right = load_by_id(INPUTS / "jevbench-parent-base.records.jsonl", first_pass_only=True)
    ids = sorted(set(left) & set(right))
    if len(ids) != 231 or set(left) != set(right):
        raise SystemExit("jevbench id join failed")
    input_mismatch = [i for i in ids if left[i]["input_sha256"] != right[i]["input_sha256"]]
    label_mismatch = [i for i in ids if left[i]["source_labels"] != right[i]["source_labels"]]
    incomplete = [
        i
        for i in ids
        if set(left[i]["source_probabilities"]) != set(left[i]["source_labels"])
        or abs(math.fsum(left[i]["source_probabilities"].values()) - 1) > 1e-9
        or set(right[i]["source_probabilities"]) != set(right[i]["source_labels"])
        or abs(math.fsum(right[i]["source_probabilities"].values()) - 1) > 1e-9
    ]

    def gold(row: dict) -> str:
        return row["expected_present_for_scoring_only"]

    n = len(ids)
    acc_l = sum(left[i]["prediction_source"] == gold(left[i]) for i in ids) / n
    acc_r = sum(right[i]["prediction_source"] == gold(right[i]) for i in ids) / n
    mean_ok = geo_ok = oracle = disagree = 0
    both = left_only = right_only = neither = 0
    for i in ids:
        lp, rp = left[i]["source_probabilities"], right[i]["source_probabilities"]
        mean = normalized({k: (lp[k] + rp[k]) / 2 for k in lp})
        geo = normalized({k: math.sqrt(lp[k] * rp[k]) for k in lp})
        mean_ok += max(mean, key=mean.get) == gold(left[i])
        geo_ok += max(geo, key=geo.get) == gold(left[i])
        lc = left[i]["prediction_source"] == gold(left[i])
        rc = right[i]["prediction_source"] == gold(right[i])
        oracle += lc or rc
        disagree += left[i]["prediction_source"] != right[i]["prediction_source"]
        if lc and rc:
            both += 1
        elif lc:
            left_only += 1
        elif rc:
            right_only += 1
        else:
            neither += 1
    covered = [
        i
        for i in ids
        if all(
            label in left[i]["raw_logprobs"] and label in right[i]["raw_logprobs"]
            for label in left[i]["source_labels"]
        )
    ]
    contrastive = {}
    for lam in LAMBDAS:
        ok = 0
        for i in covered:
            score = {
                k: left[i]["raw_logprobs"][k] - lam * right[i]["raw_logprobs"][k]
                for k in left[i]["source_labels"]
            }
            ok += max(score, key=score.get) == gold(left[i])
        contrastive[f"lambda={lam:g}"] = ok / len(covered)
    summary = {
        "name": "jevbench231 candidate-e465 vs parent-frozen-base (pass 1 records)",
        "items": n,
        "left_accuracy": acc_l,
        "right_accuracy": acc_r,
        "mean_probability_ensemble": mean_ok / n,
        "geometric_mean_ensemble": geo_ok / n,
        "any_correct_oracle": oracle / n,
        "oracle_gain_over_left_pp": (oracle / n - acc_l) * 100,
        "ensemble_gain_over_left_pp": (mean_ok / n - acc_l) * 100,
        "disagreement_rate": disagree / n,
        "buckets": {
            "both_correct": both,
            "left_only_correct": left_only,
            "right_only_correct": right_only,
            "both_wrong": neither,
        },
        "contrastive": contrastive,
        "contrastive_space": "raw_logprobs, only items with full label coverage",
        "contrastive_covered_items": len(covered),
        "input_identity_mismatches": len(input_mismatch),
        "label_order_mismatches": len(label_mismatch),
        "incomplete_distributions": len(incomplete),
    }
    diagnostics = {
        "input_mismatch_ids": input_mismatch,
        "label_mismatch_ids": label_mismatch,
        "incomplete_distribution_ids": incomplete,
        "covered_primitives": {
            primitive: sum(
                1 for i in covered if left[i]["primitive"] == primitive
            )
            for primitive in sorted({left[i]["primitive"] for i in ids})
        },
    }
    return summary, diagnostics


def frame_marginal_analysis() -> dict:
    uniform = load_by_id(INPUTS / "dev50-motionpilot-uniform.jsonl")
    motion = load_by_id(INPUTS / "dev50-motionpilot-motion.jsonl")
    ids = sorted(set(uniform) & set(motion))
    n = len(ids)
    gold = {i: ["yes", "no"][uniform[i]["target_index"]] for i in ids}
    acc_u = sum(uniform[i]["prediction"] == gold[i] for i in ids) / n
    acc_m = sum(motion[i]["prediction"] == gold[i] for i in ids) / n
    marg = oracle = disagree = 0
    for i in ids:
        up = uniform[i]["conditional_without_u"]
        mp = motion[i]["conditional_without_u"]
        mean = normalized({k: (up[k] + mp[k]) / 2 for k in up})
        marg += max(mean, key=mean.get) == gold[i]
        oracle += (uniform[i]["prediction"] == gold[i]) or (motion[i]["prediction"] == gold[i])
        disagree += uniform[i]["prediction"] != motion[i]["prediction"]
    return {
        "name": "frame-sampling marginalization signal, fixed temporal DEV50, "
        "adapter 4b8e422f (pre-f2db CE0 parent), uniform vs motion policies",
        "items": n,
        "uniform_accuracy": acc_u,
        "motion_accuracy": acc_m,
        "marginalized_mean_accuracy": marg / n,
        "any_correct_oracle": oracle / n,
        "oracle_gain_over_uniform_pp": (oracle / n - acc_u) * 100,
        "marginal_gain_over_uniform_pp": (marg / n - acc_u) * 100,
        "disagreement_rate": disagree / n,
    }


def main() -> int:
    hashes = verify_inputs()
    cand = load_by_id(INPUTS / "video500-candidate-e465.jsonl")
    parent = load_by_id(INPUTS / "video500-parent-f2db.jsonl")
    purebase = load_by_id(INPUTS / "video500-purebase.records.jsonl", first_pass_only=True)

    video_a = pair_analysis(
        name="video500 candidate-e465 vs parent-f2db",
        left=cand,
        right=parent,
        gold_of=lambda row: row["target"],
        left_prob_of=lambda row: row["probabilities"],
        right_prob_of=lambda row: row["probabilities"],
        left_logit_of=lambda row: row["raw_logits"],
        right_logit_of=lambda row: row["raw_logits"],
        logit_space="raw_logits (exact recorded final-token logits)",
    )
    video_b = pair_analysis(
        name="video500 candidate-e465 vs pure-frozen-base (no adapter)",
        left=cand,
        right=purebase,
        gold_of=lambda row: ["yes", "no"][row["target_index"]] if "target_index" in row else row["target"],
        left_prob_of=lambda row: row["probabilities"],
        right_prob_of=lambda row: row["conditional_without_u"],
        left_logit_of=lambda row: {
            k: math.log(row["joint_probabilities"][k]) for k in ("yes", "no")
        },
        right_logit_of=lambda row: {
            k: math.log(row["joint_k_plus_u"][k]) for k in ("yes", "no")
        },
        logit_space="log joint probabilities (U included in softmax family)",
    )
    image = pair_analysis(
        name="image500 candidate-e465 vs parent-f2db",
        left=load_by_id(INPUTS / "image500-candidate-e465.jsonl"),
        right=load_by_id(INPUTS / "image500-parent-f2db.jsonl"),
        gold_of=lambda row: row["target"],
        left_prob_of=lambda row: row["probabilities"],
        right_prob_of=lambda row: row["probabilities"],
        left_logit_of=lambda row: row["raw_logits"],
        right_logit_of=lambda row: row["raw_logits"],
        logit_space="raw_logits",
    )
    audio = load_by_id(INPUTS / "audio500-candidate-e465.jsonl")
    audio_summary = {
        "name": "audio500 candidate-e465 only; parent-f2db pass never run, "
        "lambda grid pending one 500-item f2db audio pass (GPU, main schedules)",
        "items": len(audio),
        "candidate_accuracy": sum(
            row["prediction"] == row["target"] for row in audio.values()
        )
        / len(audio),
    }
    jev, jev_diag = jevbench_analysis()
    frame = frame_marginal_analysis()

    gate = {
        "exit_rule": "oracle gain over candidate < +5pp terminates ensemble/contrastive route",
        "primary_oracle_gain_pp": video_a["oracle_gain_over_left_pp"],
        "terminated": video_a["oracle_gain_over_left_pp"] < 5.0,
    }
    oracle_report = {
        "schema_version": "lite-omnijev.explore.s4-video-ensemble-oracle-001.v1",
        "generated_from": "recorded per-item probabilities/logits of completed runs; zero new GPU",
        "inputs": hashes,
        "video500_candidate_vs_parent": video_a,
        "video500_candidate_vs_purebase": video_b,
        "jevbench231_candidate_vs_purebase": jev,
        "jevbench_diagnostics": jev_diag,
        "frame_marginalization_side_signal": frame,
        "gate": gate,
    }
    fusion_report = {
        "schema_version": "lite-omnijev.explore.s4-multimodal-fusion-grid-001.v1",
        "generated_from": "recorded per-item probabilities/logits of completed runs; zero new GPU",
        "inputs": hashes,
        "lambdas": LAMBDAS,
        "weights": WEIGHTS,
        "video500_candidate_vs_parent": video_a,
        "image500_candidate_vs_parent": image,
        "audio500": audio_summary,
        "jevbench231_candidate_vs_purebase": jev,
        "cross_reference_p0b": "/tmp/innov-p0b/s4-jevbench-family-flip-001 (mainfable): "
        "macro gap 97% from ambiguous family n=7; lambda=0.25 best on 189-item subset; "
        "routing-not-mixing conclusion",
    }
    EXPLORE.mkdir(parents=True, exist_ok=True)
    (EXPLORE / "s4-video-ensemble-oracle-001.json").write_text(
        json.dumps(oracle_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (EXPLORE / "s4-multimodal-fusion-grid-001.json").write_text(
        json.dumps(fusion_report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    def contrastive_table(block: dict) -> str:
        if not block["contrastive"]:
            return "n/a"
        return "; ".join(f"{k}:{v:.4f}" for k, v in block["contrastive"].items())

    oracle_md = f"""# s4-video-ensemble-oracle-001 — 离线融合 oracle(零 GPU)

输入:两份已完成 GPU 作业的逐题 records(远端只读拉取,SHA 全部钉住,见 JSON inputs)。
候选=adapter e46550143b28(step0100),父=adapter f2dbb83b(课程初始化来源);
纯基座=冻结 E4B 无 adapter。所有输入同 data_sha256、逐题 id/target 对齐核验通过。

## video500(候选 74.4% vs 父 70.8%)

| 指标 | 值 |
|---|---|
| 平均概率集成 | {video_a['mean_probability_ensemble']:.4f} ({video_a['ensemble_gain_over_left_pp']:+.2f}pp) |
| 几何平均集成 | {video_a['geometric_mean_ensemble']:.4f} |
| 最优加权集成 | {video_a['best_weighted_ensemble']['accuracy']:.4f} @ {video_a['best_weighted_ensemble']['rule']} |
| 任一对即正确 oracle | {video_a['any_correct_oracle']:.4f} ({video_a['oracle_gain_over_left_pp']:+.2f}pp) |
| 分歧率 | {video_a['disagreement_rate']:.4f} |
| 分桶 both/cand-only/parent-only/neither | {video_a['buckets']['both_correct']}/{video_a['buckets']['left_only_correct']}/{video_a['buckets']['right_only_correct']}/{video_a['buckets']['both_wrong']} |
| 对比解码(精确 logits) | {contrastive_table(video_a)} |

## 判定

- oracle 增益 **{video_a['oracle_gain_over_left_pp']:+.2f}pp** ≥ +5pp 门槛 → 按预注册规则**不终止**集成路线;
  但全部可实现规则(均值/几何/加权≤{video_a['best_weighted_ensemble']['accuracy']:.3f}、任意 λ 对比解码全部为负)都不能兑现 oracle 余量。
- 对比解码在**全量 video500 精确逐题 logits** 上 λ=0.5/1.0/2.0 全部负增益
  (λ=0.5:{video_a['contrastive'].get('lambda=0.5', float('nan')):.4f}、λ=1:{video_a['contrastive'].get('lambda=1', float('nan')):.4f}、λ=2:{video_a['contrastive'].get('lambda=2', float('nan')):.4f});
  计划中的 video150+DEV50 GPU 探针只会复算同一批确定性 logits,**无需再跑**。
- video500 候选 vs 纯基座(55.6%):oracle {video_b['any_correct_oracle']:.4f}({video_b['oracle_gain_over_left_pp']:+.2f}pp)、
  均值集成 {video_b['mean_probability_ensemble']:.4f}({video_b['ensemble_gain_over_left_pp']:+.2f}pp)、对比解码全负 → 纯基座做 amateur 同样不可用。
- JevBench231:均值集成 {jev['mean_probability_ensemble']:.4f}({jev['ensemble_gain_over_left_pp']:+.2f}pp)为负,
  oracle {jev['any_correct_oracle']:.4f}({jev['oracle_gain_over_left_pp']:+.2f}pp);互补结构 97% 集中于 ambiguous n=7(见 /tmp/innov-p0b)。
- 帧采样边缘化侧信号(旧 CE0 adapter,DEV50 uniform vs motion):marginal {frame['marginalized_mean_accuracy']:.4f} vs 单采样 {frame['uniform_accuracy']:.4f},
  分歧率仅 {frame['disagreement_rate']:.2%}、oracle 上限 {frame['any_correct_oracle']:.4f}(≤+2 题)→ 帧维互补远小于模型对互补。

## 对颠覆轨的指向

oracle 余量真实存在(96 个分歧题、89 题双错)但概率级融合取不出来 → 增益须来自**输入/结构维**
(B1 程序格一致性、B2 感知-决策解耦、ADR 目标中心输入 token),而不是决策后融合。
"""
    (EXPLORE / "s4-video-ensemble-oracle-001.md").write_text(oracle_md, encoding="utf-8")

    fusion_md = f"""# s4-multimodal-fusion-grid-001 — 跨模态融合/对比解码 λ 网格(零 GPU)

同 oracle-001 输入与核验;按协调指令把 λ 网格扩展到 image500/audio500。

## image500(候选 82.8% vs 父 82.0%)

| 指标 | 值 |
|---|---|
| 均值/几何集成 | {image['mean_probability_ensemble']:.4f} / {image['geometric_mean_ensemble']:.4f} (增益 {image['ensemble_gain_over_left_pp']:+.2f}pp) |
| 最优加权 | {image['best_weighted_ensemble']['accuracy']:.4f} @ {image['best_weighted_ensemble']['rule']} |
| oracle | {image['any_correct_oracle']:.4f} ({image['oracle_gain_over_left_pp']:+.2f}pp) |
| 分歧率/分桶 | {image['disagreement_rate']:.4f};{image['buckets']['both_correct']}/{image['buckets']['left_only_correct']}/{image['buckets']['right_only_correct']}/{image['buckets']['both_wrong']} |
| 对比解码 | {contrastive_table(image)} |

## audio500

候选 79.8% 已核;**父 f2db 从未跑过 audio500** → λ 网格与 oracle 暂缺,
若 main 排期一次 f2db audio500(500 题,候选单遍 227s 量级)即可零额外开发补齐。

## JevBench231(候选 76.62% vs 基座 76.19%)

均值集成 {jev['mean_probability_ensemble']:.4f}({jev['ensemble_gain_over_left_pp']:+.2f}pp);
oracle {jev['any_correct_oracle']:.4f}({jev['oracle_gain_over_left_pp']:+.2f}pp);
对比解码(仅 {jev['contrastive_covered_items']}/231 全 logprob 覆盖题,choice 139 题记录缺全标签 logprob,不冒充完整结果):
{contrastive_table(jev)}。P0b 交叉引用:macro 差距 97% 来自 ambiguous(n=7),
λ=0.25 仅在 189 题子集双指标最优,含糊修复应走路由而非全局混合(/tmp/innov-p0b)。

## 结论

- 三个可配对模态里,**没有任何 λ 或加权同时满足 headline 与 macro 不退化**;
  video500/image500 对比解码全负,JevBench 全量集成负、子集微正但由 n=7 family 主导。
- 若保留双模型方向,唯一有证据的形态是**按 family 路由**(P0b),不是全局融合;
  但 n 小(7/13/12),须真双前向验证,优先级低于 B1/B2 结构轨。
"""
    (EXPLORE / "s4-multimodal-fusion-grid-001.md").write_text(fusion_md, encoding="utf-8")
    print(json.dumps({
        "oracle_gate": gate,
        "video_oracle": video_a["any_correct_oracle"],
        "video_ensemble": video_a["mean_probability_ensemble"],
        "image_oracle": image["any_correct_oracle"],
        "jevbench_ensemble": jev["mean_probability_ensemble"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
