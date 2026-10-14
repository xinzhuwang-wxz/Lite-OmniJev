# 现成gold封存与自动gate

当前首发合同：ADR0012（2026-10-15用户明确删除新增真人标注）。使用现成可靠数据集gold/公开benchmark gold/可验证程序真值，无需两名真人、12000次判断、κ或用户提供外部gate账号。

1. S1入训前冻结pool分配规则，预留未训练的holdout/calibration family；查历史dev union及完整实际训练账本，禁止事后挪已训练题。
2. 六cohort各1000holdout，来源许可/revision/文件与媒体hash、真值映射和标签完整性必须有真实证据。已下载不等于已完成这套映射；dev500不能兼作独立最终评测。
3. 自动gate独立作业读取封存内容，开发agent不读内容/逐题答案，只看汇总。模型封存评测只在S4/5/6触发。05负责实际工作流隔离/权限/日志拒绝工程，尚未实现前不宣称已通过；不把root同机chmod当外部安全隔离。
4. 输出来源/数量/池间交集/缺标/非法标签、配置与代码hash、模型指标及退出码。κ为not_applicable，不能填1.0；来源不明/模型伪gold/实际隔离失败不得通过。
5. S4放量和FINAL以05实际完成为前置。它是工程数据任务，不要求用户安排真人或外部账号。

下面v2双标协议仅保留为历史工具兼容说明，不是当前首发验收要求。`eval.sealed_gate`/`scripts.sealed_gate`旧接口仍只实现该历史canary，不能将它当成新的自动gold gate已实现。

---

# Sealed Gate Runbook

S0 sealed evidence is accepted only from an isolated gate that can read raw
human annotations and sealed items. Development agents receive only aggregate
summaries.

## Frozen Protocol

- Each evaluation cohort needs 1,000 holdout items and 500 development items before the
  real gate can be admitted.
- Every holdout item must have two raw independent human labels. A later adjudicated
  consensus may be stored, but it cannot replace raw double labels for Cohen
  kappa.
- The manifest freezes exactly two annotator identities for the cohort. Kappa is
  paired by those identities, not by row order, and replaced or third annotators
  fail closed.
- Contract `sealed-cohort-gate-v2` follows ADR0009: non-canary manifests declare
  text/image/screenshot/audio/video/crossmodal, each with exactly 500 dev and
  1,000 holdout items. The wire field `modality` denotes this evaluation cohort;
  physical model input remains text/image/audio/video.
- Dev uses a separate redacted ledger with item_id, modality, family_hash,
  source_hash, source_dataset, source_revision, license and label_origin.
  Origins are hard_gold/program_truth/human_verified_gold/public_benchmark_gold;
  these declarations require source evidence and do not themselves prove admission.
  No labels or payloads enter this ledger. Dev summaries report no kappa and no
  human annotations. Raw human kappa >=0.7 applies separately to each holdout cohort.
- The allocation is 3,000 dev and 6,000 holdout items, with 12,000 raw human
  judgments for holdout. A dev ledger does not establish human holdout agreement.
- The gate rejects missing labels, duplicate annotators, annotation hash
  mismatches, payload fields, undeclared modalities or pools, family overlap
  between development and holdout pools, and degenerate non-canary kappa.
- The redacted output contains only counts, kappa, failure categories, dataset
  version, source digest, audit id, and whether the run was a canary.
- CLI validation failures are category-only. They must not echo record content,
  labels, item ids, file paths, exception reprs, or sealed payload snippets.
- The current root SSH development environment is not a real isolation
  boundary. A chmod-only or same-host UNIX account setup cannot prove that root
  did not read sealed content. Real acceptance requires an externally
  controlled identity or host.
- Historical four-cohort canaries remain engineering evidence only; they cannot
  be upgraded to real six-cohort receipts. This development CLI always blocks
  real acceptance even when structural and annotation checks pass.

## Canary Command

```bash
uv run python -m scripts.sealed_gate \
  --manifest path/to/canary-manifest.json \
  --annotations path/to/canary-annotations.jsonl \
  --out reports/sealed-gate-canary.json
```

Canary success proves only the parser, rejection semantics, and redacted output
contract. It does not satisfy the real human-label or holdout isolation gate.

## Operator inputs

The independent custodian runs the same pinned source with `--development
path/to/redacted-dev-ledger.jsonl` in addition to the manifest and annotations.
The developer must never run this command with real holdout inputs on the root
GPU host. Provide the operator a source bundle and hashes, declared cohort
counts, frozen two-person identities, allocation exclusions and this runbook;
receive only the aggregate receipt and external identity/provenance evidence.
An external attestation/verifier remains required; this CLI has no implemented
way to self-certify that isolation.
