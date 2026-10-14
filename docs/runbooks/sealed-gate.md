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
