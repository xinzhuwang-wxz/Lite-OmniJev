# Sealed Gate Runbook

S0 sealed evidence is accepted only from an isolated gate that can read raw
human annotations and sealed items. Development agents receive only aggregate
summaries.

## Frozen Protocol

- Each modality needs 1,000 holdout items and 500 development items before the
  real gate can be admitted.
- Every item must have two raw independent human labels. A later adjudicated
  consensus may be stored, but it cannot replace raw double labels for Cohen
  kappa.
- The manifest freezes exactly two annotator identities for the cohort. Kappa is
  paired by those identities, not by row order, and replaced or third annotators
  fail closed.
- Non-canary manifests must declare all four modalities with exactly 500 dev and
  1,000 holdout items per modality.
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

## Canary Command

```bash
uv run python -m scripts.sealed_gate \
  --manifest path/to/canary-manifest.json \
  --annotations path/to/canary-annotations.jsonl \
  --out reports/sealed-gate-canary.json
```

Canary success proves only the parser, rejection semantics, and redacted output
contract. It does not satisfy the real human-label or holdout isolation gate.
