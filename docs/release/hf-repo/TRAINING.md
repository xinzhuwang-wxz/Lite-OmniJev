# Training: amortized symbolic execution with teacher-distribution distillation

Our training recipe compiles two *structured* supervision signals into a direct-answer adapter. Both are unusual for LoRA releases; both are measured (see [EVALUATIONS](./EVALUATIONS.md)).

## Signal 1 — Teacher-distribution distillation (D3P)

A 27B decision teacher answers the same questions under the same protocol. Instead of distilling its argmax, we distill its **full option distribution** (legal answer set + internal sufficiency mass) into the student's single-token readout. The student learns *how confident to be*, not only *what to answer*.

- Slice scaling measured: 150 → 600 teacher questions improved sealed video accuracy monotonically; the recipe, slices, and hashes are pinned.

## Signal 2 — Program-facet structural supervision (SAS/WSFT)

Where questions derive from programs (CLEVRER train ships ground-truth programs), we compile each program **offline** into per-predicate facet targets — count, color, motion-direction, begin/end/enter/exit, collision — and train on the facets *jointly with* the main questions. Symbol execution happens at training time; the adapter amortizes the executor.

- Ablated: facet supervision with a single-family focus beat both the no-facet control and the all-families-at-once variant (depth beats breadth at fixed step budget — measured, twice).

## What we deliberately do not do

- No RLHF / preference tuning (an RL form is prototyped for the scene-formalizer layer — reward passes through a discrete executor, which is not gradient-decomposable — and is gated by pre-registered F1 targets).
- No distillation of teacher *weights* or APIs; distributions only, on our own training pools.
- No benchmark training: evaluation families are pool-isolated by sample family with provenance hashes.

## Reproducibility

Fixed seeds; per-run bindings for code, data, model, adapter (SHA-256); two-pass exact-match verification on dev benchmarks; sealed holdout drawn pool-isolated and evaluated once. Full chain in the repository `docs/runbooks/` and ADR records.
