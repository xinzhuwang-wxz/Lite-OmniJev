# Evaluations

> ⚠ **NUMBERS PENDING FINAL MODEL** — tables below carry the current candidate's sealed numbers as format examples. At release, `scripts/finalize_hf_numbers.py` injects the final sealed summary (source: sealed 6,000-question holdout, single protocol, single forward pass, two-pass exact-match verified on dev).

## Sealed holdout (6,000 questions, temperature-calibrated)

| Modality | N | Accuracy (raw) | Accuracy (calibrated) | ECE (raw) | ECE (calibrated) | Brier (cal.) | Coverage@5% |
|---|---|---|---|---|---|---|---|
| text | 1000 | {{TEXT_ACC_RAW}} | {{TEXT_ACC_CAL}} | {{TEXT_ECE_RAW}} | {{TEXT_ECE_CAL}} | {{TEXT_BRIER}} | {{TEXT_COV}} |
| screenshot | 1000 | | | | | | |
| image | 1000 | | | | | | |
| audio | 1000 | | | | | | |
| video | 1000 | | | | | | |
| crossmodal | 1000 | | | | | | |

Gates we hold ourselves to (pre-registered, per ADR0023): calibration must not cost >0.5pp accuracy anywhere; per-modality ECE target ≤ .03; misses are disclosed, never retuned silently.

## Public benchmark

| Benchmark | N | Ours | Frozen base (same size) | Notes |
|---|---|---|---|---|
| JevBench public decisions | 231 | {{JEVBENCH}} | {{JEVBENCH_BASE}} | two-pass exact-match; identical protocol |

## Reference lines (not comparable — published for context)

- 27B decision teacher, identical video protocol: 94.2% (we distill its *distributions*; we do not match its scale)
- Symbolic-execution systems on CLEVRER-class benchmarks: 96–99% (programs answer at inference; our model answers directly in one pass — a different product shape)

## Calibration deep-dive

Per-modality temperature values, reliability diagrams, and coverage@5% curves ship in [`calibration/`](./calibration/) — generated from the same sealed run, no post-hoc cherry-picking.
