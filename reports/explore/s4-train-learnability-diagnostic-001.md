# Fixed12TRAIN learnability result

CPU paired verification completed; `s4-train-learnability-comparison-001.json`
binds before/after receipts, records and completed10step training report.
Canonical sample/gold/mapping and complete prepared-input hashes match all12.

- Correct10→12/12:2corrected,0regressed.
- Mean joint gold CE .433525→.005090:98.8259% reduction.
- Mean gold-vs-opposite margin1.375→5.895833:+4.520833.
- Completed10steps; adapter changed from CE0 `4b8e422f…` to `02a579d5…`.
  Gradient evidence is successful passage through trainer nonfinite/nonzero guards,
  not an asserted gradient-norm history.

Frozen classification: **selected_train_examples_learnable**, strong selected-set
fit. This demonstrates that the current pipeline can learn these12examples; it
neither proves generalization nor identifies the backbone's full capability.

Prefer broader audited TRAIN-family coverage with bounded disjoint segments and
existing modality replay over another loss-weight or frame-policy sweep. S3's
actual video exposure was200families/draws, not the whole8299admitted TRAIN pool
(`s3-training-exposure-audit-001.json`). Do not promote this memorization adapter.
Root owns the already-running fixedDEV50 check; its result must be assessed
separately before any claim of generalized benefit. No new GPU job ran here.
