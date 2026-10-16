# Private calibration setup repair review

Reviewed commit `09ba1f68069072a2930ded10b598fac2c295c4fd`. No specific cal-fit setup blocker found in this delta. This is source/CPU review, not an activation or fit-success receipt.

The base-view helper verifies every source SHA before creation. Large files must already be root-owned, world-readable, not group/other writable and on the same filesystem; hardlinks are never chmod/chowned. Small metadata is copied independently as root0444. Both target hashes and public ancestor permissions are checked. Adapter assets always use the bound private snapshot. Fixed failure categories and phases do not stringify or export private exception text.

Both prior holdout findings are resolved in source: `temperatures` requires and snapshots the actual raw receipt, binds its hash to the completed fit and all seven temperature bindings, checks model/adapter/template/token map and spec/code identity, and requires Transformers/BF16/native-LoRA raw complete legal logits. `development_metrics` checks actual runtime evidence; mismatches cannot justify a gap pass and produce `pending_runtime_or_missing_evidence`. The 280/1120 image-budget video equivalence remains a separate evidence task; this review infers no actual quality difference.

Narrow synthetic CPU verification: **16 passed, 32 deselected**, exit0. Exact source hashes and working-copy match checks are in the adjacent JSON. No private data/logs, ACL changes, UID launches or GPU were used. New immutable source/spec Linux negative probes and actual calibration remain pending; no holdout/stage acceptance is asserted.
