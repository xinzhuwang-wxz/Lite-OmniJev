# Fixed private holdout worker: CPU first version

The new `private-holdout-job-v1` input binds seven complete holdout files: text/image/screenshot/video/crossmodal1000 each and audio choice500+noul500. These are the selected manifest's seven actual file boundaries; no DEV substitution or private splitting is allowed. The trusted worker verifies all6000 records, pool, exact groups, hard gold, unique item IDs and media cohorts. Current independent inventory receipt binds family isolation; it does not create a new gold or IID claim.

The only new broker job name is `holdout-eval`. It selects an internal Literal branch and the fixed `scripts.private_holdout` module. It shares existing UID leases, ACL guardian, recovery journal, negative permission probes, read-only snapshots, byte-budgeted private-media copy, no-new-privileges/capability verification, bounded timeout, progress relay and revocation. No arbitrary module or command surface exists. Calibration remains the default branch; its existing export checks were moved intact into a helper. Per-source admission data/pool/cohort/count is checked explicitly in both paths.

The worker extracts actual complete legal+U Transformers BF16 logits through the existing frozen native-LoRA backend. It computes conditional-answer accuracy/Brier/ECE15/coverage, joint legal+U NLL and mean p(U), per populated primitive group and per cohort. Raw results are always present. Optional calibration requires all seven actual SHA/Binding-bound temperature artifacts and their completed independent calibration receipt; it never fits or changes T on holdout. The eleven absent groups remain absent, not invented.

Only `receipt.json` aggregate data is exported through a fixed allowlist. The supervisor rechecks source/model/adapter/template/global33 map, exact group counts, numeric metric schemas, data/inventory/gates bindings, and recomputes quality observations/deltas. IDs, text, rows, logits, arbitrary nested fields, logs and worker-selected files are not exported. Runtime recipe remains truthfully Transformers/bfloat16/native_lora, not vLLM parity. Every100/cohort-end aggregate progress uses the existing strict public progress schema.

SHA-bound full500 public DEV receipts are optional metadata. When present, their adapter, selected-original data SHA and complete count must match. Missing cohorts produce an explicit failed/missing gap observation. Original S4 gap≤.03 and S5 ECE≤.03/coverage≥.65/accuracy-regression≤.005 remain unchanged; altered thresholds are rejected. The coverage observation uses the minimum six-cohort coverage and is labelled diagnostic. Source distribution differences are reported. Passing these numeric observations never sets stage acceptance: teacher ratios, DecisionIndex/JevBench, permutation/noul consistency, R1 and remaining stage evidence stay explicit.

## Deployment and preparation

Root must review the new high-risk source, deploy a new immutable projection containing `scripts/private_holdout.py`, the changed broker and lifecycle helper, and rebind producer revision/code hashes. Do not mutate the currently deployed calibration projection. Existing active calibration jobs keep their pinned source.

Metadata-only helper:

```sh
PYTHONPATH=NEW_SOURCE PYTHON .scratch/s5/prepare-private-holdout-job.py \
  --calibration-job ACTUAL_CAL_JOB --calibration-job-sha256 ACTUAL_SHA \
  --producer-revision NEW_FULL_REVISION --output NEW_HOLDOUT_METADATA_DIR \
  --gates NEW_SOURCE/configs/gates.json \
  --development text=FULL500_AGGREGATE_REPORT
```

Repeat `--development cohort=report` for the other five frozen original DEV cohorts. Add `--calibration-receipt ACTUAL_COMPLETED_FIT_REPORT` only after real fit completion. Without it the holdout job is raw-only. This helper reads public metadata and temperature artifacts only, never holdout rows, logits or logs.

Root execution after review and GPU scheduling:

```sh
PYTHONPATH=NEW_SOURCE PYTHON -m scripts.private_gold_job \
  --job holdout-eval --spec NEW_JOB --sha256 NEW_JOB_SHA \
  --work-root ROOT_OWNED_0755_PRIVATE_WORK_ROOT --report NEW_AGGREGATE_REPORT \
  --worker-uid 64601 --probe-uid 64602 --timeout ROOT_BOUNDED_SECONDS
```

64601/64602 are suggestions only; fresh lease/process/account checks run immediately before launch. No metadata helper, GPU execution, ACL mutation or private-data read was performed for this first version. CPU tests include a full synthetic6000-record worker→strict aggregate-export path, pool contamination rejection, incomplete T rejection, raw/T metric behavior, immutable threshold checks and unchanged default calibration dispatch. These tests do not establish actual Linux or model quality acceptance.
