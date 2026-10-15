# S4 existing-gold automatic gate: implementation assessment

Date: 2026-10-15. Code inspected at `32f3846db43000ff8999efd110fd384a7c968ec3`.
Scope: preparation for s0:05 under ADR0012; no implementation, real sealed run,
GPU action, or milestone acceptance. Only repository code, contracts and status
were inspected; no actual calibration/holdout payload or answer was read.

## Decision and current gaps

Add a separately versioned existing-gold gate rather than changing the meaning of
`sealed-cohort-gate-v2`. Keep `eval/sealed_gate.py`, `scripts/sealed_gate.py` and
their historical canary tests compatible. Their non-canary path still requires
two annotators, calculates κ, only models dev/holdout, and always blocks real
acceptance. Removing that block would not implement ADR0012.

The new contract must require all six evaluation cohorts (text, image, screenshot,
audio, video, crossmodal), each with 500 dev and 1,000 independent holdout items.
Define an item as a frozen question identity, with family counts also reported;
derived variants cannot inflate the quota. Freeze this counting rule in the
manifest before selection. Physical input modality does not identify screenshot
or crossmodal cohort. κ is explicitly `not_applicable`, never 1.0. There is no
new annotator identity, signature, κ threshold, or user-supplied account prerequisite.

`configs/gates.json` already has S4 `holdout_gate_pass` and
`holdout_gold_provenance_verified`; S5/S6 retain `holdout_gate_pass`. These numeric
flags must be derived from a verified receipt, not accepted as caller assertions.
`eval/gate.py:evaluate_gate` can compare final metrics but explicitly does not
attest data/model provenance. Its CLI errors are not a safe sealed-data interface.

## Concrete reuse and limits

| Existing surface | Reuse | Additional requirement |
| --- | --- | --- |
| `data/schema.py:Sample`, `Question`, `answer_tokens` | Strict task/target legality, source/revision/hash, usage, four pools, media SHA256 | A legal hard target with an arbitrary `source` string is not proof of gold; reject teacher-derived targets through source evidence |
| `data/schema.py:validate_pool_isolation` | Duplicate sample and same-family cross-pool rejection | Does not connect differently named families through media or historical use; its exception includes identity and must remain internal |
| `scripts/s1_text_pools.py:allocation_rows`, `scripts/s2_media_pools.py:allocation_rows` | Frozen pool, family, original/derived sample hash, source/revision/license | Existing allocation rows omit media hashes; cannot certify media isolation from these rows alone |
| Same modules' `canonical_family_hash` | Preserve already-hashed IDs and canonicalize raw family IDs | Validate raw/hash agreement when both exist; use explicit source alias mapping rather than double hashing or source revision to split a family |
| `scripts/s0_development_bundle.py:exclusions` | Union-find joining family nodes by every shared media hash, linked crossmodal components | Currently dev-only; generalize the algorithm to full four-pool metadata plus all historical dev/training exposures |
| `scripts/s0_crossmodal_dev.py` component metadata | Original image/audio family and media links | Component records also contain program truth/proposals: process inside job and export only allowlisted alias metadata |
| `scripts/vqa2_new_family.py:_exclusions` | Handles family_id, family_hash and singular/plural media SHA fields | Both aliases must be inspected, not `or` fallback; permissive missing fields are insufficient for final audit |
| `eval/sealed_gate.py` summary and `scripts/sealed_gate.py` error handling | Aggregate counts, fixed error categories, no exception repr or item IDs | New strict output schema, runner-failure coverage and no arbitrary strings copied from sealed inputs |
| `scripts/s1_dev_eval.py` inference/metric helpers | Existing real forward and metric semantics | Do not expose its CLI: it writes per-item `records.jsonl`, partial journals and tracebacks; all such output stays private or is disabled |

## Complete family/media isolation from metadata

1. Freeze a manifest of every input ledger/build report and its checksum. Include
   all actual training inputs across S1–S3 and augmentations/failed candidates,
   every historically inspected dev version, calibration reservations and
   holdout reservations. Current train files alone cannot erase past exposure.
   Match training-run input hashes to the inventory. Missing historical coverage
   blocks admission; an empty optional ledger is not proof of no exposure.
2. Inside the independent job, join allocation item identity and original/derived
   sample hashes with source manifests, media manifests and component ledgers.
   Existing `Sample.meta` and `prefix.media[*].sha256` supply missing metadata
   when a bound media-only ledger is unavailable. This extraction occurs in the
   job, never by printing or handing sealed payloads to the development agent.
   Check joins one-to-one where required and reject missing/ambiguous joins,
   digest mismatch or an incomplete media list. Aggregate reports export only
   counts/digests, not per-item ledger records.
3. Build one global graph, not one graph per cohort/source/pool. An item's
   aliases include every declared raw/canonical family, original family,
   explicit cross-source equivalence, crossmodal component family, and every
   source/derived media content hash. Join all aliases transitively. Preserve
   source provenance without letting a dataset name or revision hide shared
   original identity. Do not treat a whole-source archive hash as a family/media
   identity: it would incorrectly merge the entire dataset.
4. Assign pool memberships to connected components. Reject any component in
   more than one of train/dev/calibration/holdout; treat historical dev and
   training exposures as persistent memberships even if absent from today's
   selected files. Also reject inconsistent item identity, cross-cohort duplicate
   counting, and a crossmodal item whose source components cannot resolve.
   All image/audio members of a crossmodal item must belong to its pool.
5. Exact media SHA equality catches copies, not re-encoding, cropping, video
   clips or audio segments. Preserve source-original family and parent/derived
   media links for these transformations. If current manifests omit such
   lineage, explicitly block that source pending metadata reconstruction;
   never report “all aliases checked” merely because the available SHA sets
   are disjoint. No perceptual matching dependency is required for this slice.

The graph enforces all *declared and verified* aliases. Existing metadata can
seed it, but completeness is a separately checked provenance condition; this
assessment does not claim that the current reserves satisfy that condition.

## Minimal implementation and tests

1. Add `eval/gold_gate.py`: strict versioned manifest, source-policy references,
   metadata normalization/connected-component audit, validated gold admission,
   and aggregate receipt models. Reuse schema and SHA helpers; keep public v2
   models unchanged. A source-policy entry binds source/revision/license/usage,
   raw-label artifact hash, mapping/rule code hash and target origin. Dataset gold
   must match the pinned source mapping; program truth must be recomputed by the
   pinned deterministic rule. Missing, soft teacher, forged origin or illegal
   labels fail. Research-only sources stay in their permitted evaluation scope
   and cannot silently enter commercial training.
2. Add `scripts/gold_gate.py`: private job entrypoint with separate data-audit
   and stage-evaluation modes. Audit can prepare s0:05; actual sealed inference
   is admitted only for S4/S5/S6. Bind model revision/adapter hash, tokenizer and
   chat-template/answer mapping, configuration, seed, code version, dataset and
   ledger digests, runtime/hardware, elapsed time, cost and exit code. Separate
   data validity, execution completion and stage quality outcomes; missing
   inputs/receipts are blocked, known invalid evidence fails, neither passes.
3. Add `tests/test_gold_gate.py` and `tests/test_gold_gate_cli.py`, using synthetic
   fixtures only. Cover six-cohort quotas and counting, missing labels, invalid
   targets, forged gold/source-rule hashes, unsupported licensing, teacher labels,
   every pair of four pools, raw/hash disagreement, multi-hop alias bridges,
   cross-source media copies, crossmodal parent links, missing media lineage,
   historical trained/dev items removed from current files, duplicate items and
   incomplete ledgers. Mutation of each bound artifact must invalidate receipt.
4. CLI black-box tests capture stdout/stderr, report and failure artifacts for
   invalid JSON, Pydantic errors, filesystem errors, backend exceptions, timeout
   and interrupted/partial inference. Insert synthetic secret sentinels into
   labels, IDs, paths and nested backend errors; none may reach exported surfaces.
   Test unknown output fields rejected, immutable outputs and no success receipt
   on partial run. Retain and run `tests/test_sealed_gate.py` unchanged as legacy
   compatibility evidence.
5. Update `docs/runbooks/sealed-gate.md` only after executable behavior exists;
   clearly separate new first-release contract from historical v2 instructions.
   Add a narrow stage-receipt verifier/assembler when integrating S4, rather than
   broadening `eval.gate` into an execution/security service. Tests must reject
   canary receipts, mismatched model/data/config/stage, stale/replayed runs and
   arbitrary caller-supplied success flags. Run targeted tests, lint/typecheck
   for changed modules, then milestone checks. No new dependency is needed.

## Independent job, root host and evidence boundary

Run a pinned, noninteractive job with private inputs and private logs/artifacts.
Only a strict allowlisted aggregate receipt is copied out. Do not expose arbitrary
shell, file-read, query/filter, per-item diagnostics, arbitrary output paths or
model callbacks to the receipt consumer. Frozen source, model/config inputs and
trigger budget prevent turning repeated aggregate queries into holdout tuning.
Keep backend stdout/stderr and partial records private; the outer supervisor
exports fixed failure categories and exit status even if the worker crashes.

Test ordinary development-role denial of input mounts, private logs, sidecars,
temporary files and other runs' artifacts, including symlink/path traversal and
failure paths. Bind the tested launcher/permissions/config hashes to the receipt;
a self-declared `isolated=true` or an unverified signature is insufficient.

A separate process/container or account on the root-access GPU host supports a
workflow boundary and negative tests for ordinary identities. It does **not**
prove security against that host's root user, container runtime administrator or
the developer with root SSH. Report this explicitly as root-visible workflow
separation, not external access isolation. ADR0012 removes the requirement for
the user to supply an external gate account; do not restore that prerequisite.
If a stronger non-root trust boundary is required by deployment policy, an
independently administered host/identity and actual access tests are additional
evidence, not something chmod or the data auditor can attest. Any unavailable
required isolation evidence remains pending rather than being fabricated.

## Rejected shortcuts and exit condition

- Do not disable the v2 non-canary block or fake two annotations/κ to obtain green.
- Do not count dev as holdout, select holdout by candidate scores, reduce the
  six-cohort quotas, or move previously trained families after the fact.
- Do not infer reliable gold from a label-origin enum, a source string or a
  known-looking license string alone; verify pinned source policy and mapping.
- Do not export an unrestricted runner report and redact selected keys afterward;
  use an explicit receipt schema and private error boundary.
- Do not infer readiness from reservations, canary tests, hashes without source
  binding, a successful CPU pipeline or data-audit pass without model evaluation.

Implementation is ready to start after S3's integration decision and authorized
S4 scope/budget are known. s0:05 closes only with actual source-bound counts,
complete four-pool/historical alias audit, independently executed private-data
job, observed permission/log/error rejection evidence, and reproducible aggregate
receipts. This report establishes a bounded implementation plan, not that exit.

## Assessment validation

Inspected the code symbols and contracts named above, ADR0012, s0:05,
`plan/STATUS.md`, `docs/02-project.md` and `CONTEXT.md` at the stated revision.
Documentation-only deliverable: no production or GPU tests executed, no sealed
data certification claimed. Final diff whitespace validation is recorded in the
commit; implementation tests listed above are proposed, not passing evidence.
