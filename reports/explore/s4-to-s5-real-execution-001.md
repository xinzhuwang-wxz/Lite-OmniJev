# Measured S4 → S5 execution readiness

This is CPU preparation, not a calibration run or stage pass. No real calibration/holdout rows or private logs were opened. ADR0023 permits progression with measured quality limitations; it does not turn unmet quality gates into passes. Freeze the selected checkpoint and complete the final exposure inventory before launching private work.

## Existing execution paths

The existing `scripts.s1_dev_eval` handles public full-500 text/image/screenshot/audio/video with an explicitly bound adapter and official media/template preprocessing. Reuse the selected-manifest original DEV datasets, actual model asset manifest, image budget1120/video16, and frozen adapter; do not substitute a similarly named unnormalized audio dataset. Crossmodal full500 needs the existing shared evaluator's public full-dev/adapter path completed: `s1_dev_eval` is single-media; `private_stage_eval` engineering mode caps50 and stage modes forbid DEV. No extra mirror wrapper is proposed.

Existing `scripts.private_stage_eval` validates selected-manifest/admission/evidence and can run complete holdout summaries, but runs its trusted worker as root. It does not provide an independent UID boundary. The existing fresh-UID broker previously only ran inventory audits. Production private holdout execution must reuse the reviewed UID boundary, rather than claim the root subprocess is isolated. Holdout remains measurement only; never fit temperature on holdout.

The selected manifest is `.scratch/s4/selected-gold-manifest-001.json`, SHA `e073a601eab235644f91f058a99dd331c568de2560660441192778553429876f`. Global admission is `reports/s4-gold-data-admission-001.json`, SHA `fd5a42f6a311a2b6cb5d91cf473c46ee532c65b6abef13cd463fd118e7713c7a`. These historical bindings must be refreshed if the selected evidence or exposure changes; the producer also requires the current independently audited inventory.

## Real calibration coverage

| Cohort | Choice | Score | Noul |
|---|---:|---:|---:|
| text | 3000 | absent | absent |
| image | 3000 | absent | absent |
| screenshot | 3000 | absent | absent |
| audio | 1600 | absent | 1400 |
| video | absent | absent | 3000 |
| crossmodal | absent | absent | 3000 |

These counts come from public manifests and source-builder contracts. The worker must verify them against all actual18000 calibration records; CPU fixtures do not prove the real counts. Each modality totals3000. The eleven absent groups remain unfitted. No synthetic T, relabeled group, pooled substitute, or inherited metric is acceptable.

## Implemented minimal producer seam

`scripts.private_gold_job --job calibration-fit --spec SPEC --sha256 SPEC_SHA --work-root /var/tmp/omnijev-gold-jobs-001 --report REPORT --worker-uid FRESH_UID --probe-uid OTHER_FRESH_UID --timeout SECONDS` dispatches only the fixed `scripts.private_calibration` worker. `JobSpec` contains seven full S5/calibration `private_stage_eval.Spec` bindings, one frozen model/adapter, actual producer revision, pinned runtime binary, current inventory and its independent audit receipt. Exact source artifact schemas remain existing Pydantic contracts.

The supervisor checks metadata/admission and copies private calibration bytes without parsing rows. The worker validates all rows, emits a private asset-copy plan, then runs the frozen official model and extracts actual pre-softmax legal-answer+U logits. It fits existing `train.calibration.fit_temperature` independently for seven groups. Only seven temperature artifacts, aggregate raw-logit provenance and aggregate results are exported. Raw inputs/logits/logs remain in the revoked private run. Full33 answer-map SHA uses `sha256(json.dumps(mapping, sort_keys=True).encode())`; exact official template bytes use `check_tokens.probe`. These are independently checked at export, along with model/config/adapter/data/source binding. Historical baseline log-softmax records are never relabeled raw logits.

## Runtime access and review boundary

The remote overlay has approximately18GB free; copying 15GB model plus8.5GB environment is unsuitable. Optional `root_traverse_acl: true` grants only the fresh worker UID execute/traverse on `/root` via installed libacl. It requires the original base0700 ACL, refuses existing extended ACLs, saves the exact private recovery ACL, and restores it on success, exceptions and SIGTERM. Public read-only model/media assets are referenced in place, with content binding and write-denial probes; private data remains copied under the job. No world permission change is made. Runtime leaf symlinks preserve venv discovery but their resolved binary must match the pinned supervisor binary hash.

Before real activation, root must independently review the boundary and run actual Linux failure/permission probes: worker cannot list `/root`, cannot read ANY inventory calibration/holdout original, cannot write model/source; unrelated UID cannot read private snapshots/output. A global flock serializes grants. An independent root guardian revokes the fresh UID processes and restores the ACL on supervisor pipe EOF, including SIGKILL. A fsynced fixed recovery journal is checked/restored under the lock at next startup after host loss; unexpected ACL changes fail closed. The exact private per-run recovery record remains available. Actual worker bootstrap sets no-new-privileges and verifies zero effective/permitted/ambient capabilities. No namespace/network/root-adversary isolation is claimed. ACL tests here use a fake libacl adapter; no real ACL mutation was performed.

## Remaining product integration

The current gateway registry insists on18 groups and only routes15 (no trusted screenshot distinction). A separate narrow gateway slice must accept explicitly missing groups, preserve raw responses, and fail clearly on require-calibrated for unfitted groups. It must not infer screenshot from arbitrary caller input. Root owns actual private execution, final holdout measurements, serving activation and milestone review. No HF publishing or new downloads are part of this slice.
