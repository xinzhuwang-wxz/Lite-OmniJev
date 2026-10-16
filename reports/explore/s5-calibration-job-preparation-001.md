# Frozen candidate calibration metadata preparation

Actual remote CPU preparation completed in11.474 seconds, exit0, against immutable source `9cc3f3a168fe11eb20dbc7c1001c3eff3c7d74ab`. This is metadata preparation, not calibration inference, isolation activation or a quality pass. The preparer never opened private sample/logit/log contents; private dataset checks were stat plus independently audited inventory bindings.

- Job: `/root/autodl-tmp/s1-text-lora-001/artifacts/s5-calibration-job-001/job.json`
- Job SHA256: `69456c8df2f58e056196b8682a643ad8dc77640f45537d515a84a5888dbdddd4`
- Preparation report: same directory, `preparation-report.json`
- Report SHA256: `d820cf54aeb2506a6a7ca87a42ad21ad3fcdefb08cc0220bf0438f28b24b51dd`
- Candidate adapter SHA256: `e46550143b28f8798484515f0abe21996fe92ad241625a4cf6da15dc093277fd`
- Producer file SHA256: `3228707a1e4f77416659aa849dabd919ad748d8f2bd9c922bcf587bf561d0808`
- Actual scripts/train/eval/data code bundle SHA256: `786b44ee6bbc044c4625212877796f97a47a2594878caec408c971581601610c`

All seven source specs strict-parse and pass existing `verify_catalog`; public model and adapter bytes match their bindings. The selected manifest/admission are unchanged. Seven per-source admission metadata declarations refer to those exact existing bindings and are not new quality claims. The028 inventory receipt was produced by the prior unchanged broker: the producer checks its observed isolated boundary and exact inventory hash, not equality with the producer revision. No revision was rewritten.

Audio is already split in the selected manifest: `crossmodal-source-bundle-001/samples-0000.jsonl` contains the declared1400 Clotho records; `samples-0002.jsonl` contains1600 LibriSpeech records. No private inspection or splitting occurred. The complete job declares text/image/screenshot choice3000 each, audio choice1600+noul1400, and video/crossmodal noul3000 each. The worker still verifies actual counts and groups before inference.

Private media metadata confirmed0700 directories and0600 assets. Commitbc16ac9 adds hash-bound private-media snapshots with unique-byte/statvfs budget checks; model/runtime stay public-read-only in place. Original files are never chmodded. A focused real Linux probe is prepared with `--case-filter private_snapshot`; root owns its execution and review. Only worker-declared calibration media are copied, never a holdout directory scan/copy. The earlier directory byte totals are stat-only capacity context and include both pools.

Root launch command, after focused boundary evidence passes and within the root-controlled GPU schedule:

```sh
cd /root/autodl-tmp/s5-calibration-code-9cc3f3a
PYTHONPATH="$PWD" /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python -m scripts.private_gold_job \
  --job calibration-fit \
  --spec /root/autodl-tmp/s1-text-lora-001/artifacts/s5-calibration-job-001/job.json \
  --sha256 69456c8df2f58e056196b8682a643ad8dc77640f45537d515a84a5888dbdddd4 \
  --work-root /var/tmp/omnijev-calibration-jobs-001 \
  --report /root/autodl-tmp/s1-text-lora-001/reports/s5-calibration-fit-001.json \
  --worker-uid 64201 --probe-uid 64202 --timeout 14400
```

The new work-root was created and verified root-owned0755 during preparation; it contains no leased calibration UID or job. UIDs64201/64202 were unused at preparation time; broker checks again and permanently leases them. The14400-second value is the prepared fit-child upper bound, not permission to exceed the root's remaining resource budget; it excludes metadata/snapshot phases. Root must apply its overall job cap. This slice did not start GPU work or lease calibration UIDs.

Local verification:20 focused CPU tests pass; Ruff/mypy pass for producer/helper/probe. Actual private calibration, actual copied-media size, full model inference and final temperatures remain unmeasured.
