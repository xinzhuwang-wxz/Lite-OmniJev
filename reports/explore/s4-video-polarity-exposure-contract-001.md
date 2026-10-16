# Single-arm polarity exposure contract

CPU implementation only; no full inventory audit was run while control was updating. New `--polarity-arm control|polarity` extends the existing reconcile entry without changing default paths/behavior. Control binds settled009; polarity requires settled010 carrying `polarity_arm=control`. Every previous bounded pilot remains checked. All base input bindings carry forward to the final stability check; old indexed input and update-report discovery remain unchanged.

The selected arm must have terminal final/runtime training receipts and terminal launcher (including DEV completion/failure). `failed` with steps>0 is accepted as a real update-bearing exposure; no adapter final file is required for failed training. Completed10 requires the actual final adapter hash. Live training or launcher is rejected. Full runtime Sample objects must equal prepared objects; actual data hash is arm-specific. Selected f2db init must equal runtime/final/config metadata. Frozen prep outputs, isolated audit, explicit120-row TRAIN allocation ledger canonical hashes, full config, fixed training argv/timeouts, launcher source and all its bound inputs, and launcher actual-training-receipt hash are verified.

For an update-bearing arm, all prior TRAIN current inventory artifacts become historical before appending this arm's canonical120 input as the sole TRAIN current artifact. This prevents duplicate shared replay current IDs. This historical inventory classification does not claim every source row was drawn: actual completed-step count is separately retained. Other pools are unchanged. Allocation ledger is SHA-bound in the reconciliation receipt, not falsely encoded as a crossmodal component ledger. New video data_version→video mapping is taken from the selected arm's bound preparation manifest. Existing report/data-hash deduplication and complete aggregate inventory audit are reused.

After root confirms control launcher terminal, run once into new010:

```sh
PYTHONPATH=/root/autodl-tmp/s4-claim-pair-code-001 \
 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python \
 /root/autodl-tmp/s4-exposure-polarity-code-001/reconcile.py \
 --root /root/autodl-tmp/s1-text-lora-001 \
 --legacy-root /root/autodl-tmp/lite-omnijev \
 --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-010 \
 --polarity-arm control
```

This command is prepared for a separate immutable deployment; deployment/execution is not yet claimed. After root reviews control and explicitly runs polarity, its terminal audit uses `--polarity-arm polarity --output-dir .../s4-pilot-exposure-011`;010 is then mandatory. Neither command trains, schedules another arm or changes admission. Overwrite remains rejected.

22 targeted tests (new polarity contracts plus existing coverage branch),mypy,ruff pass. No GPU, source research, sealed content, main STATUS/ticket or complete inventory audit was performed in this CPU slice. Live terminal artifact execution remains pending root notification.
