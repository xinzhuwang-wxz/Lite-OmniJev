# Public video CPU input proof

Two real CPU runs used the same public DEV video sample and official processor assets: candidate sourcefe58854 and the historical public evaluator source74e96e2 (archive SHA `b8c5b3184ea2d09e6bb1184c7065c27fd6d354022dcf0e37169e6e390f1d6516`). Each prepared image-budget280 and1120 through the baseline/BudgetProcessor path, then the official DEV prepare_media_example path with16frames. No GPU or model weights were loaded; no private samples were opened.

All input tensor shapes, dtypes and byte hashes, prompt SHA and answer-token IDs are identical across both budgets and both source versions. Actual video processor configuration SHA is `752904c106c4e3ba2648b8b0b5546f990e96298f1221bcef479eba247232c3e7`. The original DEV dataset SHA is `79ed30be3a687a6c767c4d7c55c24dc7ac08d5a1698a6f49e3bc7d47e74e63d2`.

The historical and candidate load_media source functions differ (new explicit-frame support), so source inspection alone was insufficient. The second actual CPU run verified that their default16-frame paths produce identical tensors on the same public sample. The combined, immutable proof is `/root/autodl-tmp/s1-text-lora-001/reports/s5-video-budget-source-input-parity-001.json`, SHA `11547bde626f12941c0e73428b54ddeba39fcbe36b8aff83901336ba2c462350`; its contents include both reports and their source/tensor hashes.

This is one public-input preprocessing proof plus exact source/config binding. It is not500-row replay, probability parity, GPU dtype-cast parity or a stage-quality pass. No old report was edited. A first invocation with an incorrect supplied dataset digest rejected before preprocessing; the corrected invocation used the selected manifest's exact digest.

The holdout evaluator now treats image budget as not applicable to pure text/audio. Image/screenshot/crossmodal still require1120. Video with another image-processor profile remains pending unless the explicitly SHA-bound proof validates: same original DEV hash, model processor config, historical archive, current preprocessing source hashes,16frames and identical tensors. The actual nested DEV runtime report is separately bound; wrong base,8frames or float32 remain rejected from gap calculations.

Metadata preparation must pass both:

- `--development-runtime video=/root/autodl-tmp/s1-text-lora-001/artifacts/s4-program-stratified-video500-step0100/report.json`
- `--video-input-parity /root/autodl-tmp/s1-text-lora-001/reports/s5-video-budget-source-input-parity-001.json`

The calibration metadata helper also now accepts explicit paired `--inventory/--inventory-sha256` and `--inventory-receipt/--inventory-receipt-sha256` overrides, allowing inventory029 to be pinned honestly after the current E2B exposure is recorded. New JobSpec003 remains unprepared until root supplies that current inventory and a new immutable source revision. No training/calibration UID is leased by these helpers.
