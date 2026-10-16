# Video-specialist CPU preparation

Feasible without trainer changes: `train_smoke` selects `examples[order[(step*accumulation+offset)%len(examples)]]` after one seed42shuffle.60stored unique native video samples therefore produce120actual video draws in10steps with accumulation12, exactly two exposures per sample. Each gold CE is divided by12before backward; one gradient clip (finite/nonzero guarded) and AdamW update occurs per complete step. The optimizer persists through all10steps; existing checkpoint machinery saves optimizer tensors and Python/Torch/CUDA RNG together with data/config/order binding. This proposal does not alter that machinery.

The helper binds the previously prepared native official60video questions and targets byte-for-byte through canonical samples, source receipt401fb4248c079c37e25cff1827b9470b3f3105ee8a3182997aee3c4de0fc7696 and exact allocation ledger. Only sample_count60, modalityvideo, and single-version declaration differ from the native mixed CE config. LR/rank/seed/template/16frames/steps/accumulation remain unchanged. Storage has60unique IDs; `draw-recipe.json` records120actual draws with repeated canonical sample/family bindings rather than inventing120families. No replay rows are included.

This changes both video exposure2x and replay removal, so it tests a sampling recipe rather than establishing a single-factor causal explanation. The result can only be a video-specialist research candidate. Production remains the selected single-LoRA candidate until independent retention and quality evidence establishes otherwise. No old modality gate is inherited. Parent is null, training_authorizedfalse; GPU dispatch and the unique10step/240second budget belong to the main task.

Two targeted CPU tests verify exact two-pass exposure, TRAIN-only scope and minimal config differences. Ruff check and format passed. Full CPU dataset/inventory preparation is pending the current016exposure receipt;015was explicitly not reused after native training. This document does not claim data readiness before that audit. No GPU or new dependency was used. Cost null.

```sh
CUDA_VISIBLE_DEVICES= PYTHONPATH=. timeout 120 python .scratch/s4/prepare-video-specialist-pilot.py --root /root/autodl-tmp/s1-text-lora-001 --output-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-video-specialist-pilot-001 --exposure-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s4-pilot-exposure-016 --exposure-report-sha256 ACTUAL_COMPLETED_016_SHA
```

The command is implemented and CPU-tested; real artifact execution awaits the fixed completed inventory. Output creation is exclusive and rejection leaves a failure receipt. The helper rejects incomplete/changed inventory, active update jobs and isolation failures.
