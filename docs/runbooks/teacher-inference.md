# Jev-Omni Teacher Preflight

This runbook covers the CPU-safe teacher source/head/input-preparation boundary. It does not run a teacher forward pass, load the 24GB backbone, download from Hugging Face, or import the upstream `jev_omni.py` module.

The adapter keeps the upstream decision prompt text, processor call shape, decoder-capture/head semantics, and `2-256` option range. Audio must pass through `prepared_teacher_audio`; callers must not use the upstream `predict` audio path because it can overwrite an empty temporary WAV and fail before the processor consumes it. After that WAV is validated, the adapter reads the same mono 16kHz PCM16 samples into a float32 numpy waveform for the official processor, avoiding the path-based `audio_utils.load_audio` branch that would otherwise import heavier decoding stacks.

Run the CPU preflight with the converted safetensors head:

```bash
uv run python -m scripts.teacher_preflight \
  --source-dir artifacts/quarantine/teacher/5addda86ddee081a68fb067477ea100c221b8917 \
  --head artifacts/teacher/jev-omni/5addda86ddee081a68fb067477ea100c221b8917/head.safetensors \
  --head-spec configs/teacher-head-jev-omni-spec.json \
  --head-conversion-report reports/teacher-head-conversion.json \
  --synthetic-modality audio \
  --report reports/teacher-preflight-cpu-001.json
```

Expected pass evidence:

- `jev_omni.py` SHA256 is `11d761b0b6cefc8aac29757f43af4b2b02af8f9b6c6dad19834c0b14538180f0`.
- `decision_config.json` has `hidden_size=3840`, `output_classes=256`, and `dtype=float32`.
- `head.safetensors` is loaded with `safetensors.torch.load_file`, matches `configs/teacher-head-jev-omni-spec.json`, has finite tensors, and never uses `torch.load`. The expected head SHA defaults to the passing `reports/teacher-head-conversion.json` output SHA; unanchored head checks are invalid.
- `AutoProcessor.from_pretrained` uses `local_files_only=True` and `trust_remote_code=False`.
- One synthetic input is built through `processor.apply_chat_template(..., tokenize=True, return_tensors="pt", enable_thinking=False)`.

The report intentionally records tensor keys, shapes, dtypes, hashes, command environment, and boundary flags only. It must not include prompts, token IDs, sample text, private URLs, credentials, or hidden-set content.

Forward integration code is now present but real teacher execution remains gated. `scripts.teacher_forward` performs source/head readiness and refuses before `from_pretrained` when CUDA is unavailable. The loader admits only the official `Gemma4UnifiedForConditionalGeneration` class with `local_files_only=True`, `trust_remote_code=False`, and `use_safetensors=True`; it uses the converted safetensors decision head and never imports `jev_omni.py` or loads `head.pt`. CPU tests use tiny fake Torch modules to prove decoder `last_hidden_state[:, -1].float()` capture, `_Head256` masking/softmax semantics, option range handling, and hook cleanup. Real 24GB backbone loading and teacher probability evidence remain pending until an explicit CUDA run after card approval.

Run the CUDA-gated readiness check without loading the backbone:

```bash
uv run python -m scripts.teacher_forward \
  --source-dir artifacts/quarantine/teacher/5addda86ddee081a68fb067477ea100c221b8917 \
  --head artifacts/teacher/jev-omni/5addda86ddee081a68fb067477ea100c221b8917/head.safetensors \
  --head-spec configs/teacher-head-jev-omni-spec.json \
  --head-conversion-report reports/teacher-head-conversion.json \
  --report reports/teacher-forward-cpu-004.json
```

On CPU-only hosts this report should be `blocked` with `loader_stage=pre_from_pretrained`, `model_backbone_loaded=false`, `model_forward_run=false`, and an error stating that CUDA is required before `from_pretrained`.

Video uses the pinned upstream round-and-deduplicate frame sampling formula for same-condition preflight. If that formula yields fewer unique frame indices than requested, the adapter rejects the input before decoding instead of silently reducing the frame count; changing the sampling formula needs a separate protocol decision.
