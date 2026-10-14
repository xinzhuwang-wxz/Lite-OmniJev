# S0 Teacher Benchmark Runner

This runner connects the fixed public S0 benchmark inputs to the project-owned
Jev-Omni teacher forward adapter. It does not replace the GPU evidence: CPU
preflight and fake-forward reports only prove mapping, media preparation,
resume, comparison, and denominator behavior.

Inputs:

- MMAU test-mini: fixed 1000 public rows from `configs/video-eval-assets.json`.
- MVBench ADR-0006 public baseline: fixed 9 tasks / 1800 rows from the canonical
  PNG manifest produced by `reports/public-video-protocol-freeze-001.json`.
- Teacher adapter: `train.teacher_inference.load_teacher_forward_bundle` and
  `predict_prepared`; the runner does not import upstream `jev_omni.py`.

CPU preflight:

```bash
PYTHONPATH="$PWD:/Users/bamboo/.codex/worktrees/s0-multimodal-cpu/Lite-OmniJev/artifacts/mac-media-libs" \
artifacts/public-eval-media-env/bin/python -m scripts.s0_teacher_eval \
  --mode cpu-preflight \
  --benchmark all \
  --video-cache-root artifacts/s0-teacher-eval/video-cache \
  --sample-mmau-audio 3 \
  --sample-video 1 \
  --output reports/s0-teacher-eval-cpu-preflight-001.json
```

This validates all MMAU/MVBench row identities and fixed denominators, then
actually prepares selected public audio through the teacher PCM adapter and
selected canonical PNG frames through the processor call shape. The report
records hashes, counts, summaries, and boundary flags only; it must not contain
question text, options, answers, labels, or sealed data.

Fake forward control-flow check:

```bash
PYTHONPATH="$PWD:/Users/bamboo/.codex/worktrees/s0-multimodal-cpu/Lite-OmniJev/artifacts/mac-media-libs" \
artifacts/public-eval-media-env/bin/python -m scripts.s0_teacher_eval \
  --mode fake-forward \
  --benchmark all \
  --limit-mmau 2 \
  --limit-mvbench 1 \
  --video-cache-root artifacts/s0-teacher-eval/video-cache \
  --output reports/s0-teacher-eval-fakeforward-001.json \
  --records-path artifacts/s0-teacher-eval/fakeforward-001.jsonl
```

Fake forward uses deterministic synthetic probabilities. It proves two-pass
recording, exact comparison, resume binding, probability normalization, and
scoring with fixed denominators, but it is not model quality evidence.

GPU run after explicit card approval:

```bash
uv run python -m scripts.s0_teacher_eval \
  --mode real-forward \
  --benchmark all \
  --video-cache-root /root/autodl-tmp/lite-omnijev/artifacts/public-video-protocol-freeze-001/canonical-mac \
  --output reports/s0-teacher-eval-gpu-001.json \
  --records-path artifacts/s0-teacher-eval/gpu-001.jsonl
```

The real run loads the teacher only through
`load_teacher_forward_bundle(...)`, which remains CUDA-gated and verifies the
pinned local model assets and converted safetensors head before `from_pretrained`.
Reports refuse overwrite by default; use `--resume` only for the same bound row
set and code/config/media hashes. Any missing media, non-finite probability,
row failure, or two-pass mismatch leaves the fixed denominator intact and fails
the batch.
