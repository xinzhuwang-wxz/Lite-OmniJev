# S0 native vLLM processor CPU check

This CLI compares the installed primary vLLM 0.26 processor with the frozen canonical HTTP HF reference (0.31 remains diagnostic only). It creates `ModelConfig` and the official `MULTIMODAL_REGISTRY` processor only: no `LLM`, engine, backbone constructor, weights, generation or CUDA forward. A processor pass is not service, raw-probability, quality or release acceptance.

Run against the complete existing local checkpoint metadata and the public fixture media. Disable network access through the HF offline settings; use the shared heavy-CPU lock and one thread on the 2 GiB / 0.5 CPU host:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
flock /tmp/omnijev-cpu-heavy.lock \
/root/autodl-tmp/envs/serve-next-vllm031-offline-check/bin/python \
  -m scripts.s0_vllm_processor \
  --manifest tests/fixtures/s0-prepare-public-001/manifest.json \
  --asset-root /root/autodl-tmp/lite-omnijev/artifacts/s0-prepare-public-001 \
  --model-dir /root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --hf-reference reports/s0-canonical-http-cpu-003.json \
  --output reports/s0-native-processor-new.json
```

The default six-case canary covers text, image, screenshot, audio, video with 8 frames, and video with 16 frames. Only after it succeeds should `--full` process all 100 public variants. Existing report paths are refused. Reports contain hashes, dimensions, counts and sanitized error stack locations; prompts, media bytes and token IDs remain in memory.

Comparisons require exact expanded prompt IDs, frozen HF input hashes, and these documented field translations:

| HF field | Native field | Comparison |
|---|---|---|
| `pixel_values` | `pixel_values` | Exact shape, dtype and bytes |
| `image_position_ids` | `pixel_position_ids` | Exact shape, dtype and bytes |
| `input_features` | `input_features_padded` | HF valid mask frames, matching native per-item unpadding |
| `input_features_mask` | `input_features_mask` | HF valid mask frames |
| `pixel_values_videos` | `pixel_values_videos` | A single HF video batch `[1,F,P,D]` maps to native flat frames `[F,P,D]`; exact dtype and bytes |
| `video_position_ids` | `pixel_position_ids_videos` | Same single-video batch translation; exact dtype and bytes |
| `num_frames_per_video` | `video_frame_counts` | Exact shape, dtype and bytes |
| Frame indices / fps | `video_timestamps` | Exact numerical timestamps |

The processor cache is disabled. This does not establish cache-hot/cold equality or native HTTP media IO behavior. Canonical image PNG, audio PCM16 WAV and video JPEG bytes are decoded into the native processor's documented input objects; video retains frame indices and fps.

If normal `ModelConfig` architecture inspection's child process fails, preserve that evidence. The optional `--same-process-inspection` is a separate diagnostic: it registers the identical installed official `Gemma4ForConditionalGeneration` class through `ModelRegistry.register_model` to avoid the lazy-inspection subprocess. It never changes model info, classes, host packages or the production engine. Diagnostic success must not relabel normal initialization as passed. Keep its report, source hash and exit separate; do not repeatedly retry a failed environment.

Official interfaces: [Gemma4 processor tests](https://github.com/vllm-project/vllm/blob/v0.31.0/tests/models/multimodal/processing/test_gemma4.py), [registry](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/multimodal/registry.py), [Gemma4 field adaptation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/gemma4_mm.py). The installed 0.31 source is the runtime authority; test helpers that download a model are not used.

## Observed 2026-10-14 result

Both bounded Linux attempts are blocked with zero cases. Normal initialization exited 1 after the lazy architecture-inspection child received SIGKILL. The separate in-process official-class diagnostic constructed `ModelConfig` on `UnspecifiedPlatform`, then exited 137 while creating the native processor. Its last checkpoint RSS was 1,338,568 KiB. The post-failure cgroup was near its 2 GiB limit, but `oom=0` and `oom_kill=0`; neither the killer nor CPU-platform incompatibility is established. No full-100 run was attempted and no further retry is scheduled.

The original checkpoint files are preserved, including the second file's interrupted `running` status; its separate exit record establishes failure. The first archived snapshot discovered the ancestor remote repository's Git HEAD when reporting its revision. The summary records this limitation and the real snapshot base plus exact source hashes; the CLI now honors an explicit `S0_CODE_VERSION` for non-Git snapshots. See `reports/s0-native-vllm-processor-summary-001.json` for the evidence hashes and remaining gates.

## H20 resource recovery, 2026-10-14

Primary 0.26 normal initialization, six cases and full100 now pass on the 150GiB/16CPU host. The official pinned checkpoint config has image `do_normalize=false` and video `do_normalize=true` (identity mean/std). Native 0.26 processes video frames through its image processor. Its default caused a maximum pixel difference of 5.96e-8. The supported per-request `mm_processor_kwargs={"images_kwargs":{"do_normalize":true}}` restores the official video normalization for video-only requests; image requests retain their defaults. HF preparation binds these kwargs by SHA, and both native comparison and actual HTTP transport use them. No engine patch or numerical tolerance is used. A one-ULP changed-pixel regression still fails.

Reports: `reports/gpu-transition-input-and-hf-acceptance-001.json`; full raw evidence on the persistent remote tree `/root/autodl-tmp/gpu-transition-ba4b22a/artifacts/input-acceptance-001`. New references must come from the same installed processor/runtime, not historical Mac summaries. For the gateway fixed100 test, set `S0_CANONICAL_REFERENCE` to the new passed `hf-a.json`; the test validates its cohort manifest hash. This CPU input result is separate from live HF/vLLM probability acceptance.

## Explicit image-budget profile (S2, 2026-10-15)

`python -m scripts.s0_prepare ... --profile canonical_http --image-max-soft-tokens 1120`
produces a separately frozen reference. Pass the same `--image-max-soft-tokens 1120`
to `scripts.s0_vllm_processor`; a default-budget reference is rejected for this profile.
The model assets remain unchanged. Video keeps its own frame budget and normalization.

Real CPU evidence: `reports/s2-budget1120-hf-cpu-001.json` and
`reports/s2-budget1120-native-cpu-001.json`, all100 public cases, exact input equality
under pinned vLLM0.26. Runs used isolated remote checkout
`/root/autodl-tmp/image-budget-cpu-check-001` at60528f2 while GPU quality evaluation
continued in its unchanged checkout. HF/native elapsed41.03/70.13seconds are CPU
preparation timings, not serving latency. Actual HTTP probabilities, LoRA serving and
performance still need their release checks.

Training uses `image_max_soft_tokens` in LoopConfig; evaluation uses
`--image-max-soft-tokens`; gateway uses `LITE_IMAGE_MAX_SOFT_TOKENS`. Requested image
budgets travel in hash-bound `mm_processor_kwargs.images_kwargs.max_soft_tokens`.
Unspecified settings retain the official default and existing checkpoint identity.
