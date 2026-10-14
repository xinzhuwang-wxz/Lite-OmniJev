# S0 native vLLM processor CPU check

This CLI compares the installed vLLM 0.31 processor with the frozen canonical HTTP HF reference. It creates `ModelConfig` and the official `MULTIMODAL_REGISTRY` processor only: no `LLM`, engine, backbone constructor, weights, generation or CUDA forward. A processor pass is not service, raw-probability, quality or release acceptance.

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
| `pixel_values_videos` | `pixel_values_videos` | Exact shape, dtype and bytes |
| `video_position_ids` | `pixel_position_ids_videos` | Exact shape, dtype and bytes |
| `num_frames_per_video` | `video_frame_counts` | Exact shape, dtype and bytes |
| Frame indices / fps | `video_timestamps` | Exact numerical timestamps |

The processor cache is disabled. This does not establish cache-hot/cold equality or native HTTP media IO behavior. Canonical image PNG, audio PCM16 WAV and video JPEG bytes are decoded into the native processor's documented input objects; video retains frame indices and fps.

If normal `ModelConfig` architecture inspection's child process fails, preserve that evidence. The optional `--same-process-inspection` is a separate diagnostic: it registers the identical installed official `Gemma4ForConditionalGeneration` class through `ModelRegistry.register_model` to avoid the lazy-inspection subprocess. It never changes model info, classes, host packages or the production engine. Diagnostic success must not relabel normal initialization as passed. Keep its report, source hash and exit separate; do not repeatedly retry a failed environment.

Official interfaces: [Gemma4 processor tests](https://github.com/vllm-project/vllm/blob/v0.31.0/tests/models/multimodal/processing/test_gemma4.py), [registry](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/multimodal/registry.py), [Gemma4 field adaptation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/gemma4_mm.py). The installed 0.31 source is the runtime authority; test helpers that download a model are not used.

## Observed 2026-10-14 result

Both bounded Linux attempts are blocked with zero cases. Normal initialization exited 1 after the lazy architecture-inspection child received SIGKILL. The separate in-process official-class diagnostic constructed `ModelConfig` on `UnspecifiedPlatform`, then exited 137 while creating the native processor. Its last checkpoint RSS was 1,338,568 KiB. The post-failure cgroup was near its 2 GiB limit, but `oom=0` and `oom_kill=0`; neither the killer nor CPU-platform incompatibility is established. No full-100 run was attempted and no further retry is scheduled.

The original checkpoint files are preserved, including the second file's interrupted `running` status; its separate exit record establishes failure. The first archived snapshot discovered the ancestor remote repository's Git HEAD when reporting its revision. The summary records this limitation and the real snapshot base plus exact source hashes; the CLI now honors an explicit `S0_CODE_VERSION` for non-Git snapshots. See `reports/s0-native-vllm-processor-summary-001.json` for the evidence hashes and remaining gates.
