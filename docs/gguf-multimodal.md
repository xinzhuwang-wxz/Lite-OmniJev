# GGUF quick start

Use llama.cpp with Gemma4 multimodal support (`llama-mtmd-cli`), a GGUF model and the bundled mmproj.
Tested runtime: llama.cpp `0.7.0-dev`, commit `9d1878a4354e743529fbe09e7350a957f5e39e9a`.
The tested quantization formats are F16/Q8_0/Q4_K_M.
[Tested llama.cpp source](https://github.com/ggml-org/llama.cpp/tree/9d1878a4354e743529fbe09e7350a957f5e39e9a).

Run the following commands from the downloaded model directory.

```bash
MODEL=gguf/lite-omnijev-e4b-d3p600-q4_k_m.gguf
MMPROJ=gguf/mmproj-lite-omnijev-e4b-d3p600-f16.gguf

llama-mtmd-cli -m "$MODEL" --mmproj "$MMPROJ" --image shot-rgb.png \
  -p '用一句话描述这张图片。' -n 48 --temp 0 --jinja < /dev/null
llama-mtmd-cli -m "$MODEL" --mmproj "$MMPROJ" --audio clip.wav \
  -p '用一句话描述这段音频。' -n 48 --temp 0 --jinja < /dev/null
llama-mtmd-cli -m "$MODEL" --mmproj "$MMPROJ" --video clip.mp4 \
  -p '用一句话描述这段视频。' -n 48 --temp 0 --jinja < /dev/null
```

Screenshots: convertRGBA toRGB and resize proportionally to maximum side1344 before inference.
Background execution uses`< /dev/null`; allow sufficient cold-load time for the selected file size.

| Format | Model file | Text answer agreement |
|---|---|---|
| F16 | 15.05GB | 20/20 |
| Q8_0 | 8.03GB | 20/20 |
| Q4_K_M | 5.34GB | 20/20 |

All three formats complete screenshot, audio and video generation probes.
