# S0 multimodal HF/vLLM parity runbook

`scripts/s0_parity.py` is the GPU-era parity harness for the S0 multimodal gate. It compares a sealed Hugging Face CUDA reference against a native vLLM chat server using the exact `canonical_http` prepared input profile from `train.multimodal.prepare_case`.

The script does not start a server and does not treat the HTTP endpoint as remote proof. The operator must provide a signed or otherwise reviewable server record whose model, adapter, manifest, dtype, launch command, raw-logprob mode, code hashes, and `raw_logprobs_confirmed=true` provenance match the current run. The report records this as operator-attested service identity and stores the launch command hash, not remote attestation.

## HF reference

Run this only after the fixed S0 model assets, PEFT adapter, processor, and 100-case manifest are present locally and CUDA is available:

```bash
PYTHONPATH=/path/to/Lite-OmniJev python -m scripts.s0_parity hf-reference   --manifest artifacts/s0/manifest-100.json   --asset-root artifacts/s0/media   --model-dir artifacts/models/google__gemma-4-E4B-it/<revision>   --asset-manifest <data-disk>/artifacts/manifests/e4b.json   --adapter-dir artifacts/s0/adapter   --dtype bfloat16   --report reports/s0-parity-hf-reference-001.json
```

`--asset-manifest` must be the per-model pinned manifest with `size` keys (box:
`/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json`, sha256 `207ce3a8…`,
identical to the passing run in `reports/s0-full100-gpu-acceptance-001.json`);
`configs/project-model-assets.json` is the whole-project inventory with a
different file schema (`bytes`) and is not a valid `--asset-manifest` input.

The command verifies the fixed model asset manifest before any forward pass, refuses to load the backbone when CUDA is absent, fixes seed `42`, disables TF32, loads the official `Gemma4ForConditionalGeneration` with `local_files_only=True`, `trust_remote_code=False`, and `use_safetensors=True`, then runs two sequential passes over the 100 cases. Each case is prepared through `canonical_http`, and the full prepared multimodal tensor dictionary is passed to the model; the code does not reduce examples to `input_ids`. The report includes package versions, CUDA/GPU summary when available, git revision, and required source hashes. If the CUDA gate blocks before model load, the command still writes a blocked report with elapsed time, environment, cost, source, and error type.

The reference report is ready only when both HF passes are exactly repeatable and every row contains complete finite raw logprobs for all answer tokens plus `U`. The report remains a pre-release gate artifact; it is not stage acceptance by itself.

### Numerical reference precision

The H20/primary0.26 six-case diagnostic found large BF16 differences and prefix-cache-dependent repeats. FP32 with `NVIDIA_TF32_OVERRIDE=0`, `TRITON_F32_DEFAULT=ieee`, eager execution and prefix caching disabled restores exact native repeats and HF/native maxima below 0.001 in the six-case canary. The formal runner also accepts `--dtype float32`; this is a numerical reference configuration, not final serving performance acceptance or a change to BF16 LoRA training.

Native Gemma4 audio/vision towers initialize from nested HF configs. For FP32, use the supported `--hf-config-path` to a copy of the original config with only root/audio/text/vision `dtype` changed to `float32`; bind original/copy hashes and changed fields in the server record. Original checkpoint files and dimensions stay fixed. Set an explicit 4GiB `--kv-cache-memory-bytes` for the single-sequence/8192-token canary, and do not overlap other GPU initialization with automatic memory profiling. Formal100 results, cache-enabled correctness/performance and final BF16/FP8 release quality still require their own evidence.

## vLLM comparison

After a separately launched vLLM server is ready, create a server record containing the exact provenance from the HF reference report plus `operator_attested=true`, `raw_logprobs_confirmed=true`, `logprobs_mode="raw_logprobs"`, matching `dtype`, a nonempty `launch_command`, and the served model name. Then run:

```bash
PYTHONPATH=/path/to/Lite-OmniJev python -m scripts.s0_parity compare-vllm   --manifest artifacts/s0/manifest-100.json   --asset-root artifacts/s0/media   --model-dir artifacts/models/google__gemma-4-E4B-it/<revision>   --asset-manifest <data-disk>/artifacts/manifests/e4b.json   --adapter-dir artifacts/s0/adapter   --dtype bfloat16   --reference reports/s0-parity-hf-reference-001.json   --server-record artifacts/s0/vllm-server-record.json   --upstream-url http://127.0.0.1:8001   --report reports/s0-parity-vllm-001.json
```

The comparison re-prepares all 100 cases locally and rejects a reference if any prompt, media, tensor summary, answer-token mapping, model, adapter, code, or dtype binding no longer matches. It then performs two native chat reads through `serve.vllm_chat.VllmChatClient` and checks exact repeatability plus HF/vLLM raw probability, K+1 joint probability, conditional probability, and `p(U)` differences. All maxima must be strictly below `0.001`.

## CPU-only control

`synthetic-control` writes a small report proving that the row math, repeatability comparator, report schema, and overwrite refusal work on CPU. This is useful before opening a GPU session, but it is explicitly synthetic evidence and cannot satisfy the HF/vLLM gate:

```bash
PYTHONPATH=/path/to/Lite-OmniJev python -m scripts.s0_parity synthetic-control   --report reports/s0-parity-cpu-001.json
```

## Boundaries

- The script refuses to overwrite an existing report.
- It never starts or manages a vLLM server.
- It never loads the official backbone on CPU.
- It does not read sealed holdout content; the supplied manifest must be the non-holdout S0 100-case manifest.
- Reports contain hashes, token IDs, probabilities, environment, elapsed time, and exit code evidence; they avoid storing prompt text or media bytes.
- A passing CPU synthetic report is not a replacement for the CUDA HF reference or the live vLLM comparison.

## 2026-10-14 admitted numerical reference

The real 100-case GPU numerical reference passed in float32 with vLLM 0.26,
`TRITON_F32_DEFAULT=ieee` and `NVIDIA_TF32_OVERRIDE=0`. The supported
`--hf-config-path` copy changes only root/audio/text/vision `dtype` fields to
float32; original weights, config, template and strict `<1e-3` gate remain fixed.
See `reports/s0-full100-gpu-acceptance-001.json` for raw artifacts, source/runtime
bindings, exact reruns, gateway100, Chromium evidence and independent review.
The gateway harness checks the existing chance-adjusted confidence contract.
This reference does not admit BF16/FP8 serving performance or cache-enabled
correctness. The failed BF16 evidence remains preserved. Exact launch paths and
resume boundaries are in `plan/handoff-2026-10-14-s0-parity-and-teacher-admission.md`.

## Explicit image token budget (ADR0019)

Both `hf-reference` and `compare-vllm` accept optional
`--image-max-soft-tokens 70|140|280|560|1120`. For an explicit1120 experiment,
append `--image-max-soft-tokens 1120` to **both** commands and use new report
paths. The HF reference, local reference re-preparation, and both native chat
passes use the same requested value. The existing official preparation path
sets HF image processing and supplies native `mm_processor_kwargs` through
`PreparedDecision`; video and audio preprocessing stay under their existing
contracts.

The explicit value is recorded in `provenance.image_max_soft_tokens` and must
also be present in the server record's exact copied provenance. Image input
identity includes its effective prepared budget and processor-kwargs hash.
A default/explicit or different-explicit reference mismatch is rejected before
native HTTP requests. Server provenance must match too; it remains operator
attestation, not remote configuration proof.

Omitting the option preserves official-default preparation and the historical
provenance/input-hash shape. Old reports are never rewritten, and all existing
source, model, tensor and dtype checks still apply: a reference from another
source revision is not automatically reusable merely because its budget is
default. Generate fresh evidence under the intended revision/profile when
those bindings differ. CLI and mocked CPU tests establish propagation and
rejection behavior only; each explicit budget still requires its own real
HF/native GPU parity and performance acceptance.
