# S6 M4 native CPU preparation — concrete result

2026-10-16. Native arm64 **raw-logits client + standard llama-mtmd-cli + llama-quantize built successfully**, within~11minutes of the25minute CPU cap. Existing cached source tar copied once, exact SHA09b36dba… verified. No new dependencies, internet model/source download, engine/core changes, GPU/Metal use, private data or HF/account operations. Exact files/flags/hashes are in JSON and `.scratch/s6/README.md`.

Actual paths are `/tmp/lite-omnijev-s6-cpu-001/build/{raw_logits_probe,bin/llama-mtmd-cli,bin/llama-quantize}`. Final configure/build exit0; standard CLI help/version exit0; raw readout selftest exit0; invalid arguments exit2. All are Mach-O arm64. CLI version honestly says0.6.0-dev/build0/commitunknown because source archive has no.git and build is outside any parent repository; pinned archive hash is provenance. No inherited repository commit. OpenSSL disabled after seeing default Homebrew dylib linkage; final clients only need system libc++/libSystem. Initial target failure exit2 is preserved in local build logs: standard llama-cli is gated behind server build in this release, so selected existing llama-mtmd-cli instead, no server launch or core patch.

## What is proven

`include/llama.h:1136–1142` defines complete n_vocab raw logits from last decode and unsampled `llama_get_logits_ith`; C++ client compiles/links directly to this API and has a real GGUF+pretokenized-prefix CPU decode path. Its actual numerical fixture includes A/B/U and an unselected word with.5 probability mass: full-vocab pA=.1/pB=.2/pU=.2, legal+U joint=.2/.4/.4, answer conditional1/3 and2/3. This demonstrates complete-vocab denominator versus selected-legal denominator and U separation. A NaN in the **unselected** token and duplicate answer IDs both reject. There is no sampler/top-k/logit-bias path. **Actual model getter output is not yet witnessed:** no accepted local E2B GGUF exists, so fixture math/compile/link are software evidence, not E2B parity/quality.

C API application integration does not alter the inference engine or add a head. Supplied prefix/token mapping must already come from official preparation with thinking disabled and generation-boundary validated complete legal+U IDs. Client does not claim to certify arbitrary answer-set completeness or render templates. No calibration applied implicitly; final trusted registry remains required.

## mtmd six-cohort boundary, based on this compiled source

|Cohort|Concrete available interface|Still unverified|
|---|---|---|
|text|Native client llama_decode + full vocab getter|Final E2B GGUF/official prefix and all legal+U token parity|
|image|Native mtmd CLI and `mtmd_bitmap_init`, Gemma4 vision source compiled|Own mmproj + official normalization/image1120/token-input equivalence|
|screenshot|Same image bitmap/projector path|Separate screenshot quality, real sizes and sameinput/performance|
|audio|`mtmd_bitmap_init_from_audio`, `mtmd_support_audio`, Gemma4a/ua preprocessing compiled|Actual E2B model capability, rate/chunks/normalization, sameinput and accuracy|
|video|MTMD_VIDEO=ON; helper support returns model vision capability; standard CLI --video/help exists|Default4FPS not project16frames. Exact frame/order/timestamp/visual tokens and E2B forward remain pending|
|crossmodal|`mtmd_tokenize_from_parts` and chunks API accept ordered text/bitmap parts|Mixed image+audio in the actual E2B context/projector/template and complete decision readout need real model test|

Actual support booleans require initialized mmproj context; no NULL-context call or fake true capability claim. The raw client currently handles supplied text prefix IDs; media context integration remains a small standard mtmd application binding, **not delivered as a six-modality decision service**. Cached llama.cpp sources are primary API evidence; this run did not fetch newer docs/code to claim compatibility.

## Minimal next real command after frozen E2B

Use the native raw client with accepted local `$E2B_GGUF`, verified official prefix-token CSV and complete legal+U mapping CSV (command in README). One text sample first proves actual unsampled row/finite outputs/token identity; then same accepted200-input package comparison. For media, populate the same llama context using standard mtmd helper and exact approved frame/audio/pixel recipe before calling the getter. Existing source converters/quantizer are available, but own E2B decision adapter→merged→GGUF/mmproj and quant/calibration are still pending; no E4B adapter transplantation or base-community GGUF substitution.

No actual model forward, conversion, distillation, six-modal quality, memory/latency, Mac clean-machine packaging, phone support or S6 acceptance was claimed. CPU portable build disables Metal and is intentionally not the final fast backend measurement. ADR0023 permits measured quality limitations and stage progress; correctness/template/probability/identity/packaging tolerances remain unchanged. No new Android/APK/dualOS requirements or engine/base decision introduced. Executable artifacts retained locally for the next accepted model; reproducible build source/scripts committed, binaries/model weights not committed.
