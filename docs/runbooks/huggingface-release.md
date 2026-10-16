# Hugging Face release preparation

Final HF open publication is user-authorized intent. **Current release is blocked by unmet product gates; this runbook does not create a repository or upload.** Namespace/account is unverified. Do not request/read/print tokens during preparation. No novelty, SOTA, calibration or readiness claim follows from a template.

Templates: `.scratch/delivery/huggingface/e4b-adapter/` and `e2b-edge/`. E4B is adapter-only. E2B is a future adapter/edge documentation layout; distilled/quantized weights and measured runtime evidence do not yet exist. The current allowlist intentionally cannot ship merged base or quantized full weights; such a later package needs its own accepted artifact schema. Each directory contains README model card, release-manifest draft, NOTICE and examples. LICENSE and actual adapter/config are currently missing, explicitly reported as incomplete.

## Verified license evidence

Existing inventories `docs/research/google--gemma-4-{E4B,E2B}-it-inventory.json` report gated=false, Apache-2.0 and Google's Gemma4 license link. Read-only remote verification2026-10-16 confirmed:

|Base|Revision|Actual downloaded README SHA256|
|---|---|---|
|google/gemma-4-E4B-it|ee0ef6023621cff504d758262d4e04895a5af4a2|b21e4f69614ccd77baa2f3797d05311040dee07b989cb9f0d25111aa4b605b2c|
|google/gemma-4-E2B-it|3e22461f65e89153144f8adb70e3b8c2cc9845a7|3e0608a6be80e3eb040a5aa6c76707809503deb85be6360184a118fb6156526c|

Paths: `/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-<E4B|E2B>-it/<revision>/README.md`. Their frontmatter explicitly states Apache-2.0 and license_link. No standalone LICENSE/NOTICE appeared in actual top-level files. No Gemma-specific acceptance receipt was found; inventories are ungated, so this is **not a fabricated acceptance record**, and no terms were accepted in this preparation.

[Google Gemma4 license](https://ai.google.dev/gemma/docs/gemma_4_license) redirects to [Google Apache2 terms](https://ai.google.dev/gemma/apache_2). Therefore Apache2 metadata here has actual asset and publisher evidence, not a blanket inference from earlier Gemma. At final packaging include full applicable license text, preserve attribution/NOTICE if applicable, identify modifications, and audit selected adapter/data provenance. The project's source-code license is separate; evaluation-only dataset admission does not grant training or redistribution.

## Packaging steps after gates

1. Root selects a single accepted adapter and locks model/base/config/source/environment/calibration identities. Bind actual aggregate S4, S5, S6 and FINAL acceptance; no private per-example gate records. Namespace ownership verification remains a later publishing step.
2. Export only `adapter_model.safetensors` and `adapter_config.json` with PEFT safe serialization. No merged weights, optimizer, RNG/checkpoint files, teacher weights or pickle. Preserve official shared template/answer mapping through the accepted project runtime; do not add tokenizer/base assets to the adapter repository.
3. Fill the model card and manifest with actual hashes, training recipe/seed, data provenance summaries, reproduction environment, evaluation limitations and genuine metrics. Include complete license text and attribution. Replace all pending placeholders honestly. E2B memory, quantization drop and runtime support require real S6 evidence.
4. Run the CPU check below on a dedicated staging directory. Unknown files, symlinks, credentials patterns and non-decoder-LoRA tensor names are rejected before upload. Missing required files remain incomplete. This checker is **structural only**, always release_ready=false: root must additionally use the project's release-readiness/gate mechanism and inspect all staged text for private material. The filename allowlist cannot prove the semantic contents of an allowed README/JSON are public.
5. Only after final acceptance and explicit target namespace verification should root perform the actual HF publishing operation using a separate exact file list. Never recursively upload the workspace/artifacts directory. This helper has no network/auth/upload capability.

```sh
python .scratch/delivery/huggingface/check_package.py .scratch/delivery/huggingface/e4b-adapter
python .scratch/delivery/huggingface/check_package.py .scratch/delivery/huggingface/e2b-edge
python -m pytest -q .scratch/delivery/huggingface/test_package.py
```

An exit0 means no forbidden file found; `inventory_incomplete` is expected for templates and is **not publication readiness**. Exit2 rejects unsafe structure. Inference example only loads local base+adapter via PEFT; it was not run against a GPU and is not a completed probability readout. Actual decision inference must reuse accepted project preparation/gateway, answer-token completeness including U, thinking disabled, raw logits and distinct p(U).

Official references checked2026-10-16: [HF model cards](https://huggingface.co/docs/hub/model-cards) specifies README YAML/base_model metadata; [HF PEFT Hub](https://huggingface.co/docs/hub/peft) describes separate base and adapter loading; [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint) distinguishes adapter config/weights from base weights. We do not invoke push_to_hub or automatic model-card generation as a release approval shortcut.
