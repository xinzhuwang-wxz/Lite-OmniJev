---
library_name: peft
base_model: google/gemma-4-E4B-it
base_model_relation: adapter
license: apache-2.0
tags:
- lora
- decision-making
---

# Lite-OmniJev E4B decision adapter — unpublished template

**Not a released model. S4 quality, S5 calibration, S6 edge/runtime and FINAL acceptance are pending.** No final adapter/checkpoint hash, reproducible released score or verified HF namespace exists yet. This template is not evidence of innovation or quality improvement. Final repo name: `<verified-namespace>/lite-omnijev-e4b-decision` (placeholder).

Frozen base: `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`. Only decision LoRA weights will be supplied; obtain base separately. E2B distillation, quantization, measured RAM/latency and supported edge runtime artifacts remain pending. No current E4B pilot score is inherited by this future artifact.

## Intended behavior

Single-step text/image/screenshot/audio/video/crossmodal decisions via the project's official template and validated answer tokens, thinking disabled. Choice2–20/Score2–10 plus U; Noul yes/no plus U. Report p(U) separately and normalize answers after excluding U. No sampled text/top-k probability substitute. No production, calibrated confidence or device compatibility claim before corresponding gates.

## Reproduction and evaluation (pending)

Publish final source commit, environment lock hash, base/adapter/config hashes, admitted TRAIN provenance summary, actual training recipe/seed, aggregate stage acceptance reports, calibration version and evaluation exclusions. Never publish raw benchmark rows, training media, private calibration/holdout content, credentials or signed URLs. CPU checks and small development pilots do not meet S4/S5/S6/FINAL gates. See `examples/reproduce.md`.

## License and attribution

Apache-2.0 is based on verified Gemma4 asset card and Google's linked Gemma4 license, not inferred from earlier Gemma versions or this project's code license. Downloaded base README SHA256 `b21e4f69614ccd77baa2f3797d05311040dee07b989cb9f0d25111aa4b605b2c`. Original assets have no standalone LICENSE/NOTICE file in their top-level inventory; include complete upstream-linked Apache2 text at packaging, preserve applicable attribution/NOTICE and prominently identify adapter modifications. Final adapter provenance/license audit remains required; data evaluation permissions do not authorize dataset redistribution. No new terms were accepted. Google/Gemma trademarks are not licensed as product endorsements.

## Loading

`examples/infer.py` documents offline adapter loading only; final decision readout uses the project gateway/shared preparation path, not generate(). Multimodal and edge examples need final verified package and runtime receipts. All quality table entries: **pending**.
