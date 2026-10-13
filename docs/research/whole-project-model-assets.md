# Whole-project model asset inventory

Generated: 2026-10-13T17:04:08.071759+00:00 UTC. Scope: model assets named by the original Lite-OmniJev plan that must be known before opening GPU time. Sources were official Hugging Face metadata/model cards plus the project design and existing research notes.

## Required inventory

| Model | Repo | Source SHA | License | Required bytes | Weight/checkpoint bytes | Role |
|---|---|---|---:|---:|---:|---|
| Gemma 4 E4B IT | `google/gemma-4-E4B-it` | `ee0ef6023621cff504d758262d4e04895a5af4a2` | apache-2.0 | 16.02 GB | 15.99 GB | student_backbone_primary_api_and_single_card |
| Gemma 4 E2B IT | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | apache-2.0 | 10.28 GB | 10.25 GB | student_backbone_edge_distillation_target |
| Jev-Omni | `akhilaaa3/Jev-Omni` | `5addda86ddee081a68fb067477ea100c221b8917` | apache-2.0 | 23.96 GB | 23.92 GB | teacher_audio_video_and_s0_reproduction_baseline |
| OneJev-4B | `OmniJev/OneJev-4B` | `c88e18653ceb7a8770716287f55fdefc79d6b588` | apache-2.0 | 10.37 GB | 10.35 GB | s0_s1_s2_text_image_screenshot_baseline |
| OneJev-27B | `OmniJev/OneJev-27B` | `aec969d68e7e8e8b0df051e065c04304c0cd7163` | apache-2.0 | 54.73 GB | 54.71 GB | teacher_text_image_screenshot_and_upper_baseline |
| Valen-Preview-0923 / Valen-2B | `Valen-Team/Valen-Preview-0923` | `30096aaabb2689614c83a79f64f38c865c2ce1c1` | apache-2.0 | 1.16 GB | 1.16 GB | required_original_sota_baseline_valen_2b |
| Qwen3.5-2B frozen base for Valen | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | apache-2.0 | 4.57 GB | 4.55 GB | required_dependency_for_valen_preview_0923 |

Known required model asset lower bound: **121.09 GB**. Optional model-card/demo assets: **0.03 GB**. Required weight/checkpoint bytes: **120.93 GB**.

Known model + download/temp + checkpoint lower bound: **289.26 GB** = required model assets + 15% model download/cache/temp reserve + 150GB checkpoint/adapter/quantization reserve. This is **not** a whole-project enough-to-open-GPU assertion: total dataset/eval-media budget is still unknown until the data audit completes.

## Notes and risks

- Jev-Omni preserves the already frozen teacher audit/download SHA `5addda86ddee081a68fb067477ea100c221b8917`. Current HF `main` was observed at `fee89b076e2263bfa8bce8d34142caa4ca733a48`, but that is informational drift only and must not change admitted teacher provenance.
- Valen-Preview-0923 is treated as the original Valen-2B baseline because the official card says it is the 2B Sokoban-trained preview and same checkpoint as Valen-Sokoban-RLCD-2B; it requires Qwen/Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc` as a separate frozen base download.
- OneJev-27B is required by the original plan and is not the prohibited JEV-27B/JEV-27B-VL family named in the project contract.
- `.gitattributes` is included as required provenance metadata when present. It is tiny metadata, not an additional model weight and not evidence that E4B has a missing weight.
- This slice did not start large model downloads; root coordinates network/download jobs. All required download claims still need local file size plus SHA verification.

## Readiness by role

- Student backbones: E4B is already reported verified elsewhere at the same SHA; E2B still needs full hash-verified download before S6/edge work.
- Teacher/reference: Jev-Omni and OneJev-27B are both required by the original teacher plan. Jev-Omni's `head.pt` is required and must not be silently omitted; its PyTorch head needs the separate safety/conversion path already owned by the teacher lane.
- Baselines: OneJev-4B and Valen-2B were not covered by the older four-model inventory. They are now explicit required downloads so S0/S1/S2/S6 baseline gates cannot quietly omit them.
- Prohibited: JEV-27B and JEV-27B-VL remain disallowed; no repo in this manifest points at those identities.

## Evidence files

- Machine-readable manifest: `configs/project-model-assets.json`.
- Existing project inventories reused for E4B/E2B/Jev-Omni/OneJev-27B: `docs/research/*inventory.json`.
- Current official metadata was refreshed with `huggingface_hub.HfApi().model_info(..., files_metadata=True)` for all repos listed above; Jev-Omni was queried at the old fixed SHA, not floating `main`.
