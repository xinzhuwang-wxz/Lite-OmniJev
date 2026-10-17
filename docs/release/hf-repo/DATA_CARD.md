# Data statement

| Source | License | Modality | Use |
|---|---|---|---|
| CLEVRER train/validation | CC-BY 4.0 | video (+ program truths) | training facets + QA |
| VQA v2 | per-original | image | training/eval |
| Clotho-AQA | per-original | audio | training/eval |
| LibriSpeech | CC-BY 4.0 | audio | training/eval |
| GUI-Odyssey | commercial-licensed | screenshot | training/eval |
| synthetic text pool | self-produced | text | training/eval |

Isolation: training / dev / calibration / sealed-holdout pools are split by original sample family (variants and counterfactuals never cross pools); per-file SHA-256 provenance for every sample and media asset is published with the release.

Excluded by policy: JEV-27B weights/outputs (terms-of-service), any eval-set contamination (audited, zero cross-pool overlap).
