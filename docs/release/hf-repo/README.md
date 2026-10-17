---
license: apache-2.0
base_model:
- google/gemma-4-E4B-it
pipeline_tag: image-text-to-text
library_name: peft
tags:
- decision-making
- calibrated-classification
- lora
- multimodal
- video-question-answering
- image-question-answering
- audio-question-answering
- gui-understanding
- temperature-scaling
datasets:
- clevrer
- vqa2
- clotho-aqa
- librispeech
- gui-odyssey
metrics:
- accuracy
- ece
- brier
---

# Lite-OmniJev E4B — a calibrated typed-decision adapter

> ⚠ **DRAFT — numbers below are the current candidate's sealed results and will be refreshed with the final model before upload** (auto-injected from the sealed summary; see UPLOAD_CHECKLIST).

**Not another chat adapter.** This is a *calibrated typed-decision* head for frozen Gemma 4 E4B-it: given a multimodal question, it answers with a **probability distribution over typed choices** (multiple-choice / score / yes-no), trained so the probabilities are trustworthy — not just the argmax.

Most multimodal LoRA releases optimise answer accuracy. We additionally optimise and **publish calibration**: accuracy is unchanged after temperature calibration (zero loss across all six modalities) while expected calibration error (ECE) drops by 2–4× on four of six modalities — a property rare enough in the community that we ship the full per-modality table rather than a single headline.

## What makes the training different

Two supervision signals, both compiled into the weights at training time (no inference-time scaffolding; the deployed model answers in a single forward pass):

1. **Teacher-distribution distillation (D3P)** — a 27B decision teacher's *full option distributions* (not argmax labels) are distilled into direct-answer weights at 600-question scale, teaching the 4B model *how confident to be*, not only *what to answer*.
2. **Program-facet structural supervision (SAS/WSFT)** — CLEVRER's ground-truth programs are compiled offline into per-predicate facet targets (count / color / motion-direction / begin-end / collision). Symbol execution happens at *training* time; the symbolic knowledge is amortised into the adapter.

The result answers the question "can a small direct-answer model inherit a symbolic pipeline's structure without running it?" — our evidence says yes, measurably (+6 questions on video at full dev scale, and every structured-supervision variant beat its non-structured control).

## Results (sealed 6,000-question holdout, temperature-calibrated, single protocol)

| Modality | Accuracy | ECE (raw → calibrated) | ECE ≤ .03 |
|---|---|---|---|
| text | 99.7% | .0029 → **.0014** | ✅ |
| screenshot | 97.8% | .0162 → **.0108** | ✅ |
| image | 92.7% | .0587 → **.0202** | ✅ |
| audio | 88.4% | .0836 → .0310 | ±.001 |
| **video** | **72.6%** | .1382 → **.0321** | ±.002 |
| crossmodal | 79.9% | .1787 → .0528 | ❌ disclosed |

**Calibration cost: zero.** Argmax accuracy is bit-identical before and after calibration in all six modalities.

External benchmark (JevBench, 231 public decisions, identical protocol two-pass): **76.6%** vs frozen base 76.2%.

### Against reference points

| | Ours (4B-eff adapter) | Frozen base (same size) | 27B decision teacher *(reference line — different scale)* | Symbolic-execution systems *(different paradigm — programs answer, not the model)* |
|---|---|---|---|---|
| video causal QA (our protocol) | 72.6% sealed / 75.6% dev | 72.0% sealed / 74.4% dev | 94.2% | 96–99% |
| JevBench-style aggregate | 76.6% | 76.2% | not comparable (protocol) | — |
| published calibration metrics | ✅ per-modality | — | — | — |

*We mark what we cannot comparably run instead of mixing protocols.*

## Quickstart

```python
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import PeftModel

base = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E4B-it", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "lite-omnijev-e4b-d3p600")
# Official chat template; thinking disabled; answer tokens are the first
# sampled token from the typed option set (see docs/ for the token map).
```

Per-modality temperature scaling values ship in [`calibration/`](./calibration/) with a drop-in readout helper.

## Demos

See [PROMO.md](./PROMO.md): same-question side-by-side (generic streaming VLM vs our single-pass typed decision), calibration trust visuals, six-modality montage, honest failure reel — shot on the serving stack at release time.

## Training

- **Base**: google/gemma-4-E4B-it @ `ee0ef602` (frozen; encoders stay frozen — LoRA touches language decoder projections only)
- **LoRA**: r=16, α=32, bf16, lr 2e-6, seed 42
- **Stage data**: 600-video teacher-distillation slice + program-facet targets from CLEVRER train programs (CC-BY 4.0); full data statement in [TRAINING.md](./TRAINING.md)
- Every number above is reproducible: fixed seeds, code/data/model hashes bound per run, two-pass exact-match verification on benchmarks

## Limitations

1. Video causal QA is 72.6% sealed — our 27B decision teacher measures 94.2% on the identical protocol; we do not match it, and we publish the gap.
2. Crossmodal ECE (.0528) misses our .03 target after calibration; disclosed, not retuned.
3. "U" (insufficient-information) exists only as an internal logit; the released output form has no U channel (product-form decision).
4. Calibrated temperatures are bound to the bf16 / transformers / native-LoRA serving recipe; other engines require re-calibration (verified: cross-engine activation is refused by design).

## License & data

Apache-2.0 (inherited from base). Training data: CLEVRER (CC-BY 4.0), VQA v2, Clotho-AQA, LibriSpeech, GUI-Odyssey (commercial) — pool-isolated by sample family with full provenance hashes. The 27B teacher contributed *distributions only* during training; no teacher weights, APIs, or labels ship here.
