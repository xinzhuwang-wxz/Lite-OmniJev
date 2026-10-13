# Stage3 remaining media plan

Generated in `/tmp/lite-omnijev-stage3-source-audit` for the remaining stage3 media question. This note owns only the remaining-media recommendation; it does not update shared asset ledgers, STATUS, tickets, training configs, or admission state.

## Scope conclusion

The original stage3 row adds 40k rows: Clotho-AQA / AudioCaps 10k, TTS speech 8k, NExT-QA / STAR / CLEVRER 12k, LLaVA-Video sampled 5k, and cross-modal combinations 5k. Per the current root instruction, AudioCaps is already omitted by the user/ADR0004 and the audio question need is covered by Clotho/TTS. The remaining hard media question is therefore the video side: the 12k video-QA tranche and the 5k LLaVA-style video instruction tranche.

CLEVRER is now enough to satisfy the 12k video-QA count without forcing raw NExT-QA/VidOR, VGGSound, STAR/Charades, or the full LLaVA-Video payload. The downloaded official CLEVRER train split has 152,572 public `question_id` occurrences over 10,000 train scenes/videos; validation has 76,368 public question IDs over 5,000 scenes/videos. The official CLEVRER README says the dataset is CC0, and the prepared offline source passed size/SHA/ZIP CRC checks in [reports/clevrer-source-download-snapshot.json](../../reports/clevrer-source-download-snapshot.json).

Recommended first-launch stage3 media plan:

1. Fill the 12k NExT-QA / STAR / CLEVRER tranche from CLEVRER train only, sampled by `scene_index` / video family so variants and siblings stay in one pool.
2. Reserve CLEVRER validation for dev/calibration if needed; do not use CLEVRER test questions for supervised training because public answers are absent.
3. Do not block GPU opening on NExT-QA raw VidOR media, VGGSound raw YouTube clips, STAR/Charades, or LLaVA-Video’s 1.28 TB payload.
4. Treat the 5k LLaVA-Video bucket as ADR-gated: either defer it or regenerate equivalent allowed-teacher soft labels over already admitted CC0/BY media. Do not admit LLaVA-Video direct media/labels for main commercial training under the current rights evidence.

## Source-by-source status

### CLEVRER

Status: prepared and count-sufficient for the 12k video-QA bucket, pending root integration into shared ledgers.

Evidence:

- Official source: [CLEVRER project page](https://clevrer.csail.mit.edu/)
- License evidence: [official README.txt](https://data.csail.mit.edu/clevrer/README.txt) says the dataset is CC0.
- Prepared source: `/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official`
- Count evidence from streaming counts only: train `question_id=152572`, train `scene_index=10000`; validation `question_id=76368`, validation `scene_index=5000`.

Executable use plan:

- Build a deterministic training manifest of 12,000 rows from train questions only.
- Group by `scene_index`/video id before splitting train/dev/calibration to preserve sample-family isolation.
- Keep downloaded ZIPs unextracted until the owned preprocessing job validates archive paths and writes a separate extraction receipt.

### NExT-QA / VidOR

Status: optional diversity source, not required for the first-launch count.

Prepared metadata shows NExT-QA has 34,132 train rows over 3,870 train videos, 4,996 validation rows over 570 videos, and 8,564 test rows over 1,000 videos. The NExT-QA repository is MIT for code/labels, but the README points raw videos to NExTVideo and maps NExT-QA videos to VidOR IDs. VidOR media comes from YFCC100M/Flickr-style sources, so individual media license mapping is the real blocker, not the QA CSV itself ([NExT-QA repository](https://github.com/doc-doc/NExT-QA), [VidOR project](https://xdshang.github.io/docs/vidor.html)).

Executable future plan, if diversity is needed:

- Use the mirrored NExT-QA CSVs and `map_vid_vidorID.json` to build an ID-only manifest; do not print question text.
- Join VidOR IDs to YFCC/Flickr license metadata before any raw media fetch.
- Admit only per-video commercial-compatible licenses with attribution captured; skip NC/unknown/unavailable media.
- Treat this as source broadening, not a GPU-opening blocker.

### STAR / Charades and MVBench STAR

Status: research/eval only under current evidence; not main commercial training.

STAR has large public split counts in the pinned source record (`Train=45,731`, `Val=7,098`, `Test=7,377`), and a MVBench STAR archive is hash-verified locally for eval. That still does not make it a main training source. STAR raw media is Charades, and the recorded Charades terms are non-commercial, allow evaluation elsewhere, restrict redistribution/modified public data release, and require contacting AI2 for commercial use. The initial MVBench STAR archive covered 588/600 public references; the later official Charades recovery restored all 600/600 (reports/star-missing-source-recovery-001.json). This is evaluation media, not evidence of the 12k main training count.

Executable future plan:

- Keep STAR/MVBench STAR for eval/research unless a separate rights decision admits it.
- Never use STAR test answers or hidden answer material.
- If root wants STAR for main training, draft a separate source-scope ADR and obtain commercial-use permission first.

### Perception Test

Status: possible future source only after root policy admission; not needed for count now.

Perception has already appeared in the project as eval/source staging, and MVBench has Perception-derived public rows. The LLaVA-Video card reports 1,803 PerceptionTest open-ended QA items. Because the original stage3 12k row names NExT-QA / STAR / CLEVRER, using Perception for that bucket should be treated as a scoped substitution. It is not necessary for first launch because CLEVRER already covers the 12k count.

Executable future plan:

- Cross-check original DeepMind/Perception terms and public-label scope before any training reuse.
- If admitted, use it for diversity/calibration, not to unblock the first GPU run.

### VGGSound

Status: optional audio/video diversity source, not required after AudioCaps omission because Clotho/TTS covers the audio question need.

Prepared metadata shows VGGSound has 199,467 metadata rows, 161,810 unique YouTube IDs, 183,971 train rows, 15,496 test rows, and 310 labels. The VGGSound repository license covers dataset metadata under CC-BY-4.0, while original video copyrights remain with video owners ([VGGSound repository](https://github.com/hche11/VGGSound), [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)). That means the CSV is usable as a candidate manifest, but raw media is not automatically cleared for commercial training.

Executable future plan:

- Keep mirrored CSV/license metadata only.
- If later needed, run a metadata-only availability/license dry run for selected YouTube IDs.
- Admit only clips with compatible current license and attribution record; skip Standard YouTube License, unknown, unavailable, or attribution-incomplete clips.

### LLaVA-Video-178K

Status: do not download the full 1.28 TB payload; do not admit direct LLaVA media/labels to main commercial training under current rights evidence.

The pinned HF card says Apache-2.0, but its Uses section restricts the dataset to academic research and education and warns users to check OpenAI policy for GPT-4 generated data. Its source groups include academic/youtube splits and borrowed datasets such as ActivityNetQA, PerceptionTest, NExT-QA, and LLaVA-Hound ([LLaVA-Video-178K dataset card](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)). The project ledger records 1,281,048,393,477 bytes across 302 files. The card reports 178,510 new captions, 960,792 open-ended QA, 196,198 multiple-choice QA, and imported groups such as 17,090 NExT-QA open-ended items, 17,024 NExT-QA MC items, and 1,803 PerceptionTest open-ended items.

Executable future plan:

- Keep only the pinned README/API metadata for now.
- If the 5k LLaVA-style bucket must remain for first launch, regenerate it with the allowed teacher on already admitted CC0/BY media, primarily CLEVRER and later any rights-filtered sources.
- If root wants research-only LLaVA sampling, select 5k metadata rows from pinned groups after source-provenance filtering, without downloading the full repository or using it for main commercial training.

## Minimal ADR draft

Title: Stage3 remaining video media source substitution for first launch

Decision: For first launch, satisfy the 12k video-QA bucket from verified CLEVRER CC0. Do not admit LLaVA-Video-178K direct media/labels to main commercial training. Defer the 5k LLaVA-Video bucket or regenerate equivalent allowed-teacher soft labels over already admitted CC0/BY media.

Context:

- AudioCaps is omitted by user/ADR0004, and the audio question count is covered by Clotho/TTS.
- NExT-QA/VidOR and VGGSound remain useful future diversity sources, but only after per-media license mapping.
- STAR/Charades is research/eval only under current non-commercial media terms.
- CLEVRER has enough CC0 public training rows to meet the 12k video-QA count without lowering row count.

Consequences:

- GPU opening should not be blocked by optional raw NExT-QA/VidOR, VGGSound, STAR, or full LLaVA-Video downloads.
- Quality target is preserved by keeping row counts and allowing future diversity slots after proper source admission.
- Any future broadening must carry an exact source manifest, family-isolated splits, and rights evidence.
