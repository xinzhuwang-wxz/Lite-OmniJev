# Whole-Project Data Asset Inventory

Date: 2026-10-14. Scope: original design stages 0-6 plus explicit separation for optional 5b, research-only sources, internal holdout, and future generated assets. This is an offline-prep ledger, not a training admission approval.

No heldout content was read. No large dataset shard was downloaded. Metadata came from the original design parsed with `HTMLParser`, existing project admission notes, Hugging Face dataset APIs, GitHub repository APIs, and official project pages.

## Decision Summary

The project is not yet ready for a whole-project download sweep. Several sources are legally or procedurally ready to mirror after root coordination, but the full stage 0-6 inventory still has blocking gaps around licenses, account/form access, raw media rights, and aggregate label provenance.

The strongest immediate download candidates are:

| Asset | Status | Why |
|---|---|---|
| `Valen-Team/Valen-Training-General-100k` | downloadable with root coordination | Apache-2.0, immutable HF revision, exact 6.36 GB metadata; `assets.zip` is >100 MB. |
| `lmms-lab/PerceptionTest` | downloadable with root coordination | MIT, immutable HF revision, exact 65.59 GB metadata; use public labels only. |
| `OpenGVLab/MVBench` | downloadable with root coordination | MIT, immutable HF revision, exact 17.33 GB metadata; freeze exact reproduction manifest first. |
| `apolinario/decision-index` | downloadable verified for kit only | MIT code at fixed commit; underlying benchmark payloads remain per-source gated. |
| `Sakshi113/MMAU` | downloadable verified for public subset only | Apache-2.0 repo at fixed commit; use public `test-mini`, not blind answers. |

The main blockers are:

| Asset | Blocker |
|---|---|
| OneJev-Data aggregate | Existing admission registry marks the mixed aggregate blocked; per-source license/provenance audit is required before any row-level use. |
| JevEmbed-Data | Mixed upstream source licenses; at least one listed upstream source is non-commercial. Needs row-level whitelist. |
| Open-Jev programmatic data | Exact dataset release/payload and label lineage not pinned. |
| A-OKVQA | Raw images inherit COCO terms; HF mirror license metadata is absent. |
| ScienceQA | CC-BY-SA-4.0 obligations need release/training review. |
| ChartQA | GPL-3.0 dataset obligations need review before model/data release. |
| DocVQA | HF mirror says Apache-2.0, but original document image rights need confirmation. |
| Multimodal-Mind2Web | OpenRAIL and website screenshot terms need review. |
| AndroidControl | Official repo found; storage object bytes and dataset terms still need capture. |
| GUI-Odyssey | Original HF source requires access; derivative metadata does not solve original terms. |
| NExT-QA / STAR / CLEVRER | Labels and raw video media terms are not fully pinned. |
| LLaVA-Video-178K | 1.28 TB known payload, no license in HF API metadata, teacher labels are future generated. |
| Clotho-AQA / AudioCaps / VGGSound | Audio media rights are per-file, form-gated, or dependent on YouTube availability. |
| Common Voice + TTS | Locale/release and TTS engine/model licenses are not selected. |
| JevBench | Exact release, group IDs, and aggregation manifest are still missing. |

## Original Design Requirements Captured

The design’s §5.1 and §6 require:

- Stage 1: 2k then 30k text questions from JevEmbed sampling, Open-Jev programmatic data, and Valen text, with at least 60% hard labels.
- Stage 2: add 40k image/screenshot/chart/document rows: A-OKVQA/ScienceQA 10k, ChartQA/DocVQA 10k, Mind2Web/AndroidControl 10k, Playwright synthetic screenshots and counterfactuals 10k.
- Stage 3: add 40k audio/video/cross-modal rows: Clotho-AQA/AudioCaps 10k, TTS synthetic speech 8k, NExT-QA/STAR/CLEVRER 12k, LLaVA-Video sampled 5k, cross-modal combinations 5k.
- Stage 4: scale to 400k-600k questions by the §5.4 ratio, adjusted by stage 1-3 weaknesses.
- Stage 5: generate 50k variant groups, real-result/programmatic subsets, and 3k-5k calibration rows per modality.
- Stage 6: distill E4B v1.0 outputs to E2B from a 200k sample and produce quantized/server/edge artifacts.
- Stage 0/4/5/6 eval: JevBench, Decision Index, MMAU, MVBench, and the project’s internal dev/holdout gates.

## Ready Versus Blocked Download Plan

Do not start any command below without root coordination if the payload exceeds 100 MB. The commands are concrete next actions after that coordination, not instructions executed in this slice.

| Lane | Asset | Fixed Source | Bytes | Next Command |
|---|---|---:|---:|---|
| Ready-large | Valen General-100k | `Valen-Team/Valen-Training-General-100k@c6128642d53ed5b9799749d82fd1ae2b2a201f0b` | 6,359,697,280 | `huggingface-cli download Valen-Team/Valen-Training-General-100k --revision c6128642d53ed5b9799749d82fd1ae2b2a201f0b --local-dir artifacts/raw/valen-general-100k` |
| Ready-large | Perception Test | `lmms-lab/PerceptionTest@a38dee602c1a8353d8a616e5f5e6a1bf0a7fc1bf` | 65,593,165,514 | Mirror media archives only after public-label scope is frozen. |
| Ready-large | MVBench | `OpenGVLab/MVBench@230a2d4fac8900333c61754641c7a13e069ac9c6` | 17,331,064,311 | Mirror only after exact 14-task Jev-Omni reproduction manifest is written. |
| Ready-small | Decision Index kit | `apolinario/decision-index@f8552a13b643d39f61185f5951bb5fed16bedf51` | repo-sized | Clone kit; enumerate underlying datasets and block any account/terms-gated source. |
| Ready-small | MMAU public subset | `Sakshi113/MMAU@110127f54c0dfba3faa5ec9feee4a7e4148679c5` | repo-sized plus audio payload | Pin public `test-mini`; never use blind 9k answers locally. |

Everything else is blocked for one of three reasons: source license is unclear or restrictive, download requires account/form access, or the aggregate/teacher label provenance is not enough for this project’s constraints.

## Per-Source Inventory

| Asset | Stage | Official Source | Revision / Version | License State | Metadata / Size | Label Origin | Media Availability | Status |
|---|---:|---|---|---|---:|---|---|---|
| OneJev-Data aggregate | reference | HF `OmniJev/OneJev-Data` | `c991b20e70359acb4448147c2ce99eecccd08ac4` | per-source / other; current registry blocked | 27 files, 17,741,763,803 bytes | mixed aggregate; preview rows do not prove full lineage | parquet shards available but not safe wholesale | provenance blocked |
| JevEmbed-Data | 1 | HF `HIT-TMG/JevEmbed-Data` | `fc7f44305315082d7d554d572024861e438aeffa` | Mixed; declared card omits at least one NC upstream called out by processing report | 15 files, 388,681,000 bytes | Mixed original/converted labels; proposed OneJev soft labels are forbidden unless replaced | Text/json | license blocked |
| Open-Jev programmatic | 1 | GitHub `Zefan-Cai/Open-Jev` | `bd4118882f733574a3250a4b65fe4d884130c08b` | MIT repo, dataset payload unresolved | repo metadata only | Expected programmatic/native decision labels | No media expected | provenance blocked |
| Valen General-100k | 1-3 | HF `Valen-Team/Valen-Training-General-100k` | `c6128642d53ed5b9799749d82fd1ae2b2a201f0b` | Apache-2.0 | 20 files, 6,359,697,280 bytes | Original one-hot decision labels | `assets.zip` 6.01 GB | downloadable with root coordination |
| A-OKVQA | 2 | GitHub `waelbou3/aokvqa-dataset`; HF `HuggingFaceM4/A-OKVQA` | GitHub `42e8c44...`; HF `d1b0efa...` | Repo Apache-2.0; COCO image terms separate | 6 HF files, 1,323,810,597 bytes | Crowdsourced MC VQA labels | HF parquet media / COCO dependency | license blocked |
| ScienceQA image subset | 2 | GitHub `lupantech/ScienceQA`; HF `derek-thomas/ScienceQA` | GitHub `2cbf831...`; HF `f18b0a7...` | CC-BY-SA-4.0 on HF mirror | 9 files, 626,493,224 bytes | Original MC labels | HF parquet media | license blocked |
| ChartQA | 2 | HF `HuggingFaceM4/ChartQA` | `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5` | GPL-3.0 | 7 files, 964,098,758 bytes | Original answer truth plus future distractors | HF parquet charts | license blocked |
| DocVQA | 2 | HF `lmms-lab/DocVQA` | `539088ef8a8ada01ac8e2e6d4e372586748a265e` | HF Apache-2.0; original document rights unverified | 58 files, 12,100,689,020 bytes | Original document QA answers plus future distractors | HF parquet document images | license blocked |
| Multimodal-Mind2Web | 2 | HF `osunlp/Multimodal-Mind2Web` | `1b4c6a8cf9f77b7a5e0d641959935c80c4a05889` | OpenRAIL | 49 files, 13,577,379,722 bytes | Next-action demonstrations | Screenshot parquet | license blocked |
| AndroidControl | 2 | GitHub `google-research/google-research/android_control` | `0a1b0fe7298a58bca6dcb47112379d829bc0c0a5` | Repo Apache-2.0; dataset terms not captured | 15,283 demos per official README | Human demonstrations | GZIP TFRecord screenshots/accessibility trees | license blocked |
| GUI-Odyssey | 2 | GitHub `OpenGVLab/GUI-Odyssey`; HF `hflqf88888/GUIOdyssey` | GitHub `5cdf76d...` | not declared; HF derivative says `other` | derivative reports 187 GB / 267,054 rows | Human GUI demonstrations | Original HF source unauthorized here | account/form blocked |
| Playwright synthetic pages | 2 | project generated | future | project-owned if templates/assets are clean | future | Programmatic truth | generated screenshots | future generated |
| NExT-QA | 3 | GitHub `doc-doc/NExT-QA` | `2432e9724f88ed9f40010e2989f104570a91de4e` | MIT labels/code; raw video separate | repo metadata only | Original video QA labels | raw video unresolved | license blocked |
| STAR video QA | 3 | paper/project still unresolved | not pinned | not verified | unknown | Original labels if found | raw media unresolved | provenance blocked |
| CLEVRER | 3 | `https://clevrer.csail.mit.edu/` | page version not immutable | no license found in inspected page | official split links, bytes not captured | Synthetic physical reasoning labels | official video downloads | license blocked |
| Perception Test | 3/eval | HF `lmms-lab/PerceptionTest` | `a38dee602c1a8353d8a616e5f5e6a1bf0a7fc1bf` | MIT | 7 files, 65,593,165,514 bytes | Public MC labels where available | video/audio zips | downloadable with root coordination |
| LLaVA-Video-178K | 3 | HF `lmms-lab/LLaVA-Video-178K` | `6d8c562dc26d70042a0d9704d1cae58c94b89098` | not declared | 302 files, 1,281,048,393,477 bytes | Future allowed-teacher soft labels required | huge video tarballs | license blocked |
| Clotho-AQA | 3 | Zenodo DOI `10.5281/zenodo.6473207` | DOI record | MIT Q/A CSV; per-file Freesound audio licenses | 1,991 audio files; split CSVs | Crowdsourced audio QA | audio zip with per-file licenses | license blocked |
| AudioCaps | 3 | GitHub `cdjkim/audiocaps`; project site | `d004db3ea1b01cf4fd0347dd8d27db90cadc8809` | MIT code; raw media via form/YouTube | train 49,838 / val 495 / test 975 rows | Human captions | form-gated raw audio/video | account/form blocked |
| VGGSound | 3 | GitHub `hche11/VGGSound` | `9bbc7491c054a203e53584c0065df336e1322bb7` | CC-BY-4.0 metadata; original video copyright remains | CSV metadata; raw bytes unknown | Audio event labels | YouTube clips unstable | license blocked |
| Common Voice + TTS | 3 | Mozilla Common Voice release plus chosen TTS | HF stub `11dc883...` | likely CC0 for CV; TTS unknown | locale payload not selected | Programmatic synthetic truth | generated after model selection | provenance blocked |
| MMAU | 0/eval | GitHub `Sakshi113/MMAU` | `110127f54c0dfba3faa5ec9feee4a7e4148679c5` | Apache-2.0 repo | public subset manifest needed | Official public labels | audio files to pin | downloadable verified |
| MVBench | 0/eval | HF `OpenGVLab/MVBench` | `230a2d4fac8900333c61754641c7a13e069ac9c6` | MIT | 41 files, 17,331,064,311 bytes | Official labels | video files | downloadable with root coordination |
| JevBench | 0/eval | exact release missing | not pinned | not verified | target known: 195 groups / 231 decisions | Official labels if found | unknown | provenance blocked |
| Decision Index | 0/eval | GitHub `apolinario/decision-index` | `f8552a13b643d39f61185f5951bb5fed16bedf51` | MIT kit; underlying datasets separate | kit metadata | Benchmark labels | underlying assets mixed | downloadable kit only |
| Internal dev/calibration/holdout | 0/4/5/6 | external isolated process | future | project-controlled or per-source | dev 500 per modality; holdout 1k per modality; calibration 3k-5k per modality | real human/real-result labels | externally controlled | externally controlled |
| Future generated derivatives | 2-6 | project generated | future | inherits all sources and generators | variants, counterfactuals, U rows, cross-modal combos, E2B distill, quant packages | programmatic/teacher/real result | generated | future generated |
| Optional 5b environments | 5b | Valen game, MiniWoB++, WebArena, Playwright synthetic | separate | mixed | separate optional lane | final task success/reward | environment assets | future generated |

## Sampling and Split Rules

- Train, dev, calibration, and holdout are separated by original sample family, not by transformed row.
- Variants, option permutations, rewrites, counterfactuals, and U examples stay in the same pool as their source family.
- Public benchmarks are eval-only unless explicitly reclassified by a later data card; benchmark answers must not contaminate training.
- Internal sealed holdout content is only read by an isolated gate; development agents receive aggregate pass/fail reports.
- Teacher soft labels are never primary truth when programmatic or dataset labels exist.
- Any future teacher-generated labels must use allowed teachers and record teacher revision, prompt/template, media hashes, and output hashes.

## Concrete Next Commands

These are the next commands for legally or procedurally closest sources. Run them only after root coordinates storage/network and the >100 MB boundary.

```bash
huggingface-cli download Valen-Team/Valen-Training-General-100k \
  --revision c6128642d53ed5b9799749d82fd1ae2b2a201f0b \
  --local-dir artifacts/raw/valen-general-100k
```

```bash
huggingface-cli download lmms-lab/PerceptionTest \
  --revision a38dee602c1a8353d8a616e5f5e6a1bf0a7fc1bf \
  --local-dir artifacts/raw/perception-test
```

```bash
huggingface-cli download OpenGVLab/MVBench \
  --revision 230a2d4fac8900333c61754641c7a13e069ac9c6 \
  --local-dir artifacts/raw/mvbench
```

```bash
git clone https://github.com/apolinario/decision-index.git artifacts/raw/decision-index
git -C artifacts/raw/decision-index checkout f8552a13b643d39f61185f5951bb5fed16bedf51
```

```bash
git clone https://github.com/Sakshi113/MMAU.git artifacts/raw/mmau
git -C artifacts/raw/mmau checkout 110127f54c0dfba3faa5ec9feee4a7e4148679c5
```

For all blocked sources, the next command is not a download. It is a metadata or legal/provenance check:

- JevEmbed: build a row-level source whitelist from `MANIFEST.json` and processing report; exclude NC/provenance-unclear rows before shard mirroring.
- OneJev-Data: use `configs/data-admission.json` as deny-by-default; audit original sources one by one and do not mirror all 18 train shards.
- Open-Jev: find the exact official dataset release and label-origin manifest.
- A-OKVQA: verify COCO media terms for training and redistribution.
- ScienceQA: review CC-BY-SA-4.0 obligations.
- ChartQA: review GPL-3.0 obligations for derived artifacts.
- DocVQA: confirm original document-image rights.
- Mind2Web: review OpenRAIL and website screenshot terms.
- AndroidControl: capture official storage object names, bytes, checksums, and dataset terms.
- GUI-Odyssey: obtain access/terms for the original HF source, not a derivative bypass.
- NExT-QA/STAR/CLEVRER: pin exact media payloads and licenses.
- LLaVA-Video-178K: resolve license/provenance before any 1.28 TB mirror.
- Clotho-AQA/AudioCaps/VGGSound: resolve per-file/form/YouTube media availability.
- Common Voice/TTS: choose exact locales and TTS model with compatible license.
- JevBench: locate official release and group-level scoring manifest.

## Verification Evidence

- `configs/project-data-assets.json` validates as JSON.
- Official metadata sampled:
  - HF API returned immutable revisions and file totals for JevEmbed, Valen, A-OKVQA, ScienceQA, ChartQA, DocVQA, Multimodal-Mind2Web, LLaVA-Video-178K, Common Voice stub, MVBench, Perception Test, and OneJev-Data.
  - GitHub API returned immutable commits for A-OKVQA, ScienceQA, AndroidControl, GUI-Odyssey, NExT-QA, VGGSound, AudioCaps, Open-Jev, JevEmbed, Decision Index, and MMAU.
  - Original HTML was parsed with Python `HTMLParser`; no `rg` scan over the full HTML payload was used for requirements extraction.

## Stop Condition

This slice stops at a complete, auditable inventory. It intentionally does not:

- download any >100 MB payload,
- read sealed holdout content,
- admit mixed-license aggregate datasets,
- accept account terms or submit forms,
- use forbidden Jev labels,
- claim full offline readiness while blocked sources remain.
