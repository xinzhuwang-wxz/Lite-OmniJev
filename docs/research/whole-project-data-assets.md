# Whole-Project Data Asset Inventory

Date: 2026-10-14. Scope: original design stages 0-6 plus explicit separation for optional 5b, research-only sources, internal holdout, and future generated assets. This is an offline-prep ledger, not a training admission approval.

No heldout content was read. No large dataset shard was downloaded. Metadata came from the original design parsed with `HTMLParser`, existing project admission notes, Hugging Face dataset APIs, GitHub repository APIs, and official project pages.

## Decision Summary

The project is not yet ready for a whole-project download sweep. The ledger now separates two decisions: whether bytes can be physically staged for source audit or benchmark evaluation, and whether those bytes are admitted to training/release. Several sources can be staged after root coordination, but training/release admission still has blockers around attribution, share-alike/copyleft obligations, raw media rights, account/form access, and aggregate label provenance.

The strongest immediate download candidates are:

| Asset | Status | Why |
|---|---|---|
| `Valen-Team/Valen-Training-General-100k` | stageable for source audit with root coordination | HF mirror is Apache-2.0 at immutable revision and has exact file hashes, but upstream source/provenance must be audited before training/release. |
| `lmms-lab/PerceptionTest` | benchmark eval staging pending original terms | HF mirror is MIT at immutable revision and has exact file hashes, but original DeepMind/Perception Test terms and public-label scope must be cross-checked before eval media is mirrored. |
| `OpenGVLab/MVBench` | downloadable with root coordination | MIT, immutable HF revision, exact 17.33 GB metadata; freeze exact reproduction manifest first. |
| `apolinario/decision-index` | downloadable verified for kit only | MIT code at fixed commit; underlying benchmark payloads remain per-source gated. |
| `Sakshi113/MMAU` | downloadable verified for public subset only | Apache-2.0 repo at fixed commit; use public `test-mini`, not blind answers. |
| Clotho-AQA | CC0/CC-BY subset stageable for source audit | Zenodo file manifest and license CSV are verified; 1,659 of 1,991 audio files are CC0 or CC-BY 3.0, backing 29,844 QA rows. The 3.14 GB audio archive still needs root coordination and subset extraction. |

The main blockers are:

| Asset | Blocker |
|---|---|
| OneJev-Data aggregate | Existing admission registry marks the mixed aggregate blocked; per-source license/provenance audit is required before any row-level use. |
| JevEmbed-Data | Mixed upstream source licenses; at least one listed upstream source is non-commercial. Needs row-level whitelist. |
| Open-Jev programmatic data | Exact dataset release/payload and label lineage not pinned. |
| A-OKVQA | Raw images inherit COCO terms; HF mirror license metadata is absent. |
| ScienceQA | Physical staging can be planned for audit, but CC-BY-SA-4.0 attribution/share-alike obligations need a training/release decision. |
| ChartQA | Physical staging can be planned for audit, but GPL-3.0 dataset obligations need a training/release decision. |
| DocVQA | HF mirror says Apache-2.0, but original document image rights need confirmation. |
| Multimodal-Mind2Web | OpenRAIL and website screenshot terms need review. |
| AndroidControl | Official repo found; storage object bytes and dataset terms still need capture. |
| GUI-Odyssey | Original HF source requires access; derivative metadata does not solve original terms. |
| NExT-QA / STAR / CLEVRER | Labels and raw video media terms are not fully pinned. |
| LLaVA-Video-178K | 1.28 TB known payload, no license in HF API metadata, teacher labels are future generated. |
| AudioCaps / VGGSound | Audio media rights are form-gated or dependent on YouTube availability. |
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
| Audit-stageable-large | Valen General-100k | `Valen-Team/Valen-Training-General-100k@c6128642d53ed5b9799749d82fd1ae2b2a201f0b` | 6,359,697,280 | `huggingface-cli download Valen-Team/Valen-Training-General-100k --revision c6128642d53ed5b9799749d82fd1ae2b2a201f0b --local-dir artifacts/raw/valen-general-100k`, then audit `data_manifest.json` and each JSONL source/provenance before admission. |
| Eval-stageable-after-terms | Perception Test | `lmms-lab/PerceptionTest@a38dee602c1a8353d8a616e5f5e6a1bf0a7fc1bf` | 65,593,165,514 | Cross-check original DeepMind/Perception Test terms and public-label scope before mirroring media archives. |
| Ready-large | MVBench | `OpenGVLab/MVBench@230a2d4fac8900333c61754641c7a13e069ac9c6` | 17,331,064,311 | Mirror only after exact 14-task Jev-Omni reproduction manifest is written. |
| Ready-small | Decision Index kit | `apolinario/decision-index@f8552a13b643d39f61185f5951bb5fed16bedf51` | repo-sized | Clone kit; enumerate underlying datasets and block any account/terms-gated source. |
| Ready-small | MMAU public subset | `Sakshi113/MMAU@110127f54c0dfba3faa5ec9feee4a7e4148679c5` | repo-sized plus audio payload | Pin public `test-mini`; never use blind 9k answers locally. |
| Audit-stageable-subset | Clotho-AQA CC0/CC-BY subset | Zenodo `10.5281/zenodo.6473207` | audio archive 3,140,677,494 | Download archive only after root coordination and extract only the 1,659 CC0/CC-BY files; exclude NC, Sampling+, blank, and metadata-missing rows from main training/release. |

Everything else is blocked for one of three reasons: source terms are unclear enough that physical staging is not yet recommended, download requires account/form access, or the aggregate/teacher label provenance is not enough for this project’s constraints.

## Per-Source Inventory

| Asset | Stage | Official Source | Revision / Version | License State | Metadata / Size | Label Origin | Media Availability | Status |
|---|---:|---|---|---|---:|---|---|---|
| OneJev-Data aggregate | reference | HF `OmniJev/OneJev-Data` | `c991b20e70359acb4448147c2ce99eecccd08ac4` | per-source / other; current registry blocked | 27 files, 17,741,763,803 bytes | mixed aggregate; preview rows do not prove full lineage | parquet shards available but not safe wholesale | provenance blocked |
| JevEmbed-Data | 1 | HF `HIT-TMG/JevEmbed-Data` | `fc7f44305315082d7d554d572024861e438aeffa` | Mixed; declared card omits at least one NC upstream called out by processing report | 15 files, 388,681,000 bytes | Mixed original/converted labels; OneJev-4B/27B is not forbidden by name, but row-level lineage must prove no forbidden Jev API/JEV-27B/JEV-27B-VL derivation | Text/json | license blocked |
| Open-Jev programmatic | 1 | GitHub `Zefan-Cai/Open-Jev` | `bd4118882f733574a3250a4b65fe4d884130c08b` | MIT repo, dataset payload unresolved | repo metadata only | Expected programmatic/native decision labels | No media expected | provenance blocked |
| Valen General-100k | 1-3 | HF `Valen-Team/Valen-Training-General-100k` | `c6128642d53ed5b9799749d82fd1ae2b2a201f0b` | Apache-2.0 mirror; upstream source/provenance audit required | 20 files, 6,359,697,280 bytes; `assets.zip` SHA-256 `fbe265...9b1`, `train.jsonl` SHA-256 `5762c3...2788b` | One-hot decision labels; row source must be audited via manifest | `assets.zip` 6.01 GB | audit-stageable with root coordination, not training-admitted |
| A-OKVQA | 2 | GitHub `waelbou3/aokvqa-dataset`; HF `HuggingFaceM4/A-OKVQA` | GitHub `42e8c44...`; HF `d1b0efa...` | Repo Apache-2.0; COCO image terms separate | 6 HF files, 1,323,810,597 bytes | Crowdsourced MC VQA labels | HF parquet media / COCO dependency | license blocked |
| ScienceQA image subset | 2 | GitHub `lupantech/ScienceQA`; HF `derek-thomas/ScienceQA` | GitHub `2cbf831...`; HF `f18b0a7...` | CC-BY-SA-4.0 on HF mirror | 9 files, 626,493,224 bytes | Original MC labels | HF parquet media | training/release decision pending |
| ChartQA | 2 | HF `HuggingFaceM4/ChartQA` | `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5` | GPL-3.0 | 7 files, 964,098,758 bytes | Original answer truth plus future distractors | HF parquet charts | training/release decision pending |
| DocVQA | 2 | HF `lmms-lab/DocVQA` | `539088ef8a8ada01ac8e2e6d4e372586748a265e` | HF Apache-2.0; original document rights unverified | 58 files, 12,100,689,020 bytes | Original document QA answers plus future distractors | HF parquet document images | license blocked |
| Multimodal-Mind2Web | 2 | HF `osunlp/Multimodal-Mind2Web` | `1b4c6a8cf9f77b7a5e0d641959935c80c4a05889` | OpenRAIL | 49 files, 13,577,379,722 bytes | Next-action demonstrations | Screenshot parquet | license blocked |
| AndroidControl | 2 | GitHub `google-research/google-research/android_control` | `0a1b0fe7298a58bca6dcb47112379d829bc0c0a5` | Repo Apache-2.0; dataset terms not captured | 15,283 demos per official README | Human demonstrations | GZIP TFRecord screenshots/accessibility trees | license blocked |
| GUI-Odyssey | 2 | GitHub `OpenGVLab/GUI-Odyssey`; HF `hflqf88888/GUIOdyssey` | GitHub `5cdf76d...` | not declared; HF derivative says `other` | derivative reports 187 GB / 267,054 rows | Human GUI demonstrations | Original HF source unauthorized here | account/form blocked |
| Playwright synthetic pages | 2 | project generated | future | project-owned if templates/assets are clean | future | Programmatic truth | generated screenshots | future generated |
| NExT-QA | 3 | GitHub `doc-doc/NExT-QA` | `2432e9724f88ed9f40010e2989f104570a91de4e` | MIT labels/code; raw video separate | repo metadata only | Original video QA labels | raw video unresolved | license blocked |
| STAR video QA | 3 | paper/project still unresolved | not pinned | not verified | unknown | Original labels if found | raw media unresolved | provenance blocked |
| CLEVRER | 3 | `https://clevrer.csail.mit.edu/` | page version not immutable | no license found in inspected page | official split links, bytes not captured | Synthetic physical reasoning labels | official video downloads | license blocked |
| Perception Test | 3/eval | HF `lmms-lab/PerceptionTest` | `a38dee602c1a8353d8a616e5f5e6a1bf0a7fc1bf` | MIT mirror; original terms need cross-check | 7 files, 65,593,165,514 bytes; media SHA-256s captured in JSON | Public MC labels where available | video/audio zips | benchmark eval staging pending original terms |
| LLaVA-Video-178K | 3 | HF `lmms-lab/LLaVA-Video-178K` | `6d8c562dc26d70042a0d9704d1cae58c94b89098` | not declared | 302 files, 1,281,048,393,477 bytes | Future allowed-teacher soft labels required | huge video tarballs | license blocked |
| Clotho-AQA | 3 | Zenodo DOI `10.5281/zenodo.6473207` | Zenodo record 6473207 | MIT Q/A CSV; audio is per-file Freesound license | 6 files; audio archive 3,140,677,494 bytes MD5 `48c2c46...6b9`; 1,659 CC0/CC-BY audio files and 29,844 QA rows stageable for audit | Crowdsourced audio QA | audio zip with per-file licenses | CC0/CC-BY subset audit-stageable, training/release pending |
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

## Existing Human Annotation Candidates

Some public sources may reduce future manual-gate burden if their raw independent human evidence is reusable. This ledger does not admit them as internal holdout, does not claim κ, and does not read holdout content.

| Source | Potential Use | Current Evidence | Still Required |
|---|---|---|---|
| Clotho-AQA | Audio Noul-style gate or calibration candidate | Split CSVs expose answer/confidence; project context notes three annotators for yes/no questions; 1,659 CC0/CC-BY audio files are stageable for source audit. | Raw distinct annotator labels or stable annotator slots, reusable audio subset, attribution packaging, sample-family isolation, and gate-only access. |
| A-OKVQA | Visual QA human-evidence candidate | Crowdsourced VQA source may include multiple direct answers/rationales. | Raw distinct answer records or annotator slots, COCO/image rights, reusable license decision, identity/independence evidence, and no train/dev overlap. |
| ScienceQA image subset | Science/visual QA human-evidence candidate | Original MC labels and explanations exist. | Raw independent label process or annotator records, CC-BY-SA release decision, and sample-family isolation. |

## Concrete Next Commands

These are the next commands for legally or procedurally closest sources. Run them only after root coordinates storage/network and the >100 MB boundary.

```bash
huggingface-cli download Valen-Team/Valen-Training-General-100k \
  --revision c6128642d53ed5b9799749d82fd1ae2b2a201f0b \
  --local-dir artifacts/raw/valen-general-100k
```

Perception Test has exact HF file hashes in the JSON, but no download command is listed here until original DeepMind/Perception Test terms and public-label scope are cross-checked.

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

- JevEmbed: build a row-level source whitelist from `MANIFEST.json` and processing report; exclude NC/provenance-unclear rows and any row proven to use forbidden Jev API/JEV-27B/JEV-27B-VL labels before shard mirroring. Do not exclude OneJev-4B/27B by name alone.
- OneJev-Data: use `configs/data-admission.json` as deny-by-default; audit original sources one by one and do not mirror all 18 train shards.
- Open-Jev: find the exact official dataset release and label-origin manifest.
- A-OKVQA: verify COCO media terms for training and redistribution.
- ScienceQA: stage only with attribution/share-alike tracking; resolve CC-BY-SA-4.0 obligations before training/release.
- ChartQA: stage only with GPL notice tracking; resolve GPL-3.0 obligations before training/release.
- DocVQA: confirm original document-image rights.
- Mind2Web: review OpenRAIL and website screenshot terms.
- AndroidControl: capture official storage object names, bytes, checksums, and dataset terms.
- GUI-Odyssey: obtain access/terms for the original HF source, not a derivative bypass.
- NExT-QA/STAR/CLEVRER: pin exact media payloads and licenses.
- LLaVA-Video-178K: resolve license/provenance before any 1.28 TB mirror.
- Clotho-AQA: root-coordinate `audio_files.zip` only with a subset extraction plan; keep only CC0/CC-BY rows in main source audit.
- AudioCaps/VGGSound: resolve form/YouTube media availability and rights before staging.
- Common Voice/TTS: choose exact locales and TTS model with compatible license.
- JevBench: locate official release and group-level scoring manifest.

## Verification Evidence

- `configs/project-data-assets.json` validates as JSON.
- Official metadata sampled:
  - HF API returned immutable revisions and file totals for JevEmbed, Valen, A-OKVQA, ScienceQA, ChartQA, DocVQA, Multimodal-Mind2Web, LLaVA-Video-178K, Common Voice stub, MVBench, Perception Test, and OneJev-Data.
  - GitHub API returned immutable commits for A-OKVQA, ScienceQA, AndroidControl, GUI-Odyssey, NExT-QA, VGGSound, AudioCaps, Open-Jev, JevEmbed, Decision Index, and MMAU.
  - Zenodo API returned Clotho-AQA file bytes and MD5 checksums; small license/metadata/split CSV files were downloaded and SHA-256 verified. `audio_files.zip` was not downloaded.
  - Clotho-AQA license manifest count: 864 CC0 audio files, 795 CC-BY 3.0, 263 CC-BY-NC 3.0, 63 Sampling+ 1.0, 6 blank-license; CC0/CC-BY subset covers 29,844 QA rows.
  - Original HTML was parsed with Python `HTMLParser`; no `rg` scan over the full HTML payload was used for requirements extraction.

## Stop Condition

This slice stops at a complete, auditable inventory. It intentionally does not:

- download any >100 MB payload,
- read sealed holdout content,
- admit mixed-license aggregate datasets,
- accept account terms or submit forms,
- use forbidden Jev labels,
- claim full offline readiness while blocked sources remain.
