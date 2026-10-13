# Stage 1 text offline source proof: JevEmbed-Data and Open-Jev

This slice converts the Stage 1 text-source candidates from “pending metadata” to concrete source-audit artifacts for the two requested lines only: `HIT-TMG/JevEmbed-Data` and `ZefanCai/Open-Jev`. It does not grant training admission. It records exact local artifacts, revisions, row-level source/license filters, label-lineage blockers, and commands root can reuse for larger coordinated mirrors.

The machine is CPU-only and no GPU work was started. No single download over 100 MB was started. The local artifacts are under ignored `artifacts/stage1-text/`; the committed source of truth for this slice is [`configs/stage1-text-assets.json`](../../configs/stage1-text-assets.json).

## JevEmbed-Data

Official source: Hugging Face dataset `HIT-TMG/JevEmbed-Data` at revision `fc7f44305315082d7d554d572024861e438aeffa`.

The dataset is a mixed-source release, not a wholesale-admitted training corpus. The official processing report says each row carries `source`, `source_repo`, `source_revision`, `source_row`, and `license`. The report lists 1,667,639 total questions across Choice, Score, and Noul, but this slice only admits source-audit staging for rows whose local shard metadata matches a clean whitelist.

Downloaded evidence:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `artifacts/stage1-text/jevembed/README.md` | 5,476 | `8714dd09834dfee34eb8158c5e08c0f5df80092a76ac408fdd9cbfa54c7cac0e` |
| `artifacts/stage1-text/jevembed/MANIFEST.json` | 2,997 | `cb7bfce8b2d174ef2ab7f8cbbbd9302ab8dd661633e0f9d1771ea220b3ac7a7e` |
| `artifacts/stage1-text/jevembed/docs__PROCESSING_REPORT.md` | 5,758 | `d5ab9633972e1da41670eb756ee7451599d7a59667787d93ee29f7cc429cd017` |
| `artifacts/stage1-text/jevembed/shards/train-00000-of-00009.parquet` | 6,315,288 | `f875388b1d699643869c97aa444452c53b09d53821039761da4bdf68377b5a97` |
| `artifacts/stage1-text/jevembed/shards/train-00001-of-00009.parquet` | 8,124,251 | `ac3c58f01426d10b3b71d11df9ab11e58f539fc121fb856286074a162098f1ad` |
| `artifacts/stage1-text/jevembed/shards/train-00002-of-00009.parquet` | 8,636,225 | `8cd78225ee905e02c87d005612487c661b86c978c6f4e07ed06f40bc9921b6e8` |

The three checked shards contain 600,000 locally stageable rows:

| Source | License | Local rows | Label lineage result |
| --- | --- | ---: | --- |
| `openjev` | CC0-1.0 | 261,113 | Open-Jev native programmatic/synthetic Choice/Score/Noul labels; no forbidden JEV-27B/JEV-27B-VL/API label evidence in the checked shards. |
| `openjev_v11` | CC0-1.0 | 30,720 | The JevEmbed processing report describes these as selected original controls; local rows carry CC0-1.0. |
| `procedural_jev` | Apache-2.0 | 308,167 | JevEmbed project-original procedural Jev decisions with rules checked. |

The official processing report gives larger totals for the same candidate whitelist: `openjev` 276,249 rows, `openjev_v11` 35,840 rows, `procedural_jev` 399,000 rows, and JevEmbed generated cases v1/v2 21,000 rows. The extra rows remain source-audit candidates, but not all were locally downloaded in this slice.

Blocked or excluded from this slice:

- `Jev Distill Corpus v3`, `System One 270M`, and other teacher-labeled subsets remain blocked until their row-level label lineage proves they are not prohibited JEV/API labels.
- PKU-SafeRLHF and BeaverTails rows appear under non-commercial source licenses in the processing report and are outside this Stage 1 training-admission slice.
- CC-BY or MIT third-party rows outside the Jev/Open-Jev/procedural whitelist may be downloadable for research/source audit, but attribution, sharealike, and release/training decisions remain pending.
- The full JevEmbed release is 388,681,000 bytes, and `data/train-00005-of-00009.parquet` is 161,346,879 bytes. That shard was not downloaded because it exceeds the 100 MB coordination limit.

Root can mirror the under-threshold JevEmbed source-audit subset with:

```bash
huggingface-cli download HIT-TMG/JevEmbed-Data --revision fc7f44305315082d7d554d572024861e438aeffa --include 'README.md' 'MANIFEST.json' 'docs/PROCESSING_REPORT.md' 'data/train-00000-of-00009.parquet' 'data/train-00001-of-00009.parquet' 'data/train-00002-of-00009.parquet' 'data/train-00003-of-00009.parquet' 'data/train-00004-of-00009.parquet' 'data/train-00006-of-00009.parquet' 'data/train-00007-of-00009.parquet' 'data/train-00008-of-00009.parquet' 'data/test-00000-of-00001.parquet' --local-dir artifacts/raw/jevembed-data
```

Coordinate before mirroring the one over-threshold shard:

```bash
huggingface-cli download HIT-TMG/JevEmbed-Data --revision fc7f44305315082d7d554d572024861e438aeffa --include 'data/train-00005-of-00009.parquet' --local-dir artifacts/raw/jevembed-data
```

## Open-Jev

Official data source: Hugging Face dataset `ZefanCai/Open-Jev` at revision `c67699e13d0ae25e35b77165a4b6b079bedc8aba`.

Official generator/source-code repo: GitHub repository `Zefan-Cai/Open-Jev` at commit `bd4118882f733574a3250a4b65fe4d884130c08b`, MIT licensed.

The downloaded `release-v2-redistributable` parquets cover all five published splits. The data license file says original generated Open-Jev records are CC0, while upstream wording, sources, game assets, model weights, and code are not relicensed by that dedication. That means the release can be staged for source audit, but training/release admission still needs row-level source checks for rows whose text was inspired by external docs/evals.

Core downloaded evidence:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `artifacts/stage1-text/open-jev/LICENSE-DATA` | 276 | `85798eb9189ae9b17b12903ba3c1154e8e96e686d40de4dd14f42d7ad9a9eb09` |
| `artifacts/stage1-text/open-jev/LICENSE-CODE-MIT` | 1,078 | `ef2f6f6db605a2f611bc2d037c59276f615cdea76ac6e7551a28f71db890c9d2` |
| `artifacts/stage1-text/open-jev/README.md` | 20,866 | `8f7d66430c63881d042a88165f4ad669e6b7c6783b69f6735b8dfedf6d58f683` |
| `artifacts/stage1-text/open-jev/REPRODUCTION.md` | 7,057 | `f0c1412446f2ed528dc6eecc95a68aa19362e775d1d609f982ea91510e4eb215` |
| `artifacts/stage1-text/open-jev/THIRD_PARTY_NOTICES.md` | 2,533 | `36350b78946d29f443cb03e276eed3934aca01079de769ed4655b93ad5031c4a` |
| `artifacts/stage1-text/open-jev/export-manifest.json` | 153,866 | `141f26d8ef8f0d8df559293b216155dc5f437551ec0e1915aedb2f2f09dcace0` |
| `artifacts/stage1-text/open-jev-github/LICENSE` | 1,078 | `ef2f6f6db605a2f611bc2d037c59276f615cdea76ac6e7551a28f71db890c9d2` |
| `artifacts/stage1-text/open-jev-github/PUBLIC_RELEASE.json` | 537,391 | `f0b94553eaf89ee119091535e6f299c427bf41fd41e072c711ee266a779261bb` |

Release-v2 split artifacts:

| Split artifact | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `train-00000-of-00001.parquet` | 79,116 | 5,026,051 | `a76be013b2986a97e4d61faad4fd5f0a6ddb934760d5dcf3161aaa118909b6bf` |
| `calibration-00000-of-00001.parquet` | 4,672 | 381,769 | `74cf0064f68dcccbcedd4849d3b347f4c900bdd5329a78ec249fcdf24f89b554` |
| `validation-00000-of-00001.parquet` | 3,723 | 314,888 | `5563764b2d92306e2bfdfd883edfee28a82acf36699f6533c0bd51978b784e59` |
| `test-00000-of-00001.parquet` | 10,356 | 734,274 | `05edb50da60abb087328e353b201d83b745e01393b4a43b5320b90720a8132ff` |
| `ood-00000-of-00001.parquet` | 15,701 | 934,037 | `ba8cb3f95b9121aa73e4b997ac9c350e2ec4dddb972cb5c3086da911f0deb9ef` |

The train split has 18,286 Choice rows, 11,923 Score rows, and 48,907 Noul rows. The train source counts are:

| Source | Train rows |
| --- | ---: |
| `painting-geometry-v1` | 23,552 |
| `snake-v1` | 12,508 |
| `workflow-controls-v1/security_incidents` | 8,874 |
| `vizdoom-basic-v1` | 6,354 |
| `reasoning-control-v1` | 5,442 |
| `workflow-controls-v1/invoice_processing` | 5,370 |
| `workflow-controls-v1/customer_service` | 4,392 |
| `customer-control-v1` | 4,206 |
| `workflow-controls-v1/agent_trace_observability` | 3,780 |
| `tic_tac_toe-v1` | 3,264 |
| `tile_platformer-v1` | 1,346 |
| `trex_runner-v1` | 28 |

The checked parquets and release metadata identify labels as programmatic, synthetic, deterministic local-policy, minimax, BFS, physics baseline, or exact programmatic oracle labels. I found no evidence in the checked Open-Jev release-v2 files that labels came from JEV-27B, JEV-27B-VL, or Jev API outputs.

Open-Jev rows still need two levels of handling:

- Clean programmatic source-audit candidates: `painting-geometry-v1`, `snake-v1`, `reasoning-control-v1`, `tic_tac_toe-v1`, `tile_platformer-v1`, `trex_runner-v1`, and `vizdoom-basic-v1`.
- Source-audit candidates needing wording-rights review before training/release: `workflow-controls-v1/*` and `customer-control-v1`. Their targets are synthetic local policies, but metadata references TypeSafe docs/evals as task-form inspiration. That is not evidence of forbidden API labels, but it is enough to keep training admission pending.

Root can mirror the Open-Jev release-v2 subset with:

```bash
huggingface-cli download ZefanCai/Open-Jev --revision c67699e13d0ae25e35b77165a4b6b079bedc8aba --include 'data/release-v2-redistributable/*' 'LICENSE-DATA' 'LICENSE-CODE-MIT' 'README.md' 'REPRODUCTION.md' 'THIRD_PARTY_NOTICES.md' 'export-manifest.json' --local-dir artifacts/raw/open-jev
```

Root can clone the fixed MIT generator/source repository with:

```bash
git clone https://github.com/Zefan-Cai/Open-Jev.git artifacts/raw/open-jev-code
git -C artifacts/raw/open-jev-code checkout bd4118882f733574a3250a4b65fe4d884130c08b
```

## Open-Jev v1.1

Remote metadata was verified for `ZefanCai/Open-Jev-v1.1` at revision `10ad6888333fa97f8c948192797bad3de3040802`. The release declares `source-dependent-cc0-1.0-and-cc-by-4.0`, has 23 files, and totals 89,758,571 bytes. It is below the 100 MB threshold but was not downloaded in this slice because it is source-dependent and needs attribution/release review before admission.

If root wants the full v1.1 source-audit mirror, use:

```bash
huggingface-cli download ZefanCai/Open-Jev-v1.1 --revision 10ad6888333fa97f8c948192797bad3de3040802 --local-dir artifacts/raw/open-jev-v1.1
```

## Resulting stage-1 rule

For immediate offline source audit, Stage 1 can use the checked artifacts as follows:

1. JevEmbed rows are only stageable when the row-level `source` and `license` match the whitelist in `configs/stage1-text-assets.json`.
2. Open-Jev release-v2 files are fully downloaded for source audit, but training/release admission must keep source groups with TypeSafe-doc/eval wording inspiration pending until wording rights are reviewed.
3. No mixed OneJev-Data or JevEmbed aggregate should be consumed wholesale. The source filter must operate at row level before any training manifest is produced.
