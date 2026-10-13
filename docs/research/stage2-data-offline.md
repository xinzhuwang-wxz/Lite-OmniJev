# Stage 2 offline source audit: ScienceQA, ChartQA, DocVQA, and A-OKVQA media provenance

This slice records physically staged source-audit assets for Stage 2. It separates remote research/source-audit staging from training or release admission. No GPU work was started, no account-gated terms were accepted, and no private/internal holdout content was read.

The committed ledger is [`configs/stage2-data-assets.json`](../../configs/stage2-data-assets.json). The local ignored operational report remains `artifacts/data/quarantine/stage2-sources/stage2_operational_download_report.json` with SHA-256 `f988a23ed32b379b183097d323f88a71cc3ddc7866981881a16649a5844fb2c8`, 67 files, and 1,137,251,557 bytes. The remote completion report is copied locally at `artifacts/data/quarantine/stage2-sources/stage2_remote_completion_report.json`, 6,167 bytes, SHA-256 `24d52b682ae7530d90abd3a2d285a4b46b075473539b6481e2a0c4f383d2723a`.

Remote persistent root: `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/stage2-sources`. Root confirmed the actual remote cgroup limits as `memory.max=2147483648` and `cpu.max=50000 100000` (0.5 CPU); host free memory/16 CPU readings are misleading. The completion report was generated under `/tmp/omnijev-cpu-heavy.lock` and avoids re-hashing the full COCO image tree.

## Download outcome

| Source | Physical staging status | Training/release status |
| --- | --- | --- |
| ScienceQA | Remote raw parquets complete and SHA-verified; local 10 MiB partial preserved as `.partial`. | Research/source-audit only. Training/release remains pending because original ScienceQA is noncommercial/sharealike. |
| ChartQA | Remote original `ChartQA Dataset.zip` complete and SHA-verified. | Research/source-audit only. Training/release remains pending GPL and original chart-source obligation review. |
| DocVQA | Metadata only; no original raw mirror. | Blocked pending official DocVQA/RRC terms/access and source-license review. |
| A-OKVQA + COCO | A-OKVQA annotations, COCO license metadata, and filtered COCO image pixels for license IDs 4/7 are staged remotely. | Research/source-audit only. Training/release requires per-image attribution and downstream decision; NC/ND and CC-BY-SA images are not admitted here. |

The lower-bound remote bytes newly staged for this slice are 2,161,862,192 bytes: 626,082,086 for ScienceQA parquets, 875,370,872 for ChartQA original zip, and 660,409,234 for the filtered A-OKVQA/COCO image subset. This excludes pre-existing metadata, logs, and preserved diagnostic partials.

## ScienceQA

Official project evidence: `lupantech/ScienceQA` says the code is MIT and the ScienceQA dataset is Creative Commons Attribution-NonCommercial-ShareAlike 4.0. The public HF mirror `derek-thomas/ScienceQA` at revision `f18b0a70359ebfb41f658fd564208d0355b013f4` declares `cc-by-sa-4.0`, is not gated, and has 626,493,224 bytes across 9 files. This slice stages the three raw parquet split files only, not the tutorial images archive.

Remote verified raw files:

| Split | Remote path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| test | `scienceqa/hf/data/test-00000-of-00001-f0e719df791966ff.parquet` | 122,386,007 | `235e92d1f30155266df76bc9f28fc6e4fcb6bec2c6a8c7d67f9086ea6b392a84` |
| train | `scienceqa/hf/data/train-00000-of-00001-1028f23e353fbe3e.parquet` | 377,460,993 | `62c90a28e3fb1bc0ad7bbcab1ac62b483ae6758291a655944d8f494bf6445745` |
| validation | `scienceqa/hf/data/validation-00000-of-00001-6c7328ff6c84284c.parquet` | 126,235,086 | `e66b5215ce71e748ade4ee629bc14fbf86762071d355a4fcd831581cc04d72d8` |

The earlier broken local transfer was preserved as `scienceqa/hf/data/test-00000-of-00001-f0e719df791966ff.parquet.partial`, 10,485,760 bytes, SHA-256 `c705e75a6c54c48f6e6a4884f251e3934c56a8cfc1936a1c90a67e7a1192d0c6`. Remote diagnostic copies from failed train attempts were also preserved; the remote completion report records their observed hashes. The `sibling.blob_id` values from HF are treated as LFS pointer Git SHA values, not raw-file SHA values; raw verification uses the LFS SHA-256 values above.

Relevant logs copied back under `artifacts/data/quarantine/stage2-sources/logs/`:

- `scienceqa-train-aria2-noalloc.log`
- `scienceqa-validation.log`
- `stage2-remaining-driver.log`

## ChartQA

Official project evidence: `vis-nlp/ChartQA` points the full ChartQA dataset with annotations to `ahmed-masry/ChartQA` on Hugging Face. The HF dataset at revision `af8b6f5c08c95085271561c2a3f9d15f2b5a9031` declares `gpl-3.0`, is not gated, and has immutable LFS metadata for the full original zip and parquet conversion.

Remote verified raw artifact:

| Remote file | Bytes | SHA-256 |
| --- | ---: | --- |
| `chartqa/hf/ChartQA Dataset.zip` | 875,370,872 | `1bf310e5a51101681495c4a24f4f29d22c4f70b52df24e2e4feb0d79cae3c160` |

This download establishes the source-audit mirror for the original zip. It does not decide whether GPL obligations or any underlying chart-source obligations are compatible with training or release.

Relevant log copied back: `artifacts/data/quarantine/stage2-sources/logs/chartqa-original-zip.log`.

## DocVQA

DocVQA remains a verified blocker for this slice. I staged metadata for `lmms-lab/DocVQA@539088ef8a8ada01ac8e2e6d4e372586748a265e`, but did not treat it as original-source provenance. The mirror declares `apache-2.0`, is not gated, and totals 12,100,689,020 bytes, but original DocVQA/competition lineage must be cleared through the official DocVQA/RRC channel before mirroring or training use.

Blocked actions remain unchanged:

- Do not mirror `lmms-lab/DocVQA` as if it were original DocVQA provenance.
- Do not accept RRC or DocVQA terms automatically.
- Do not read private test answers or internal holdout payloads.

## A-OKVQA and COCO media provenance

A-OKVQA annotations and official repo files are staged, along with COCO 2017 annotation/image-info metadata. COCO pixels were staged only from the generated CC-BY/public-domain manifest subset, preserving `image_id`, `license_id`, `coco_url`, `flickr_url`, split, local path, bytes, and observed SHA-256 in `aokvqa/aokvqa_coco_ccby_publicdomain_image_download_report.json`.

Remote verified image subset:

| Metric | Value |
| --- | ---: |
| Manifest rows | 4,039 |
| Unique images expected | 3,832 |
| Unique images downloaded | 3,832 |
| Missing images | 0 |
| Total bytes | 660,409,234 |
| Download report bytes | 2,191,415 |
| Download report SHA-256 | `42522ce63453844027cf4520e1e453c3b327a7b3297220199ae6a4444c9699d8` |

By COCO split:

| COCO split | Unique images |
| --- | ---: |
| train | 2,644 |
| val | 212 |
| test | 976 |

By staged license ID:

| COCO license ID | Meaning | Unique images |
| ---: | --- | ---: |
| 4 | Attribution License / `http://creativecommons.org/licenses/by/2.0/` | 3,820 |
| 7 | No known copyright restrictions / `http://flickr.com/commons/usage/` | 12 |

License IDs 1, 2, 3, and 6 remain excluded because they are NC or NoDerivs variants. License ID 5 remains in separate `*_coco_ccbysa_image_manifest.jsonl` files and is not pixel-staged by this report. License ID 8 is permitted by the filter but had no rows in this A-OKVQA subset.

Relevant logs copied back:

- `aokvqa-coco-ccby-publicdomain-driver-restart.log`
- `aokvqa-coco-ccby-publicdomain-images.log`
- `aokvqa-coco-ccby-publicdomain-aria2-input.txt`

## Stop condition

This slice stops with three concrete ready source-audit assets and one explicit blocker:

1. ScienceQA train/test/validation parquets are physically staged on the remote persistent disk and verified against upstream LFS SHA-256 values.
2. ChartQA original zip is physically staged on the remote persistent disk and verified against upstream LFS SHA-256.
3. A-OKVQA/COCO license-filtered image pixels are staged on the remote persistent disk with per-image attribution and observed SHA metadata.
4. DocVQA remains blocked on original-source terms/access.
5. No staged material is training/release admitted by this report alone.
