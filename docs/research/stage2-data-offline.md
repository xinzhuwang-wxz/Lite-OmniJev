# Stage 2 offline source audit: ScienceQA, ChartQA, DocVQA, and A-OKVQA media provenance

This slice records physically staged source-audit assets for Stage 2. It separates remote research/source-audit staging from training or release admission. No GPU work was started, no account-gated terms were accepted, and no private/internal holdout content was read.

The committed ledger is [`configs/stage2-data-assets.json`](../../configs/stage2-data-assets.json). The local ignored operational report from the first Stage2 pass remains `artifacts/data/quarantine/stage2-sources/stage2_operational_download_report.json` with SHA-256 `f988a23ed32b379b183097d323f88a71cc3ddc7866981881a16649a5844fb2c8`, 67 files, and 1,137,251,557 bytes. The remote completion report from that pass is copied locally at `artifacts/data/quarantine/stage2-sources/stage2_remote_completion_report.json`, 6,167 bytes, SHA-256 `24d52b682ae7530d90abd3a2d285a4b46b075473539b6481e2a0c4f383d2723a`.

Remote persistent root: `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/stage2-sources`. Root confirmed the actual remote cgroup limits as `memory.max=2147483648` and `cpu.max=50000 100000` (0.5 CPU); host free memory/16 CPU readings are misleading. CPU-heavy verification used `/tmp/omnijev-cpu-heavy.lock`.

## Download outcome

| Source | Physical staging status | Training/release status |
| --- | --- | --- |
| ScienceQA | Remote raw parquets complete and SHA-verified; local 10 MiB partial preserved as `.partial`. | Research/source-audit only. Training/release remains pending because original ScienceQA is noncommercial/sharealike. |
| ChartQA | Remote original `ChartQA Dataset.zip` complete and SHA-verified. | Research/source-audit only. Training/release remains pending GPL and original chart-source obligation review. |
| DocVQA | Metadata only; no original raw mirror. | Blocked pending official DocVQA/RRC login, download-tab terms, and source-license review. |
| A-OKVQA + COCO | A-OKVQA annotations, COCO license metadata, and all A-OKVQA-referenced COCO pixels with known COCO license IDs 1-7 are staged in separated obligation pools. | Research/source-audit only. No restricted pool is training/release admitted here; NC/SA/ND obligations remain downstream decisions. |

The selected Stage2 raw staging total is 5,506,080,944 bytes against the approved 10 GB Stage2 cap. This is 2,161,862,192 bytes from the first verified pass (ScienceQA parquets, ChartQA original zip, and CC-BY/public-domain A-OKVQA/COCO pixels) plus 3,344,218,752 bytes from the remaining A-OKVQA-referenced known-license COCO pools.

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

Official DocVQA pages say the dataset is downloaded from the RRC challenge page, that the user must log in before downloading, and that the Download page terms and conditions must be read first. The DocVQA Task 1 page points users to the challenge page Download tab under “Single Document Visual Question Answering.” The ICDAR 2021 competition page also states that a team must register in the RRC portal and download only after agreeing to the respective dataset terms. Evidence URLs:

- `https://site.docvqa.org/datasets`
- `https://site.docvqa.org/datasets/docvqa`
- `https://iapr.org/archives/icdar2021/index.html%3Fp=29526.html`

Blocked actions remain unchanged:

- Do not mirror `lmms-lab/DocVQA` as if it were original DocVQA provenance.
- Do not accept RRC or DocVQA terms automatically.
- Do not read private test answers or internal holdout payloads.

## A-OKVQA and COCO media provenance

A-OKVQA annotations and official repo files are staged, along with COCO 2017 annotation/image-info metadata. COCO pixels are staged only for A-OKVQA-referenced image IDs joined through official COCO metadata, preserving `image_id`, `license_id`, `coco_url`, `flickr_url`, split, local path, bytes, and observed SHA-256 in the copied reports.

First-pass CC-BY/public-domain subset:

| Metric | Value |
| --- | ---: |
| Manifest rows | 4,039 |
| Unique images expected | 3,832 |
| Unique images downloaded | 3,832 |
| Missing images | 0 |
| Total bytes | 660,409,234 |
| Download report bytes | 2,191,415 |
| Download report SHA-256 | `42522ce63453844027cf4520e1e453c3b327a7b3297220199ae6a4444c9699d8` |

By staged license ID:

| COCO license ID | Meaning | Unique images |
| ---: | --- | ---: |
| 4 | Attribution License / `http://creativecommons.org/licenses/by/2.0/` | 3,820 |
| 7 | No known copyright restrictions / `http://flickr.com/commons/usage/` | 12 |

Remaining known-license obligation pools were staged separately, not mixed into the CC-BY/public-domain pool:

| Pool | COCO license ID | Unique images | Bytes | Rule |
| --- | ---: | ---: | ---: | --- |
| `ccbyncsa` | 1 | 6,744 | 1,133,013,759 | Attribution + noncommercial + sharealike; research raw staging only. |
| `ccbync` | 2 | 3,259 | 544,282,032 | Attribution + noncommercial; research raw staging only. |
| `ccbyncnd` | 3 | 6,280 | 1,055,748,732 | Attribution + noncommercial + no-derivatives; raw unmodified research staging only. |
| `ccbysa` | 5 | 2,358 | 398,879,324 | Attribution + sharealike; research raw staging only. |
| `ccbynd` | 6 | 1,219 | 212,294,905 | Attribution + no-derivatives; raw unmodified research staging only. |

The remaining-pool downloader used two persistent HTTP connections. The S3 size preflight checked all 19,860 remaining unique files with zero failures and predicted 3,344,218,752 bytes. The final remote download/hash report confirms 19,860 downloaded images, 0 missing, 0 bad-size files, and 3,344,218,752 bytes. Unknown or unhandled license rows/images: 0.

Evidence copied back:

- `artifacts/data/quarantine/stage2-sources/aokvqa/aokvqa_coco_remaining_known_license_s3_size_report.json`, 10,772,059 bytes, SHA-256 `684092d63a04d7181e7375aeb42d2add0cecd527b8f82c96d6717a308774bd3d`
- `artifacts/data/quarantine/stage2-sources/aokvqa/aokvqa_coco_remaining_known_license_download_report.json`, 17,690,975 bytes, SHA-256 `aab0cbb756fd3063b57d83810d0d74cb39054342c27e096954ddaa17c44be71d`
- `artifacts/data/quarantine/stage2-sources/logs/aokvqa-coco-remaining-known-license-driver.log`, 13,223 bytes, SHA-256 `317e8d250d3fda24ba6ee5e49b4100c3a48d63e85d72d942fd03a4826f7338c8`
- `artifacts/data/quarantine/stage2-sources/logs/aokvqa-coco-remaining-persistent-driver-progress.log`, 10,715 bytes, SHA-256 `9403ad0a5dcfd016ab70dd34807cadb83e33adb8430cbde45ddde5964967c131`

## Remaining whole-project data blockers by class

Account or terms requiring user action:

- DocVQA original raw data: official pages require RRC portal login, Download-tab terms review, and agreement before download. An agent must not accept those terms or mirror original raw data automatically.
- CommonVoice and HLE: root already made the one authorization request; this slice does not repeat or bypass it.

License or capacity decisions pending:

- ScienceQA raw files are staged for research/source-audit only; NC/SA training and release use remains a decision.
- ChartQA original zip is staged for research/source-audit only; GPL and underlying chart-source obligations remain a decision.
- A-OKVQA/COCO pools are staged separately by COCO license obligation. CC-BY/PD, CC-BY-SA, NC, SA, and ND pools remain research/source-audit assets until downstream training/release decisions are made. ND pools should remain raw/unmodified unless a separate decision changes that.

Network-only or agent-autonomous items:

- Selected Stage2 ScienceQA, ChartQA, and A-OKVQA/COCO public sources are staged and verified on the remote persistent disk within the approved 10 GB Stage2 cap.
- Future public ungated metadata or small source-audit files can be staged by agents when they do not require account terms, new budget, or training/release admission.

## Stop condition

This slice stops with three concrete ready source-audit assets and one explicit blocker:

1. ScienceQA train/test/validation parquets are physically staged on the remote persistent disk and verified against upstream LFS SHA-256 values.
2. ChartQA original zip is physically staged on the remote persistent disk and verified against upstream LFS SHA-256.
3. A-OKVQA/COCO license-separated image pixels are staged on the remote persistent disk with per-image attribution, observed SHA metadata, and explicit obligation pools.
4. DocVQA remains blocked on original-source terms/access.
5. No staged material is training/release admitted by this report alone.
