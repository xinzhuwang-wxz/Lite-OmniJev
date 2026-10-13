# Stage 2 offline source audit: ScienceQA, ChartQA, DocVQA, and A-OKVQA media provenance

This slice records what is physically staged under `artifacts/data/quarantine/stage2-sources`, what is only exact-remote-ready, and what remains blocked by original-source terms or provenance. It does not grant training or release admission.

The committed ledger is [`configs/stage2-data-assets.json`](../../configs/stage2-data-assets.json). The ignored operational hash report is `artifacts/data/quarantine/stage2-sources/stage2_operational_download_report.json`, with SHA-256 `f988a23ed32b379b183097d323f88a71cc3ddc7866981881a16649a5844fb2c8`, 67 files, and 1,137,251,557 bytes. No GPU work was started.

## Download outcome

| Source | Physical staging status | Training/release status |
| --- | --- | --- |
| ScienceQA | Metadata and tutorial files staged; first large parquet has a 10 MiB partial only. | Blocked until raw mirror completes and noncommercial/sharealike obligations are accepted for this project. |
| ChartQA | HF README and immutable file metadata staged; 875 MB original zip not launched after ScienceQA transfer failure. | Blocked until GPL/original chart-source obligations are reviewed. |
| DocVQA | Metadata only; no original raw mirror. | Blocked pending official DocVQA/RRC terms/access and source-license review. |
| A-OKVQA + COCO metadata | A-OKVQA annotations, official repo docs, COCO train/val annotations, and COCO test image-info metadata staged. | Image pixels blocked until per-image license manifests are used with attribution. |

## ScienceQA

Official project evidence: `lupantech/ScienceQA` says the code is MIT and the ScienceQA dataset is Creative Commons Attribution-NonCommercial-ShareAlike 4.0. The public HF mirror `derek-thomas/ScienceQA` at revision `f18b0a70359ebfb41f658fd564208d0355b013f4` declares `cc-by-sa-4.0`, is not gated, and has 626,493,224 bytes across 9 files.

Completed local artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `scienceqa/scienceqa_hf_api_info.json` | 4,385 | `783631ad6fd384a91962f2f848395eafa60f5b1d439371fcc139532b86de7be0` |
| `scienceqa/hf/README.md` | 10,308 | `3729fae304968ed8b9917b41034412aa142fb4760e2bfdab3dfe67a7fd78a17a` |
| `scienceqa/hf/tutorial/ScienceQA.py` | 5,401 | `a0806fb01130872aecb6ede47cf4fd392b373c993b303ccc6830a1ae9a258d1e` |
| `scienceqa/hf/tutorial/download.sh` | 684 | `621b0720f17935e64992d6ce86621ade6678056b165c52ace28d0453de0b4d17` |
| `scienceqa/hf/tutorial/create_dataset.ipynb` | 392,442 | `67ede9b581b5c5332c1bc067a6e3000daf7d5cfa1662ab22ce149601ba5936e8` |

The raw parquets are exact-remote-ready but not complete locally:

| Remote file | Bytes | Expected SHA-256 |
| --- | ---: | --- |
| `data/test-00000-of-00001-f0e719df791966ff.parquet` | 122,386,007 | `235e92d1f30155266df76bc9f28fc6e4fcb6bec2c6a8c7d67f9086ea6b392a84` |
| `data/train-00000-of-00001-1028f23e353fbe3e.parquet` | 377,460,993 | `62c90a28e3fb1bc0ad7bbcab1ac62b483ae6758291a655944d8f494bf6445745` |
| `data/validation-00000-of-00001-6c7328ff6c84284c.parquet` | 126,235,086 | `e66b5215ce71e748ade4ee629bc14fbf86762071d355a4fcd831581cc04d72d8` |

The attempted raw mirror stalled with HF CLI at zero bytes, then `curl -C -` wrote a 10 MiB partial for the test parquet and stopped with `curl: (18) Transferred a partial file`. An `aria2c` retry failed on DNS resolution for `huggingface.co`. The partial file is not usable:

| Partial file | Local bytes | Partial SHA-256 |
| --- | ---: | --- |
| `scienceqa/hf/data/test-00000-of-00001-f0e719df791966ff.parquet` | 10,485,760 | `c705e75a6c54c48f6e6a4884f251e3934c56a8cfc1936a1c90a67e7a1192d0c6` |

Resume command:

```bash
huggingface-cli download derek-thomas/ScienceQA --repo-type dataset --revision f18b0a70359ebfb41f658fd564208d0355b013f4 --local-dir artifacts/data/quarantine/stage2-sources/scienceqa/hf
```

## ChartQA

Official project evidence: `vis-nlp/ChartQA` points the full ChartQA dataset with annotations to `ahmed-masry/ChartQA` on Hugging Face. The HF dataset at revision `af8b6f5c08c95085271561c2a3f9d15f2b5a9031` declares `gpl-3.0`, is not gated, and has immutable LFS metadata for the full original zip and parquet conversion.

Completed local artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `chartqa/chartqa_hf_api_info.json` | 3,319 | `988befbd769c0b0c27f20737ea3497e950cadf5a6dba4b3144d60436f2ddd011` |
| `chartqa/hf/README.md` | 2,121 | `eacad5aa4aeb6ee7575fe8434a04d24554915d7898ecd1b02f4adb240f0c6814` |

The best first raw artifact to mirror is the original zip:

| Remote file | Bytes | Expected SHA-256 |
| --- | ---: | --- |
| `ChartQA Dataset.zip` | 875,370,872 | `1bf310e5a51101681495c4a24f4f29d22c4f70b52df24e2e4feb0d79cae3c160` |

Resume command:

```bash
huggingface-cli download ahmed-masry/ChartQA --repo-type dataset --revision af8b6f5c08c95085271561c2a3f9d15f2b5a9031 --include 'README.md' 'ChartQA Dataset.zip' --local-dir artifacts/data/quarantine/stage2-sources/chartqa/hf
```

## DocVQA

DocVQA remains a verified blocker for this slice. I staged metadata for `lmms-lab/DocVQA@539088ef8a8ada01ac8e2e6d4e372586748a265e`, but did not treat it as original-source provenance. The mirror declares `apache-2.0`, is not gated, and totals 12,100,689,020 bytes, but original DocVQA/competition lineage must be cleared through the official DocVQA/RRC channel before mirroring or training use.

Completed local metadata:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `docvqa_lmms/docvqa_lmms_hf_api_info.json` | 16,097 | `3d6b7635dcec7b2a9edde982480bd7832630ba1f7559d9dbbf9981fbd11f1707` |
| `docvqa/lmms-lab-hf/README.md` | 2,689 | `372cd2689649a29b4f4d1c9dbf7478d62b8331ab2af0720b5a1f6fa9d4f266c0` |

Blocked actions:

- Do not mirror `lmms-lab/DocVQA` as if it were original DocVQA provenance.
- Do not accept RRC or DocVQA terms automatically.
- Do not read private test answers or internal holdout payloads.

## A-OKVQA and COCO media provenance

A-OKVQA annotations and official repo files are staged, along with COCO 2017 annotation/image-info metadata. COCO image pixels are not staged in this slice. This prevents washing all COCO images with the A-OKVQA code license and gives an explicit per-image license filter for future downloads.

Completed local artifacts:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `aokvqa/aokvqa_v1p0.tar.gz` | 3,661,719 | `3992b488babc0c1147f0def18c7a55274aeeb37ab668cf80226b8f62ee35a8e1` |
| `aokvqa/github/README.md` | 8,189 | `ee2bd89d8675b9cb45d74e4587fe0aa4d1b2275d3f8dbd3f334f9e946ae2ac87` |
| `aokvqa/github/LICENSE` | 11,373 | `3bcf720dc872fe00ef7bf7e07ebc7bc720e71cb6f7b43cb90241b9104f0ab617` |
| `aokvqa/github/load_aokvqa.py` | 378 | `df7fce7d633ef923591663777bbf928725b4a2a1eb81daf74a26ef95c725081c` |
| `coco2017/annotations_trainval2017.zip` | 252,907,541 | `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268` |
| `coco2017/image_info_test2017.zip` | 1,144,034 | `e52f412dd7195ac8f98d782b44c6dd30ea10241e9f42521f67610fbe055a74f8` |
| `aokvqa/aokvqa_coco_license_summary.json` | 6,592 | `e281d5ac71c089faaf568bebb7f300879dcf81213935366c55ebceef03309d7d` |

The COCO license metadata is split into a CC-BY/public-domain-style subset and a separate CC-BY-SA obligation subset:

| COCO license ID | Meaning | Handling |
| ---: | --- | --- |
| 4 | Attribution License / `http://creativecommons.org/licenses/by/2.0/` | CC-BY/public-domain manifest |
| 7 | No known copyright restrictions / `http://flickr.com/commons/usage/` | CC-BY/public-domain manifest |
| 8 | United States Government Work / `http://www.usa.gov/copyright.shtml` | CC-BY/public-domain manifest |
| 5 | Attribution-ShareAlike License / `http://creativecommons.org/licenses/by-sa/2.0/` | Separate sharealike-obligation manifest |
| 1, 2, 3, 6 | NC or NoDerivs variants | Excluded from CC-BY/public-domain subset |

Resulting A-OKVQA image-license manifests:

| Split | Rows | COCO split | CC-BY/public-domain rows | CC-BY-SA rows | Missing image IDs |
| --- | ---: | --- | ---: | ---: | ---: |
| train | 17,056 | train | 2,741 | 1,775 | 0 |
| val | 1,145 | val | 219 | 105 | 0 |
| test | 6,702 | test | 1,079 | 586 | 0 |

Future image downloads must use the generated manifests, not the full COCO image zips:

- `aokvqa_train_coco_ccby_publicdomain_image_manifest.jsonl`
- `aokvqa_val_coco_ccby_publicdomain_image_manifest.jsonl`
- `aokvqa_test_coco_ccby_publicdomain_image_manifest.jsonl`
- the separate `*_coco_ccbysa_image_manifest.jsonl` files only if sharealike obligations are accepted for the target use.

## Stop condition

This slice stops with three concrete ready assets and two explicit blockers:

1. A-OKVQA/COCO media provenance metadata is ready for selective image staging.
2. ScienceQA public metadata is ready, but raw parquets need a resumed mirror.
3. ChartQA has exact remote hashes and a staged README, but the original zip was not launched after network failure.
4. DocVQA remains blocked on original-source terms/access.
5. No staged material is training/release admitted by this report alone.
