# GUI source preparation

The fixed source inventory is `configs/gui-source-assets.json`. Metadata inspection did not parse screenshot payloads, demonstrations, private holdout content or answer keys.

## AndroidControl

The [official README](https://github.com/google-research/google-research/blob/0a1b0fe7298a58bca6dcb47112379d829bc0c0a5/android_control/README.md) links the public GCS bucket and documents GZIP TFRecord episodes. The GCS metadata API exposes 20 data shards plus `splits.json` and `test_subsplits.json`: **49,930,349,992 bytes**. Every object has an exact immutable generation, MD5 and CRC32C in the inventory; the empty-prefix placeholder is excluded. Split ownership must use episode/app/task/category identity, not individual screenshots; official test sub-splits may overlap.

The authors' [NeurIPS supplementary datasheet, section 1.1.6](https://papers.nips.cc/paper_files/paper/2024/file/a79f3ef3b445fd4659f44648f7ea8ffd-Supplemental-Datasets_and_Benchmarks_Track.pdf) states both Apache-2.0 and a non-commercial-use condition. A mirror's unrestricted Apache tag does not resolve that condition. Internal research-only raw staging is allowed within the stated restriction; commercial training/release admission remains unresolved. No account, form or new terms were accepted.

Raw assets are downloading to `/root/autodl-tmp/lite-omnijev/artifacts/data/research-only/android-control`, separate from commercial candidates. The operational downloader pins GCS generation, resumes partial files, verifies upstream MD5/size and records SHA256 before promoting files. Existing mismatches fail without deletion. Report: `reports/android-control-source-download-001.json`; a running report alone does not prove job liveness. The downloader is a source-copy operation, not TFRecord parsing, sample-family allocation, model training or training admission.

## Multimodal-Mind2Web

The [fixed official dataset card](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web/blob/1b4c6a8cf9f77b7a5e0d641959935c80c4a05889/README.md) declares OpenRAIL metadata and states that release is solely for research. The exact license document and its scope must remain attached to any later admission. Generic license metadata cannot establish commercial-use permission.

The card reports 7,775 training actions across 1,009 tasks; public test splits differ by task, website and domain. Training screenshots can have rendering defects; three test splits received visibility/rendering checks. Action-level random splitting would leak task/page families. Screenshots and DOM candidates require joint alignment and target visibility checks before this project's Choice conversion.

The official 49-file inventory is pinned at the same revision: 13,577,379,722 bytes. Raw research-only staging has now started on the no-GPU instance, report `reports/mind2web-source-download-001.json`, immutable upstream SHA256/Git-blob checks before file completion. Raw download and family/split admission are separate operations; a running download is not a downloaded-media claim. Research-only sources must not be relabeled as commercial-ready or silently substituted for unrestricted data.
