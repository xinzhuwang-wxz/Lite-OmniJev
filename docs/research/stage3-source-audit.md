# Stage3 source audit metadata verification

Generated on 2026-10-14 from branch `codex/stage3-source-audit` in `/tmp/lite-omnijev-stage3-source-audit`.

This audit only prepared small public metadata needed before GPU work. It did not accept terms, did not use credentials, did not download raw video/audio/media payloads, did not read hidden or holdout data, and did not change shared asset ledgers. The mirrored metadata bundle is on the remote host at:

`/root/autodl-tmp/lite-omnijev/source-audit/stage3-source-audit`

Remote verification evidence:

- `find source-audit/stage3-source-audit -type f | wc -l` → `36`
- `du -sb source-audit/stage3-source-audit` → `36239763`
- `sha256sum source-audit/stage3-source-audit/manifest.json` → `bb7fe00a70dd8208de6a0f5470870083c780de389790205d9320a5944e0bcacc`

The local fetch manifest counted 35 mirrored source files excluding the generated manifest, totaling 36,218,377 bytes, under the 50 MB cap.

## Asset findings

### NExT-QA

Prepared pinned repository metadata from `doc-doc/NExT-QA@2432e9724f88ed9f40010e2989f104570a91de4e`: README, MIT LICENSE, train/val/test CSV annotations, and `map_vid_vidorID.json`.

The official README states that train/val/test raw videos are available through a Google Drive link and that QA annotations live under the repository dataset directory. It also states that the videos come from NExTVideo and can be mapped to VidOR IDs via `dataset/nextqa/map_vid_vidorID.json` ([NExT-QA README](https://github.com/doc-doc/NExT-QA/tree/2432e9724f88ed9f40010e2989f104570a91de4e)). The repository license covers code/annotation repository material, but raw video rights remain separate.

Status: metadata prepared; raw video download is blocked on rights verification for NExTVideo/VidOR/Google Drive terms.

User/root action links:

- [NExT-QA repository](https://github.com/doc-doc/NExT-QA)
- [NExT-QA raw video Google Drive link](https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view)
- [VidOR project page](https://xdshang.github.io/docs/vidor.html)

Next precise media method: use the mirrored NExT-QA split CSVs plus `map_vid_vidorID.json` to generate the exact video ID list for any selected sample. Download only those IDs after rights approval.

### CLEVRER

Prepared official metadata from the CLEVRER project page: page HTML, official `README.txt`, code README, and HEAD metadata for all official download URLs. No CLEVRER payload zips or JSON question files were downloaded.

The official CLEVRER README says the dataset is under CC0. The official project page lists train/validation/test videos, annotations, questions, object masks/attributes, and README download links ([CLEVRER project page](https://clevrer.csail.mit.edu/), [official README.txt](http://data.csail.mit.edu/clevrer/README.txt)).

Captured official payload size headers total 25,489,948,056 bytes:

| Official URL | Content-Length |
| --- | ---: |
| `videos/train/video_train.zip` | 12,354,893,389 |
| `annotations/train/annotation_train.zip` | 156,761,889 |
| `questions/train.json` | 86,651,938 |
| `videos/validation/video_validation.zip` | 6,206,059,321 |
| `annotations/validation/annotation_validation.zip` | 78,386,966 |
| `questions/validation.json` | 43,327,533 |
| `videos/test/video_test.zip` | 6,149,404,859 |
| `questions/test.json` | 17,649,911 |
| `derender_proposals.zip` | 396,807,825 |
| `README.txt` | 4,425 |

Status: official license confirmed; download URLs and bytes captured; payloads not downloaded in this slice.

Next precise media method: use `clevrer/download-heads.json` as the exact official URL and byte manifest. If a subset is chosen, select by official split URL before downloading; reserve about 25.49 GB for the full official payload.

### LLaVA-Video-178K

Prepared pinned HF metadata from `lmms-lab/LLaVA-Video-178K@6d8c562dc26d70042a0d9704d1cae58c94b89098`: README and HF API metadata. The full 1.28 TB dataset listed in `configs/project-data-assets.json` was not downloaded.

The HF README declares Apache License 2.0, but its Uses section limits use to academic research and education and warns users to check OpenAI policy for GPT-4 generated data. It also describes included source groups such as academic/youtube splits and imported sources including ActivityNet, PerceptionTest, NExT-QA, and LLaVA-Hound ([LLaVA-Video-178K dataset card](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)).

Status: metadata prepared; commercial training rights are not cleared. Full payload download remains forbidden for this project slice.

User/root action links:

- [LLaVA-Video-178K dataset card](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)
- [Pinned revision tree](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/6d8c562dc26d70042a0d9704d1cae58c94b89098)

Next precise media method: build a deterministic 5k candidate manifest from the pinned HF JSON/README source groups, stratified by source and duration. For every selected row, record the original source dataset/media URL and license before any media download. Do not download the 1.28 TB full repository.

### VGGSound

Prepared pinned repository metadata from `hche11/VGGSound@9bbc7491c054a203e53584c0065df336e1322bb7`: README, `LICENCE.txt`, official CSVs, split CSVs, common label list, AudioSet test subset, and stats.

The repository README describes `data/vggsound.csv` as the official metadata file containing YouTube ID, start seconds, label, and split. The repository license says the dataset is under Creative Commons Attribution 4.0, and the README notes that original video copyright remains with the original video owners ([VGGSound README](https://github.com/hche11/VGGSound/tree/9bbc7491c054a203e53584c0065df336e1322bb7), [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).

Status: metadata/license prepared; raw YouTube media rights and availability remain separate.

Next precise media method: use mirrored `data/vggsound.csv` as the exact media request list: YouTube ID, start second, label, split. Download only selected rows after rights and availability checks.

### AudioCaps

Prepared pinned repository metadata from `cdjkim/audiocaps@d004db3ea1b01cf4fd0347dd8d27db90cadc8809`: README, MIT LICENSE, dataset and dataset2.0 README/CSV caption splits.

The AudioCaps dataset README defines the CSV columns (`audiocap_id`, `youtube_id`, `start_time`, `caption`) and split counts. It also says raw video/audio access requires filling out the project form and waiting for a download link. The repository README says the code and dataset are free for academic purposes only ([AudioCaps repository](https://github.com/cdjkim/audiocaps), [AudioCaps project page](https://audiocaps.github.io/)).

Status: caption metadata prepared; raw audio/video access and terms are blocked on the official form and downstream YouTube/AudioSet terms.

User/root action links:

- [AudioCaps repository](https://github.com/cdjkim/audiocaps)
- [AudioCaps project page](https://audiocaps.github.io/)
- [AudioCaps raw media request form](https://forms.gle/8sFHc3XJ32ERWxqF8)

Next precise media method: use mirrored caption CSVs to build exact YouTube ID/start-time requests. After form approval, download only the selected IDs and record the granted terms.

### GUI-Odyssey

Prepared pinned GitHub metadata from `OpenGVLab/GUI-Odyssey@5cdf76d9bd72034f3cb36bcf2cad77b8dcc81251` plus fixed HF dataset metadata and README from `hflqf88888/GUIOdyssey@61632d0f3f4d51d7e9561ce4f84347dd06b2019d`. Episode JSON/media were not bulk downloaded.

The GitHub README points to the HF dataset and provides the clone entrypoint. The fixed HF API snapshot reports the dataset is public/ungated and its card data declares `cc-by-4.0`; the HF README also includes CC-BY-4.0 licensing text ([GUI-Odyssey GitHub](https://github.com/OpenGVLab/GUI-Odyssey), [GUIOdyssey HF dataset](https://huggingface.co/datasets/hflqf88888/GUIOdyssey)). The pinned GitHub tree inspected for this audit has no `LICENSE` file, so the HF dataset card is the governing mirrored evidence for dataset terms.

Status: metadata prepared; no credential was used. Root/user should decide whether current public HF CC-BY-4.0 terms satisfy project policy before training use.

User/root action links:

- [GUI-Odyssey GitHub](https://github.com/OpenGVLab/GUI-Odyssey)
- [GUIOdyssey HF dataset](https://huggingface.co/datasets/hflqf88888/GUIOdyssey)

Next precise media method: use HF API siblings and `all_annot.json` to derive an exact episode list. Select the required episodes and download only those JSON/screenshots/actions after terms are accepted by the project owner.

## Stop condition

This slice prepared the small official/public source metadata currently needed for stage3 readiness checks and captured remaining legal/account blockers. It did not admit any raw media for training. GPU opening still depends on root integrating this evidence with the remaining project-wide external dependency checklist and any human terms/account decisions.
