# CLEVRER official offline source preparation

Generated for branch `codex/stage3-source-audit` in `/tmp/lite-omnijev-stage3-source-audit`.

This slice prepared only the official CLEVRER source files approved by root. It did not accept terms, did not open GPU, did not read private or hidden answers, did not extract archives, and did not run model inference. Download and CRC verification are offline source preparation only; root still owns shared ledger integration and any training admission decision.

## Official source and license

The official CLEVRER project page lists the train/validation/test videos, train/validation annotations, train/validation/test public questions, object masks/attributes (`derender_proposals.zip`), and README downloads at `data.csail.mit.edu` ([CLEVRER project page](https://clevrer.csail.mit.edu/)). The official README states the dataset is under CC0 ([official README.txt](https://data.csail.mit.edu/clevrer/README.txt)).

The same host supports HTTPS with a valid client-verified certificate, so the download used HTTPS URLs. The upstream publishes HTTP metadata including `Content-Length`, `ETag`, and `Last-Modified`, but no upstream SHA256 or other cryptographic hash was available. The local SHA256 values below are therefore local integrity receipts, not upstream hash matches.

## Remote receipt

Remote root:

`/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official`

Payload bytes from official `Content-Length` values: `25,489,948,056`.

Remote disk usage including logs and range-recovery scratch: `26,163,785,930` bytes, under the approved 27 GB cap.

ZIP CRC log:

`/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official/logs/full-zipcrc-20261013T2108Z.log`

ZIP CRC result: `FULL_ZIPCRC_OK` for train/validation/test videos, train/validation annotations, and `derender_proposals.zip`.

## Downloaded files

| File | Bytes | Local SHA256 | ZIP CRC |
| --- | ---: | --- | --- |
| `videos/train/video_train.zip` | 12,354,893,389 | `c7401bc86f7ff230bda3b193619f41a79828bed1f0890948b16dd57c89dfdd3f` | pass |
| `annotations/train/annotation_train.zip` | 156,761,889 | `e7f9185ad3dde8139b839b00f0f75483374dd0325a3cbec2dac882078497784c` | pass |
| `questions/train.json` | 86,651,938 | `11181da673d223f41fb596aacfbbd3ff83d39af7f09a3210549e98cb283714b4` | not zip |
| `videos/validation/video_validation.zip` | 6,206,059,321 | `07f6d48bd708c80428684fcf17a7874f6c01fbb82a59b2f79ede199a63db7aae` | pass |
| `annotations/validation/annotation_validation.zip` | 78,386,966 | `d89e74714dff7a1117fd113b54af99fbbc51583ebd742c788c484d708eb426d2` | pass |
| `questions/validation.json` | 43,327,533 | `fdf841678a476655b906165e6ee4b0ed1516785c5ab927c8d0a4614e4e27e10d` | not zip |
| `videos/test/video_test.zip` | 6,149,404,859 | `37b8900f5d01e5dc477747af31b1985b2c849b3e5bf9dacb11e2908811a94aab` | pass |
| `questions/test.json` | 17,649,911 | `551716ab736a956dbde17dd0df5e2defdccc580f8f1b78df972fe30e0e074a24` | not zip |
| `derender_proposals.zip` | 396,807,825 | `68ea4785f73ed23e5b84a8b17da2c1da47429ddaed9407c5ad949af08683b7f3` | pass |
| `README.txt` | 4,425 | `a4838cc68b79ba049b54f95def1714cdbbd1380193d7d2a5d1a85440ac862e7b` | not zip |

## Operation notes

The main downloader used at most three concurrent HTTPS `curl` workers with `--continue-at -` and `--retry 0`. Several official streams closed early with curl `rc=18`; each was resumed from the actual partial byte count. The final slow file, `derender_proposals.zip`, advertised `Accept-Ranges: bytes`; after every other CLEVRER file completed, I stopped the single slow derender stream at 120,529,767 bytes and recovered the remaining bytes with three bounded HTTPS range workers. The assembled derender file matched the official `Content-Length`, received a local SHA256, and passed ZIP CRC.

Two operation-script mistakes happened before useful payload mutation: the first full downloader and the first range-recovery script each failed on a Bash same-line `local` declaration under `set -u`. The corrected scripts were rerun; the first failure transferred no files, and the range failure transferred no additional bytes beyond the already preserved partial.

## Remaining integration boundary

There is no remaining CLEVRER download or CRC blocker in this slice. The remaining work is integration by root into the shared asset ledger and any downstream project admission checks. This report must not be used to claim upstream hash verification, because the upstream did not publish cryptographic hashes.
