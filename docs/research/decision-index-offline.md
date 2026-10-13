# Decision Index Public 0.3 Offline Source Preparation

2026-10-14. This slice prepares public Decision Index 0.3 sources for offline research/evaluation staging only. It does not run the suite, does not score a model, does not reproduce the Full/private leaderboard, and does not admit any source to training or commercial release.

## Scope and source code

The pinned source is `apolinario/decision-index` at `f8552a13b643d39f61185f5951bb5fed16bedf51`. Root had already downloaded the MIT source archive, 1,065,000 bytes, SHA-256 `2719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082`. This slice reused that source and copied it to the remote source-audit tree; no duplicate network fetch of the kit source was needed.

The upstream README says public 0.3 has 37 public-index benchmarks plus seven added benchmarks, while the live Full score also includes private tests that this project cannot reproduce. The kit rebuild path was intentionally not run: upstream uses pickle inputs for RouterBench, unsafe archive extraction patterns in places, a gated HLE source, and an unpinned ARC revision in `acquire.py`. The preparation job used fixed-revision git fetches, direct HTTP downloads, and metadata-resolved HF paths instead.

## Remote evidence

Evidence reports:

- `reports/source-manifests/decision-index-public0.3-001.json`
- `reports/source-manifests/decision-index-public0.3-retry-002.json`

Remote raw root: `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3`.

The finished remote report records:

| Field | Value |
| --- | ---: |
| Raw files staged | 5,901 |
| Raw bytes staged | 2,501,210,679 |
| New raw budget | 10,000,000,000 |
| Git sources verified | 21 |
| HTTP sources verified | 11 |
| Git failures | 1 |
| HTTP failures | 3 |
| HF content sources downloaded | 1 verified, retry queue running |

No GPU work, model inference, full rebuild, archive extraction, or pickle loading was run.

## Verified downloads

The remote job verified 21 fixed-revision git source checkouts. Three added-source git fetches failed or were stopped on the constrained remote: `cand-RAGTruth`, `cand-hover`, and `jev-phishing-bench`.

The following direct HTTP sources were downloaded and hash-verified:

- BPoMP parts 1-3, SHA-256 pinned by upstream.
- Banking77 `test.csv` and `categories.json`.
- CLINC150 `data_full.json`.
- `cfcolor.zip`, with zip member path-safety inspection passing.
- Humicroedit task zip, with zip member path-safety inspection passing.
- ChessBench `action_value_data.bag`.
- ESCI examples and products parquet files, verified against the upstream LFS SHA prefixes recorded by the kit.

Failed or intentionally stopped HTTP downloads in the first report were Habermas comparisons, GPQA `dataset.zip`, and the HoVer Wikipedia SQLite DB. Retry queue 002 is still running and should be treated as the current source for final status.

## HF and gated-source status

HF metadata was resolved locally before the remote job. The first report records resolved revisions, file sizes, expected hashes/LFS metadata, and patterns for the non-gated HF sources. The initial content transfers stalled because the full AutoDL bootstrap/proxy environment was not sourced; retry queue 002 sources both `infra/bootstrap/autodl-env.sh` and `/etc/network_turbo` and has started downloading HF content successfully.

At the local snapshot, RouterBench `routerbench_0shot.pkl` was downloaded and hash-verified in quarantine at 99,567,659 bytes. RouterBench `routerbench_5shot.pkl` was actively downloading. No pickle was loaded.

Retry queue 002:

- PID file: `/root/autodl-tmp/lite-omnijev/artifacts/source-audit/public-index/decision-index-retry-002.pid`
- Nohup log: `/root/autodl-tmp/lite-omnijev/artifacts/source-audit/public-index/decision-index-retry-002.nohup.log`
- Live report: `/root/autodl-tmp/lite-omnijev/reports/source-manifests/decision-index-public0.3-retry-002.json`
- Snapshot status: 83 tasks total, 3 done, 1 failed, 1 running, 2.50 GB raw staged, still within the 10 GB cap.

HLE is explicitly blocked: `cais/hle` at `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29` is gated. The report records the expected `data/test-00000-of-00001.parquet` size and LFS metadata, but no content was downloaded and no access bypass was attempted.

RouterBench remains a quarantine-only source because the upstream files are `.pkl`. Future work must convert in an isolated process that never imports or loads untrusted pickles in the normal agent/runtime environment.

## Use boundary

This preparation is source-audit staging. It is not a Decision Index suite pass and cannot be used as evidence that public 0.3 is reproducible yet. The remaining minimum gaps are completing retry queue 002, HLE account/terms authorization, safe pickle conversion for RouterBench, the persistent RAGTruth git failure, and safe archive extraction/normalization under the remote CPU constraints.
