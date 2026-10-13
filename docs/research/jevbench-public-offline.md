# JevBench public source preparation

The required decision benchmark is [Benchmark Heaven's JevBench](https://github.com/fstandhartinger/jevbench/tree/7625ecb1b075cfe4d09d34141d353325825eea1d), not the separately named metamorphic coherence benchmark at `JevBench/jevbench` on Hugging Face. Source revision and exact files are fixed in `configs/jevbench-public-assets.json`.

Downloaded public files match the original publisher's manifest hashes:

| Public split | Decisions | Literal `group` values |
| --- | ---: | ---: |
| original | 72 | 36 |
| easy | 48 | 1 |
| hard | 111 | 111 |
| Total | 231 | 148 |

The manifest, MIT license, hard-tier authoring method, and scoring/metrics code are also staged. Eight files were copied to the persistent instance and independently passed `sha256sum --check`. Remote path: `artifacts/data/research-eval/jevbench/7625ecb1b075cfe4d09d34141d353325825eea1d`. Evidence: `reports/jevbench-public-source-verification.json`.

The public hard method states that its labels were authored and cross-reviewed before benchmarked systems ran. We fetched no Jev API responses, per-model prediction artifacts, held-out items, or imported private/judge payloads. These public benchmark files are evaluation candidates and remain outside training admission.

The fixed teacher card reports 195 matched groups / 231 decisions but does not publish its selected IDs and grouping implementation. Literal `group` counting gives 148 because all easy rows share `group=easy`. Treating easy rows as separate scenarios would produce 36+48+111=195 groups, but that arithmetic is a candidate interpretation, not independent evidence of the teacher's actual protocol. Freeze grouping and exact source IDs before GPU scores are compared; do not select aggregation based on which score fits the card.

This source preparation completes the public file download gap. It does not pass teacher quality, probability agreement, calibration or private benchmark gates.
