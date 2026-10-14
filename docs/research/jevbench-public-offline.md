# JevBench public source preparation

The required decision benchmark is [Benchmark Heaven's JevBench](https://github.com/fstandhartinger/jevbench/tree/7625ecb1b075cfe4d09d34141d353325825eea1d), not the separately named metamorphic coherence benchmark at `JevBench/jevbench` on Hugging Face. Source revision and exact files are fixed in `configs/jevbench-public-assets.json`.

Downloaded public files match the original publisher's manifest hashes:

| Public split | Decisions | Distinct `group` values (including null) |
| --- | ---: | ---: |
| original | 72 | 36 |
| easy | 48 | 1 (JSON null) |
| hard | 111 | 111 |
| Total | 231 | 148 |

The locked official `metrics.py` includes only truthy groups with at least two rows for paraphrase consistency, so only the 36 original pairs participate; that is not 195-group scenario accuracy. All 231 rows fit the current primitive cardinalities (139 Choice, 18 discrete Score, 74 Noul). Score headline correctness uses argmax in the implementation despite an ordinal docstring mentioning rounded EV; optional gold probabilities do not replace the official one-hot Brier target.

The manifest, MIT license, hard-tier authoring method, and scoring/metrics code are also staged. Eight files were copied to the persistent instance and independently passed `sha256sum --check`. Remote path: `artifacts/data/research-eval/jevbench/7625ecb1b075cfe4d09d34141d353325825eea1d`. Evidence: `reports/jevbench-public-source-verification.json`.

The public hard method states that its labels were authored and cross-reviewed before benchmarked systems ran. We fetched no Jev API responses, per-model prediction artifacts, held-out items, or imported private/judge payloads. These public benchmark files are evaluation candidates and remain outside training admission.

The fixed teacher card reports 195 matched groups / 231 decisions but does not publish its selected IDs and grouping implementation. The earlier `group=easy` description was incorrect: all 48 easy rows have JSON `null`. There are 148 distinct literal values including null and 147 nonempty group values. Treating easy rows as separate scenarios would produce 36+48+111=195 groups, but that arithmetic is a candidate interpretation, not independent evidence of the teacher's actual protocol. Freeze grouping and exact source IDs before GPU scores are compared; do not select aggregation based on which score fits the card.

This source preparation completes the public file download gap. It does not pass teacher quality, probability agreement, calibration or private benchmark gates.

2026-10-14 correction and full public-structure audit: `reports/jevbench-public-protocol-audit-001.json` rechecks all eight file hashes and records source-field/scoring evidence. Historical download receipts remain unchanged; a valid hash did not validate the earlier group interpretation. Gold provenance is publisher-declared authored/reviewed before inference; hard rows have model authors, which is not human double-label evidence or independent provenance attestation.
