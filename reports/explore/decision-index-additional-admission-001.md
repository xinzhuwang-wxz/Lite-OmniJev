# Decision Index: three additional evaluation admission proposals

2026-10-16. CPU metadata/source audit only. Three additional candidates are closed for root review; no manifest was applied. If accepted, existing13 plus these3 cover16/44;28 remain unchanged. No GPU, upstream builder, pickle, dataset download, question display, calibration or holdout access occurred. Local existing source code was read; remote data bytes were hashed without decoding or emitting rows.

Base approved manifest SHA256: `b40157e74b88f411161d3905a75ff4077f9f5b6cf2ec1d1ba0ad887b04842ac1`. Source revision `f8552a13b643d39f61185f5951bb5fed16bedf51`; archive SHA256 `12719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082`. Original input manifest/admissions remain unchanged. Exact paths, file SHA256, publisher Git blob IDs, license/card hashes and source-code hashes are in the companion JSON. Proposal objects conform to `Admission` and append to the full `PreparationManifest` without removing any catalog requirement.

|ID|Scope and evidence|Proposed license|
|---|---|---|
|4 BANKING77|Fixed recipe `normalize_direct.py:52–60` reads test.csv/categories.json. Existing files match both recipe SHA256 and official pinned Git tree blobs. PolyAI README lines149–150 explicitly applies LICENSE to datasets shared in this repository.|CC-BY-4.0|
|5 CLINC150+OOS|Recipe `normalize_direct.py:63–75` reads data_full.json, derives label names from train and emits test/oos_test. Existing bytes match official pinned Git tree blob. Publisher repository root LICENSE identifies Attribution3.0 Unported; the same repository publishes the dataset. Prior approved HF card at fixed revision identifies cc-by-3.0 and links this publisher repository.|CC-BY-3.0|
|9 Home Appliance|Fixed `home_appliance.py:1–211` imports only copy/hashlib/json/random; content originates in literal constants and local scenario/oracle functions. `home_appliance:198–208` constructs rows, writes output, then reads those newly written files only to hash them. `BUILDERS={9:home_appliance}`; acquisition NEEDS[9] is empty. No external dataset content is consumed.|MIT, fixed upstream root LICENSE|

Official metadata pins: PolyAI `57ec275d8078af65b7731c2a98be812d844a6d6b`; CLINC GitHub `828f8093932c8fe6ca7936c3d2e52903b1c523de`; existing approved HF card `155b9c710419136e17307b80d0a13e68cd46b4ec`. These were resolved/read as publisher metadata, not claimed as the original acquisition revision. Exact Git blob equality links the already downloaded files to these license-bearing publisher snapshots without downloading their data again.

CLINC boundary: `reports/clinc-admission-corrected-001.json` approved only HF plus/train.parquet and explicitly did not access test. This proposal does not extend that training approval. It independently binds the Index evaluation input to the publisher license. The prior receipt's fixed HF LICENSE URL returns404; the fixed README's detailed licensing section says More Information Needed while its machine-readable card says cc-by-3.0. Neither the missing LICENSE URL nor that incomplete subsection is treated as sole evidence: the official GitHub LICENSE, same publisher repository link and matching data blob provide the chain. This discrepancy remains recorded for traceability, not silently repaired.

Purpose remains offline local evaluation/rebuild; no training, dataset redistribution or commercial-release approval. Preserve publisher attribution/license and notices of transformations. This records publisher facts and proposes existing-policy use; it does not accept new terms or provide legal advice.

## Verification

Actual metadata-only remote hashing covered3 files (2,737,387bytes): each SHA256 equalled recipe/manifest expectations, and each Git blob SHA1 equalled the official pinned tree metadata. No raw contents crossed SSH. Local fixed source4 files matched frozen hashes. Python3.11 environment, proposal schema validation,1 positive and5 rejection checks passed (base digest, changed admission bytes, dataSHA, publisher blob, missing license evidence); ruff check/format passed. No rebuild was attempted.

Re-run using a fresh read-only captured `observed.json` object `{manifest,manifest_sha256,observed_files}`: manifest must be the exact approved remote file `artifacts/s4-index-evaluation-admission-001/manifest.json`; observed_files must contain only the3 relevant destinations with path, bytes, SHA256 and Git `sha1(b'blob '+size+b'\0'+bytes)` computed remotely without displaying contents. Full manifest contains only metadata and is not committed again. `publishers.json` is exactly companion report's `publisher_metadata` list. No dataset rows are needed.

```sh
PYTHONPATH=. /Users/bamboo/Githubs/Lite-OmniJev/.venv/bin/python \
 .scratch/s4/decision-index-additional-admission.test.py \
 --observed /tmp/index-additional-observed.json \
 --publishers /tmp/index-additional-publisher.json \
 --source /Users/bamboo/Githubs/Lite-OmniJev/artifacts/source-audit/public-index/src/decision-index-f8552a13b643d39f61185f5951bb5fed16bedf51 \
 --clinc-receipt reports/clinc-admission-corrected-001.json
```

The builder uses identical arguments plus `--output NEW_PATH`, refuses overwriting, and verifies the original approved manifest's exact indent2/newline bytes before proposing changes. The `.scratch/s4/decision-index-additional-admission.json` is reviewable admission JSON, not an instruction to apply it. Root must review and separately produce any new manifest.
