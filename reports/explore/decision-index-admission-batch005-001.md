# Decision Index batch005 evaluation-only proposal

2026-10-16. Six publisher release candidates for root review; **not applied, ready=false**. Approved manifest004 SHA `6373c2f0f72f88c129b197c9f2d1cf4e4442274a815e8d3b5732b08ab232f2e8` remains unchanged. Proposed count27/44, remaining17. No GPU, benchmark rows, pickle loading, third-party code execution, gated downloads, accounts, acceptance or new data downloads. Cost null.

The machine proposal binds fixed upstream `f8552a13b643d39f61185f5951bb5fed16bedf51`, source tar SHA `12719836807976e25e9ff5b5dee742679577b48508cbd601f7c57be9672f4082`, original recipe/acquisition evidence, each absolute input path/SHA/size, and publisher metadata snapshot. Twelve actual remote files (57,230,302 bytes) were hashed without parsing. Local generation reused this metadata-only receipt; live CLI mode independently hashes the same paths. Prior21 approvals remain mandatory. No source manifest is written.

|ID|Release grant / scope|Identity and actual boundary|
|---|---|---|
|20 BPoMP|Publisher Zenodo7299879 dataset CC-BY-4.0|All3 publisher release MD5+sizes match actual input SHA-bound files. Release acknowledges earlier OEDILF poems; preserve original source/contributor credit, do not claim poems are benchmark-author originals. Earlier5520078 API TLS timed out and is not independent permission evidence. Direct published dataset grant is the proposed basis.|
|25 GPQA Diamond|Publisher fixed HF card CC-BY-4.0|Public Git dataset.zip matches pinned publisher blob. Fixed official card names gpqa_diamond configuration; card projection SHA binds license and restrictions. No gated HF data accessed or acceptance performed. Preserve canary and no online plaintext/image examples; no password/canary literal included. Git MIT does not replace explicit data grant.|
|28 WinoGrande|Publisher explicitly says dataset CC-BY, version unspecified|Official README separates Apache code from CC-BY data. Fixed official HF validation LFS SHA+size match actual parquet. Attribution required; **version remains unspecified**, root review must retain that boundary rather than relabel4.0.|
|34 SimpleBench|Root MIT applies to publisher-released dataset Work (scope inference)|Pinned README directly names public JSON; exact publisher blob matches input. Preserve copyright/permission notice. This is not an assertion that outside datasets inherit outer repository licenses.|
|40 iSarcasmEval|Root MIT applies to publisher-released collection (scope inference)|README describes author-supplied texts, author rephrasings and linguistic annotation. Five exact pinned test blobs match. Preserve publisher notice; no blanket claim about unrelated corpora.|
|44 CLadder|Root MIT applies to publisher-generated dataset+code Work (scope inference)|README explicitly releases full dataset and code, names zip; exact pinned zip blob matches. Preserve publisher notice.|

Pinned official URLs, full revisions, document SHA256, Git tree/blob or HF LFS identity, source acquisition mapping and purpose boundaries are in `.scratch/s4/decision-index-admission-batch005-metadata.json` and the proposal JSON. Metadata SHA `946a18dbbcbdb61794a0280944af375291955a732f458963e60961e0530a6b4c`. Git tree object SHA and commit revision are recorded separately. Dynamic API body SHA means observed snapshot; GPQA additionally has a canonical selected-card projection hash, not a claim that dynamic API counters are immutable.

SATA33 is explicitly pending: [publisher paper Appendix A](https://arxiv.org/html/2506.00643v2) identifies several externally sourced constituent datasets, including MultiRC Research and Academic Use License. Root SATA MIT alone does not resolve those actual component grants. This slice did not expand into all constituent investigations. Other remaining IDs:2,6,11,12,21,22,23,38,39,41,42,45,56,59,61,64. No admission proposed for them.

Root may review six proposed `Admission` objects under existing evaluation policy. This proposal is only offline evaluation/rebuild; it does not grant training, redistribution or product-release use and does not alter all44 requirement. Scope inferences are stated, not represented as explicit dataset-specific clauses.

Validation: Python3.11.14;11 narrow pytest cases pass (positive proposal/schema, altered manifest/prior set, source path/SHA/size/MD5/blob, duplicate observation, license snapshot, overwrite refusal); Ruff check/format pass. Proposal generation exit0 and duration recorded in JSON. No full rebuild or permission application performed.

Reproduce locally with metadata receipt (receipt contains hashes/paths only; full manifest remains outside Git):

```sh
PYTHONPATH=. .venv/bin/python .scratch/s4/decision-index-admission-batch005.py --manifest /tmp/index-batch005-manifest.json --observed /tmp/index-batch005-observed.json --output /tmp/batch005-new.json
PYTHONPATH=. .venv/bin/pytest -q .scratch/s4/decision-index-admission-batch005-tests.py
```

Remote live verification after root deploys helper+metadata beside each other, with project modules on PYTHONPATH (choose a new proposal output path; no apply):

```sh
PYTHONPATH=/root/autodl-tmp/s4-claim-pair-code-001 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python /root/autodl-tmp/decision-index-admission-batch005.py --manifest /root/autodl-tmp/s1-text-lora-001/artifacts/s4-index-evaluation-admission-004/manifest.json --output /root/autodl-tmp/s1-text-lora-001/reports/s4-index-admission-batch005-proposal-001.json
```
