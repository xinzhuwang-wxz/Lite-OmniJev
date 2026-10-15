# S4 image/screenshot original-gold capacity plan

2026-10-15; read-only repository analysis at `287a0cd4a3bb91f1f2003b479f8832cc0d4f70ed`.
No remote access, source payload access, sealed content access, pool construction or reassignment.
This is a capacity investigation, not a gold-admission receipt. Counts below are existing evidence unless explicitly marked as bounds or pending measurements.

## Counting and frozen inputs

The target is 3,000 calibration and 1,000 holdout **original question identities** per cohort. Report distinct original families separately; multiple genuinely different original questions in a family may count, but all remain in the same pool. Rewordings, options permutations, repeated frames and counterfactual variants do not add original items. This follows `reports/s4-gold-gate-implementation-assessment-001.md`.

Use the parent's private `artifacts/s4-gold-inventory-001/with-clevrer-gold-manifest-001.json`, SHA256 `3bff87d13b41244636dacccf61114db363f26544bc759f9884c920c632482ea7`, and its bound historical inputs plus `quarantined-reserves/exclusions.json`. Preserve every existing pool assignment and quarantine exclusion. Do not use the old screenshot reserve files directly.

| Cohort | Current calibration/holdout items | Distinct families | Item shortfall |
| --- | --- | --- | --- |
| image | 623 / 211 | 600 / 200 | 2,377 / 789 |
| screenshot | 119 / 59 | 119 / 59 | 2,881 / 941 |

Evidence: `reports/s3-reserve-inventory-001.json`, `reports/s4-reserve-quarantine-summary-001.json`, current STATUS. Image counts are not 623/211 independent images. Existing reserves still need source-label and permissions verification before they can contribute to final accepted totals.

## Image: reuse VQA2 source audit, measure remaining families first

`reports/s2-new-family-feasibility-001.json` already measures 16,826 CC-BY-2.0 COCO train2017 images outside **all** 16,540 A-OKVQA train families; 9,674 families have unanimous VQA2 answers, and 8,744 families support the restricted yes/no, integer or color conversion. These are pre-S2 candidate counts, not remaining S4 capacity. `reports/s2-new-family-admission-001.json` proves 400 distinct new-family pilot items were subsequently admitted (160 color, 160 integer, 80 yes/no); exclude all historical exposure, including these 400. Do not claim 8,344 is the final available count: crossmodal/dev/media alias exclusions can remove more.

Exact existing source files on the remote:

- `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/stage2-sources/coco2017/annotations/captions_train2017.json`; SHA `4b62086319480e0739ef390d04084515defb9c213ff13605a036061e33314317`.
- `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/stage2-sources/aokvqa/extracted/aokvqa_v1p0_train.json`; SHA `94820c4452d29102365647598c54608d4f48f56bee01b514c996a09cc0972029`.
- `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/vqa2-annotations-20261015/v2_Annotations_Train_mscoco.zip`; SHA `fb101bcefe91422c543c2bb6d70af11eb3119d0ff745ae283d09acdf66250853`; member `v2_mscoco_train2014_annotations.json`.
- Same directory, `v2_Questions_Train_mscoco.zip`; SHA `05a64b6e2582d06d7585f5429674a9a33851878be1bff9f8668cdcf792df611e`; member `v2_OpenEnded_mscoco_train2014_questions.json`.

License provenance: VQA2 annotation CC-BY-4.0 is recorded with official terms URL and archive hashes in `reports/s2-vqa2-staging-001.json`; per-image COCO license id 4 must exactly map to `http://creativecommons.org/licenses/by/2.0/`, with image source/Flickr attribution and downloaded-byte SHA retained. The A-OKVQA revision is `42e8c44ff40d13cf272776a3197365f582137772`. Use source checksums as VQA2 source identity; do not invent a repository revision.

Reuse predicates in `scripts/vqa2_new_family.py:plan`: membership exclusion before answer access; 10/10 normalized annotator unanimity; same-image/same-normalized-question conflict rejection; yes/no, integer 0–20, or supported color; legal Choice options. Integer/color distractors are derived options, not extra gold questions. For the new capacity probe count all eligible original identities and distinct images, without category quotas. The current planner's total quota cap of 600 and its training-oriented materializer mean it is **not** an S4 builder; do not simply call it repeatedly or relabel its train output.

Lowest-cost sequence:

1. Private metadata/annotation scan with full current exclusion graph; no image download. Count eligible original items/families, available local media, and missing media. Existing local media SHA manifests may cheaply identify reuse, but final download/hash/decode is required for new images.
2. Prefer one eligible original question per unused image if measured capacity permits filling 2,377/789. This avoids relying on correlated multiple questions and adds only 3,166 selected JPEGs at most; exact bytes remain unknown until source/member sizing or download receipts. If reserves fail gold admission, size against all 4,000 instead.
3. Freeze deterministic seed-42 selection privately, keep calibration/holdout original components disjoint, download only selected missing images, verify hashes/decoding and re-audit media aliases against *all* historical exposures before admitting anything. Drop colliding candidates and continue deterministic selection, without changing historical pools.

A-OKVQA alone is not the expansion answer: `configs/stage2-data-assets.json` reports only 2,741 allowed train rows and 219 allowed val rows, before existing exposures. Its 1,079 allowed test-media rows do not prove available test gold. Already staged NC/ND/SA pools do not become approved commercial gold by virtue of being on disk. The 1,811 VQA2 augmentation rows in `reports/s2-vqa2-gold-001.json` explicitly reuse 1,341 parent train images; none supplies new S4 families.

## Screenshot: count all original steps in unexposed cached episodes

`configs/gui-odyssey-assets.json` pins `hflqf88888/GUIOdyssey` revision `61632d0f3f4d51d7e9561ce4f84347dd06b2019d`, dataset-card/README CC-BY-4.0 with attribution. Official source has 8,334 episodes / 127,893 screenshots; app split has 7,165 train and 1,169 test episodes. These are source totals, not usable gold counts.

The already downloaded, SHA-verified and fully decoded tranche has **1,679 app-train episodes / 25,809 PNGs / 19,499,790,993 bytes** (`reports/gui-odyssey-download-acceptance-001.json`, `reports/gui-odyssey-full-decode-cpu-001.json`). Cache paths:

- Source plan: `/root/autodl-tmp/lite-omnijev/source-audit/stage3-source-audit/gui-odyssey/app-split-train-media-plan-under-19_5gb.json` (actual S0 command); equivalent source manifest path is under `artifacts/data/quarantine/gui-odyssey-source-audit/`. Verify whichever is used against SHA `661a56214ee442ae0c694b21c41b374fc02dad0ea1a962bb8d9638b3d393c0c4`.
- Annotation directory: `/root/autodl-tmp/s0-gui-dev-001/artifacts/annotations`; S0 v2 receipts prove 630 annotations present, **not** all 1,679 currently present. Later source scans may have added more; measure availability without assuming it.
- Media root for the builder: `/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/gui-odyssey-source-audit/media/app_split_train_under19_5gb`; screenshots are inside `screenshots/`.
- Bound full source episode list: `artifacts/data/quarantine/gui-odyssey-source-audit/pinned-core/all_annot.json` (SHA `46720388d7cbf8e0d4bb0d9d4bf286f6c2292c9bf64a4d0a88134475e10348b2`); app split SHA `2f45f6b809162141d7ec5414b820c32ddfd0acc2354b0f85eb3aac8c702e4704`.

Existing conversion: `scripts/s0_gui_dev.py:eligible_step` validates episode filenames and step indices, then finds the **first** eligible original CLICK with nonempty instruction, legal bbox, click inside bbox, and three legal negatives. `sample_for` keeps family `gui:<episode_id>`; `provenance_row` records original step, raw-step hash, click and bbox as private program-truth evidence. Current v2 build produced 500 dev families; reserve inventory records another 420 train families and initially 180 reserved families. Therefore cached media cannot provide 4,000 independent episode families. This does **not** establish failure of the 4,000-original-question requirement.

The cheapest next probe is to count **every distinct eligible original CLICK step** on unexposed cached episodes while keeping whole episodes in one pool. Do not count option permutations or repeated screenshots; deduplicate source identity and original/decoded media aliases. It may also count further original steps inside existing reserve families *in their already assigned pool only*; report these capacities separately from new unassigned families, without moving them. For a conservative no-extension first pass, exclude all assigned reserve families as well. Do not infer capacity from average episode length or the 500/630 first-valid-step yield. Missing annotation downloads are small compared with media and can precede any new screenshots; source URL pattern is pinned `https://huggingface.co/datasets/hflqf88888/GUIOdyssey/resolve/61632d0f3f4d51d7e9561ce4f84347dd06b2019d/annotations/<episode_id>.json`.

If cached eligible steps cannot fill 2,881/941 with component-level allocation, the next route is pinned GUIOdyssey unused app-train episodes, select original usable steps before downloading their PNGs. Existing `scripts/gui_odyssey_media.py` and central directory manifest prove ZIP64 range extraction is feasible. Download only selected steps, not entire episodes or the ~92.6GB multipart archive. The current `validate_episode_steps` requires every episode screenshot; a new narrow S4 validator must validate full annotation structure but require media only for selected steps, with tests. Record selected-member compressed sizes from central directory and check remaining approved download budget **before** fetching; the prior ~20GB tranche budget is not authorization for arbitrary expansion. Keep 1,169 official app-test episodes available for separate external evaluation rather than silently consuming them.

Alternative sources are lower priority: `reports/android-control-source-download-002.json` and `reports/mind2web-source-download-002.json` prove downloads/hashes only (`payloads_parsed=false`, research-only, no commercial admission). Mind2Web explicitly carries OpenRAIL plus solely-for-research disclaimer in `configs/gui-source-assets.json`; neither has demonstrated usable original screenshot-choice capacity here. ScreenSpot has only 610 media files (`configs/screenspot-assets.json`) and evaluation-only component-provenance limits. DocVQA/chart sources are not a substitute for GUI screenshot localization and have separate license/admission boundaries. Do not change screenshot cohort semantics merely to meet its count.

## Smallest private probe specification and exit condition

Run one CPU-only, read-only capacity job. The provided `.scratch/s4/visual-capacity-private.py` implements the conservative cached GUI scan, reusing the inventory audit and existing GUI geometry/decode functions. It excludes every assigned family, including current reserves; reserve extensions are not measured. VQA2 capacity scan remains a separate parent integration step.

1. SHA-bind the clean inventory, quarantine exclusion file and source manifests. Reuse `scripts/gold_inventory.py:Manifest`, `bound_lines`, `extract`, `link_components` and `eval/gold_gate.py:audit_pool_isolation` internally. Resolve original aliases and all historical/current component links; reject inconsistent metadata. Source candidates are never assigned new pools in this probe.
2. VQA2: scan pinned COCO/license metadata and source annotation/question archives; compute the above full-exclusion eligible counts. GUI: scan cached plan and available annotations, compute all-step eligibility with existing v2 predicates; count missing annotations separately. Suppress helper exceptions/logs which contain filenames or source IDs.
3. Export only a strict allowlist: input/runner/code digests, seed, elapsed seconds, CPU environment, exit code, GPU=false, network bytes=0, candidate originals/families/components, known exclusions, missing annotation/media counts, media-ready originals, and per-fixed-pool capacity (existing calibration, existing holdout, unassigned). Include `gold_admission=not_validated` and `permissions=not_validated`. Errors are fixed enum categories, never exception messages, paths, IDs, questions, labels or dynamic source-derived keys.
4. No private candidate files are returned to the agent. Parent may let the independent process retain private metadata for a later builder under the actual gate boundary. Report capacity that can be placed into 3,000/1,000 while preserving connected components; raw sum >=4,000 alone is insufficient. If allocation feasibility is not tested, mark it pending rather than invent quotas.

Stop the investigation when this one probe establishes residual capacities and missing bytes/annotation counts. The next action is a source-specific builder only for the demonstrated viable route. Current evidence favors VQA2 for image and cached multi-step GUIOdyssey for the first screenshot measurement; screenshot sufficiency is still **unmeasured**, not established or rejected.

Validation for this document: repository-only source/code cross-check and `git diff --check`; no GPU, network or actual private probe executed.

## Executable cached-GUI probe

Run with the project's Python/Pillow environment from repository root:
`PYTHONPATH=. python .scratch/s4/visual-capacity-private.py /absolute/private/config.json`.
Config keys: `inventory`, `quarantine`, `plan` each contain exactly `path` (absolute) and `sha256`; `annotation_root` and `media_root` are the absolute directories above. Quarantine requires exactly `families` and `media` SHA256 lists; its bound digest is `8901483c170a7b1b1d04cf339eb4fbc6fc9e9722b8c2beb9cde0dc56b437ba7f`.

The probe verifies the full inventory first, excludes all declared families and component aliases, hashes/decodes each unassigned candidate original CLICK screenshot, rejects historical byte-SHA collisions, and connects candidate episodes sharing byte/RGB hashes before conservative one-media/one-question counting. Whole colliding components are excluded. It reports cached original steps, missing annotations/media, invalid geometry/media, duplicate media/questions, unique families/components, and a bitset subset-sum feasibility boolean for the 2,881/941 shortfall without exporting or writing a pool allocation. Imported helpers' stdout/stderr and all runtime exceptions are suppressed into fixed `job_failed`; no path-bearing exception leaves the job. Run inside the actual private launcher because this script does not establish permissions.

Limitation: historical inventory contains original family and byte media hashes, not historical decoded-RGB hashes. Candidate RGB dedup is real; historical re-encoding lineage remains a separate unvalidated boundary. The measured count cannot claim complete historical decoded/perceptual exclusion. The probe neither fetches missing annotations nor assumes their eligible yield. It deliberately conservatively counts at most one original per identical media, even if two legitimate instructions differ.

Local executable checks: Python compile, ruff lint and format, `git diff --check` passed. Synthetic aggregation checks passed for repeated screenshot/question, missing media, and quarantined-family rejection (inventory and image decode mocked). Missing-config failure returned exit 1 with only fixed aggregate JSON and no sentinel path. Local project environment lacks Pillow, so a real PNG decode probe could not run here; use the existing remote media environment and do not count this synthetic check as source capacity evidence.
