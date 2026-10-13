# GUI-Odyssey official source audit

This bounded slice corrects the older GUI-Odyssey account-blocked assumption without editing the shared Stage3 or whole-project ledgers. The official Hugging Face source is `hflqf88888/GUIOdyssey` at revision `61632d0f3f4d51d7e9561ce4f84347dd06b2019d`. The copied HF API metadata reports `private=false`, `gated=false`, and `disabled=false`, and the dataset card declares `license: cc-by-4.0`.

The committed ledger is [`configs/gui-odyssey-assets.json`](../../configs/gui-odyssey-assets.json). The detailed selected-episode manifest is [`reports/gui-odyssey-metadata-verification.json`](../../reports/gui-odyssey-metadata-verification.json). No account login, terms acceptance, GPU work, or screenshot media download was performed.

## Verified public metadata

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `artifacts/data/quarantine/gui-odyssey-source-audit/hf-api.json` | 418,757 | `bab7c8796bdecc623f9073f3de79f0d4ea99e606f837fc50b20be78d03a5d894` |
| `artifacts/data/quarantine/gui-odyssey-source-audit/hf-README.md` | 7,470 | `fd627ee0a6ee0596e0ee8147996b9a7a3436f754fc95ed63cf5f408d778b2831` |
| `artifacts/data/quarantine/gui-odyssey-source-audit/hf-head-core-files.json` | 15,405 | `1ac3597c9a95f811d6b308904e527d2936c3c16ac1095529c90bc9d0c51ecd32` |
| `artifacts/data/quarantine/gui-odyssey-source-audit/pinned-core/all_annot.json` | 2,862,103 | `46720388d7cbf8e0d4bb0d9d4bf286f6c2292c9bf64a4d0a88134475e10348b2` |
| `artifacts/data/quarantine/gui-odyssey-source-audit/pinned-core/splits/app_split.json` | 275,065 | `2f45f6b809162141d7ec5414b820c32ddfd0acc2354b0f85eb3aac8c702e4704` |
| `artifacts/data/quarantine/gui-odyssey-source-audit/pinned-core/splits/device_split.json` | 275,065 | `cc3ba7df9a1002ad63a7b1c195c1cd2440c54f9508b6c5a2859ae3fe03088a52` |
| `artifacts/data/quarantine/gui-odyssey-source-audit/pinned-core/splits/random_split.json` | 275,065 | `4feb280e7688ade825d3f504cfb2ef4540d44b12d9f72f5d5f09be3ffed5be5d` |
| `artifacts/data/quarantine/gui-odyssey-source-audit/pinned-core/splits/task_split.json` | 275,065 | `fb16a764ace673316073bd92c87c6bb97d0062d9d09f3171147f066c2cfe51f5` |

The HF file manifest has 8,350 paths: 8,334 `annotations/*.json` files, four `splits/*.json` files, `all_annot.json`, `README.md`, `.gitattributes`, and nine multipart screenshot archive files. The path list hash is `97d81749303fcb1d11662ae1e1e2e508aa0546461b4e968cc8d123810b35219c`.

The README states the dataset has 8,334 episodes, six mobile devices, six cross-app task categories, 212 apps, about 1.3K app combinations, and 127,893 screenshots after extraction. The downloaded `all_annot.json` matches 8,334 episodes and totals 127,893 step lengths.

## Size and capacity result

| Payload | Bytes | Status |
| --- | ---: | --- |
| `all_annot.json` + four split files | 3,962,363 | Downloaded and hashed. |
| Full multipart screenshot archive | 92,639,510,437 | Not downloaded; exceeds the approved extra 20 GB GUI cap. |
| HF `usedStorage` | 92,640,312,796 | Metadata only. |

The screenshot media is only exposed as multipart archive files: `screenshots.z01` through `screenshots.z08` at 10,737,418,240 bytes each plus `screenshots.zip` at 6,740,164,517 bytes. The current HF tree does not expose per-episode or per-screenshot media files, so even an episode subset cannot be physically staged without first downloading the full archive or finding a different official per-file source.

## Official splits and selected episode manifest

All four split files cover exactly the same 8,334 episode IDs with no duplicates. Counts:

| Split | Train episodes | Test episodes |
| --- | ---: | ---: |
| `app_split` | 7,165 | 1,169 |
| `device_split` | 6,953 | 1,381 |
| `random_split` | 6,668 | 1,666 |
| `task_split` | 7,199 | 1,135 |

The prepared first tranche is `gui_odyssey_app_split_test_v1`, using `splits/app_split.json:test`. It preserves the official app-family boundary and contains exactly 1,169 episode files. Its average-size pixel estimate from the full archive is 12,994,430,970 bytes, which is below 20 GB, but media is still not directly fetchable by episode from the current HF tree. The exact episode file list is embedded in `reports/gui-odyssey-metadata-verification.json` under `selected_episode_manifest.exact_episode_files`.

Candidate official test tranches are also recorded in the report for `device_split`, `random_split`, and `task_split`; all must preserve whole `episode_id` boundaries and must not split by individual step or screenshot row.

## Concrete next commands

Metadata/core resume command:

```bash
huggingface-cli download hflqf88888/GUIOdyssey \
  --revision 61632d0f3f4d51d7e9561ce4f84347dd06b2019d \
  --include all_annot.json 'splits/*.json' \
  --local-dir artifacts/data/quarantine/gui-odyssey-source-audit/pinned-core
```

If a later decision approves more than 92.64 GB for public CC-BY screenshot media, download the official multipart archive rather than a derivative mirror:

```bash
huggingface-cli download hflqf88888/GUIOdyssey \
  --revision 61632d0f3f4d51d7e9561ce4f84347dd06b2019d \
  --include 'screenshots/*' \
  --local-dir <approved-persistent-gui-media-root>
```

If only per-episode annotation JSONs are needed, use the exact `selected_episode_manifest.exact_episode_files` array from the report and request `annotations/<episode>.json` for those files. That is annotation-only staging; it does not stage screenshot pixels.

## Stop condition

GUI-Odyssey is no longer account-blocked on current public HF metadata. The source audit has a pinned official revision, license evidence, core metadata hashes, official split hashes, exact selected episode list, and a concrete media capacity blocker. No GUI-Odyssey screenshot media is downloaded or admitted for training/release by this report.

## ZIP64 range feasibility checkpoint

A bounded HTTP Range probe confirmed the screenshot payload is a ZIP64 split archive, not tar/gzip/7z. The final `screenshots.zip` part contains the ZIP64 EOCD and the central directory. The central directory is range-readable at byte offset `6725921680` inside `screenshots/screenshots.zip`, is 14,242,739 bytes, and has SHA-256 `866804191c030a8f2de9844ae36bf5edc1c05021f4700282156c44f6a60e14d9`.

The parsed central directory has 127,894 entries, including 127,893 screenshot PNG members. Member names use `screenshots/<episode_id>_<step>.png`, so selective recovery by episode is feasible with HTTP Range reads of each member's local header and compressed data. The materialized manifest is `artifacts/data/quarantine/gui-odyssey-source-audit/central-directory-manifest.jsonl`, 24,213,251 bytes, SHA-256 `94516edabd36183777bcb98e750753258ecaeb01fa1a4dd16294e3b8cc48315b`.

The helper [`scripts/gui_odyssey_media.py`](../../scripts/gui_odyssey_media.py) implements the bounded range reader, ZIP central-directory parser, whole-episode planner, and selective extractor. Focused tests are in [`tests/test_gui_odyssey_media.py`](../../tests/test_gui_odyssey_media.py).

## Train/eval media plans

The official `app_split:test` manifest remains evaluation-only. It contains 1,169 whole episodes, 11,993,498,013 compressed ZIP-member bytes, and 12,290,101,987 extracted PNG bytes. It is not used as a training-preparation manifest.

The frozen training-media candidate uses only `app_split:train`, keeps whole `episode_id` units, and does not cross into official test episodes. It selects 1,679 train episodes by lexical `episode_id` order while keeping extracted PNG bytes under the media cap:

| Plan | Episodes | Compressed bytes | Extracted PNG bytes | Role |
| --- | ---: | ---: | ---: | --- |
| `app_split:test` | 1,169 | 11,993,498,013 | 12,290,101,987 | Evaluation-only candidate. |
| `app_split:train` selected media | 1,679 | 19,066,700,692 | 19,499,790,993 | Training-media candidate under 20 GB actual PNG-byte cap. |

Plan artifacts:

- `artifacts/data/quarantine/gui-odyssey-source-audit/app-split-test-eval-plan.json`, 340,045 bytes, SHA-256 `1ac75f5d60e4015cb90d288bd34ae0bc5e973d0907e3dbd454e27c139ce9533f`
- `artifacts/data/quarantine/gui-odyssey-source-audit/app-split-train-media-plan-under-19_5gb.json`, 490,161 bytes, SHA-256 `661a56214ee442ae0c694b21c41b374fc02dad0ea1a962bb8d9638b3d393c0c4`

The selective media extraction is queued on the remote persistent disk behind `/tmp/omnijev-cpu-heavy.lock` to avoid competing with other CPU-heavy jobs. Queued command output is `artifacts/data/quarantine/gui-odyssey-source-audit/logs/gui-media-fetch.log`, and the expected completion report is `/root/autodl-tmp/lite-omnijev/reports/gui-odyssey-media-fetch-report.json`. Until that report exists, the train media candidate is planned and queued, not yet physically complete.

## Actual media checkpoint

Selective media extraction was started under `/tmp/omnijev-cpu-heavy.lock` using the range extractor. The first stable checkpoint is copied to [`reports/gui-odyssey-media-partial-checkpoint.json`](../../reports/gui-odyssey-media-partial-checkpoint.json): 207 PNG screenshots recovered, 168,055,047 bytes, 0 bad PNG headers, manifest SHA-256 `a6bbb12e5b3158eb1c51e257ef1721ca0887f792addd35144986ff783d7f9b21`.

This proves actual member recovery, decompression, hashing, and PNG header validation. The full 1,679-episode train-media candidate remains possible under the 20 GB media cap, but slow at current shared-lock/network throughput. It does not require a 92.64 GB full archive download unless root chooses to stage all GUI screenshots for a broader disk plan.
