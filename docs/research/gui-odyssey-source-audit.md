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
