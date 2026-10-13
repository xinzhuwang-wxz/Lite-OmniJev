# Speech Offline Assets

Date: 2026-10-14.

This slice chooses and verifies the minimal speech/TTS assets needed before the next GPU window. It does not modify `configs/project-data-assets.json`, ticket state, or `plan/STATUS.md`.

## Current Recommendation

Use Kokoro 82M ONNX with `onnx/model_quantized.onnx` for the first synthetic speech pass. The selected source is `onnx-community/Kokoro-82M-v1.0-ONNX` at revision `1939ad2a8e416c0acfeecc08a694d14ef25f2231`, with the upstream model `hexgrad/Kokoro-82M`. The ONNX model card declares Apache-2.0, lists `model_quantized.onnx` as the 8-bit option, and documents the text path as Misaki phonemes mapped to Kokoro token IDs. The selected artifact payload is 179,453,458 bytes, below the 1 GB ceiling.

Remote staged path:

```text
/root/autodl-tmp/lite-omnijev/artifacts/speech-tts/downloads/kokoro-onnx
```

Verified remote hashes:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `README.md` | 10,627 | `b8fd888b4782f4d4ae85e2f1587cafd727037f4f138f72429ef1bd5908c0e9d8` |
| `config.json` | 44 | `df34b4f930b23447cd4dc410fabfb42eb3f24e803e6c3f97d618fb359380a36f` |
| `tokenizer.json` | 3,497 | `77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34` |
| `tokenizer_config.json` | 113 | `be1cb066d6ef6b074b3f15e6a6dd21ac88ff3cdaedf325f0aaed686c70f75d20` |
| `onnx/model_q8f16.onnx` | 86,033,585 | `04c658aec1b6008857c2ad10f8c589d4180d0ec427e7e6118ceb487e215c3cd0` |
| `onnx/model_quantized.onnx` | 92,361,116 | `fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478` |
| `voices/af_heart.bin` | 522,240 | `d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b` |
| `voices/am_puck.bin` | 522,240 | `fcf73c989033e9233e0b98713eca600c8c74dcc1614b37009d5450ff4a2274a0` |

## Text-To-WAV CPU Canary

Canonical readiness proof is now the Python 3.11 text-to-WAV canary, not the earlier fixed-token README proof. It used direct ONNX Runtime, Misaki English G2P `0.9.4` with `fallback=null`, `tokenizer.json` coverage checks that fail on unsupported phoneme characters, `CPUExecutionProvider`, and single-thread ORT/OMP settings. It produced four finite WAVs from two short English sentences across both selected voices.

Report: `reports/speech-kokoro-text-g2p-cpu-canary.json`

Remote output directory:

```text
/root/autodl-tmp/lite-omnijev/artifacts/speech-tts/canary/text-g2p
```

Runtime evidence:

| Field | Value |
| --- | --- |
| Python | 3.11.17 |
| Env | `/root/autodl-tmp/lite-omnijev/artifacts/speech-tts/.venv311` |
| ONNX Runtime | 1.31.0 |
| NumPy | 2.3.4 |
| Misaki | 0.9.4 |
| spaCy / model | 3.8.11 / `en_core_web_sm==3.8.0` |
| Dependency fingerprint | `965795436bc7638cafdb99c1bcd7bdb43179adb1d9610029b7ba2f8b3b1b20cc` |
| cgroup CPU | `50000 100000` |
| cgroup RAM | `2147483648` |

Generated WAVs:

| Sentence | Voice | Duration | Tokens | WAV SHA-256 |
| --- | --- | ---: | ---: | --- |
| `Clear speech assets are ready.` | `af_heart` | 2.300s | 29 | `5d42bff0f1e33eb96761a258bad6d44ec4adaba17b61e314c23caccf226a8c35` |
| `Clear speech assets are ready.` | `am_puck` | 2.225s | 29 | `d05f6e391c5c65d1324ba8bfeedfe1d91de8ef6988c89f8a3d5aa15c1cf7945a` |
| `Two voices render offline today.` | `af_heart` | 2.625s | 33 | `b30aae9f1b0689d882c3b96b28d22679fefe41c68be40d4f9975ebb32967bf28` |
| `Two voices render offline today.` | `am_puck` | 2.475s | 33 | `e641ddf034b5cd251dea9c239b91d54fc5fc53ba534d449a0401e0676edeb37d` |

The previous fixed-token report remains useful as historical ONNX execution proof only: `reports/speech-kokoro-cpu-wav-canary.json`. It did not exercise text normalization, G2P, or tokenizer coverage. The `model_q8f16.onnx` attempt remains a known failure in this environment: it segfaulted with ONNX Runtime 1.31.0 CPU, exit 139. It is staged for audit but not recommended for this CPU canary path.

## Candidate Comparison

| Candidate | License Evidence | Artifact Shape | Decision |
| --- | --- | --- | --- |
| Kokoro 82M ONNX | ONNX HF card and upstream Kokoro card declare Apache-2.0. Voice files are shipped in the same Apache-2.0 model repository. Misaki package metadata reports Apache-2.0. | ONNX plus `.bin` voice embeddings; no PyTorch pickle load required. | Selected; quantized ONNX Python 3.11 text-to-WAV CPU canary passed. |
| MeloTTS English | HF model card declares MIT; GitHub README says the library is MIT and free for commercial/non-commercial use. | `checkpoint.pth` PyTorch checkpoint, 207,860,748 bytes. | Not selected for this slice; quarantine unless converted/audited. |

MeloTTS remains a plausible fallback after a safe conversion plan, but it is not the minimal risk path for pre-GPU offline preparation.

## Common Voice Source Family

Primary selected source family:

| Dataset | Dataset ID | Size | SHA-256 | Status |
| --- | --- | ---: | --- | --- |
| Common Voice Spontaneous Speech 5.0 - English | `cmu5nqn1h00vwmi07b4dbk085` | 544,267,165 | `390b5a54c9afe0cc01da039ad206248f85682f247dd2b27d4cc0ab9a68e860b6` | Metadata selected; audio download blocked by official API-key and terms flow. |

Secondary tiny canary candidate:

| Dataset | Dataset ID | Size | SHA-256 | Status |
| --- | --- | ---: | --- | --- |
| Common Voice Scripted Speech 26.0 - Southern American English | `cmryvrc6i014vo90786kriix5` | 33,912,283 | `6badd6d6421fa395f680c23f6af0377df15a0429ff39f6e2230b05e1d29bee54` | Metadata selected; audio download blocked by official API-key and terms flow. |

Both records are public Mozilla Data Collective records, declare CC0-1.0, and include audio formats. MDC API docs say `GET /datasets/{datasetId}` is public, while `POST /datasets/{datasetId}/download` requires authentication and only works after the caller already satisfies dataset access requirements, including terms agreement completed in the web interface. I did not bypass that control and did not stage the Common Voice audio tarballs.

## Common Voice Download Steps For User-Owned Access

Human steps, once root/user is ready to proceed:

1. Open the selected MDC dataset page: https://mozilladatacollective.com/datasets/cmu5nqn1h00vwmi07b4dbk085
2. Sign in, read and accept the dataset terms in the MDC web interface. Do not delegate terms acceptance to an agent.
3. Create an API key in MDC profile settings. MDC docs describe API credentials under Profile > API and require `Authorization: Bearer YOUR_API_KEY` for authenticated endpoints.
4. Store the key on the remote host only, never in chat, Git, shell history, or logs:

   ```bash
   mkdir -p /root/.config/lite-omnijev
   install -m 0600 /dev/null /root/.config/lite-omnijev/mdc.key
   # paste the key into that file by a secure operator path
   ```

5. After terms are accepted and the key file exists, the prepared helper can request the official presigned download URL and verify size/hash:

   ```bash
   cd /root/autodl-tmp/lite-omnijev
   .venv/bin/python scripts/mdc_download_dataset.py \
     --dataset-id cmu5nqn1h00vwmi07b4dbk085 \
     --key-file /root/.config/lite-omnijev/mdc.key \
     --output artifacts/speech-commonvoice/common-voice-spontaneous-speech-5-0-engl-97a82389.tar.gz \
     --expected-size-bytes 544267165 \
     --expected-sha256 390b5a54c9afe0cc01da039ad206248f85682f247dd2b27d4cc0ab9a68e860b6
   ```

The helper calls only the official `POST /datasets/{datasetId}/download` API endpoint. It does not accept terms automatically and refuses to read a group/world-readable key file.

## Common Voice Public Alternative Check

No legitimate ungated original-publisher audio subset was found for this preparation slice. I rejected older direct S3 release links and third-party mirrors as substitutes because Mozilla's current Common Voice download path is Mozilla Data Collective, and its old-version FAQ says older releases are restricted through a request and terms flow so Mozilla can honor speaker deletion requests. A third-party Hugging Face mirror such as `fsicoli/common_voice_17_0` exposes direct CC0-tagged tar files, but it is not a current original-publisher download path and does not resolve the deletion-request stewardship constraint. The correct state remains: MDC metadata is pinned, audio bytes are account/terms blocked, and no older mirror is staged.

## Ready And Blocked

Ready:

- Kokoro q8f16 ONNX, quantized ONNX, tokenizer, and two voices are staged on remote under the dedicated speech artifact directory and hash-verified.
- Kokoro quantized ONNX generated four finite Python 3.11 text-to-WAV CPU canaries from two short English sentences across two voices.
- G2P/tokenizer coverage, input length guard, style-index bound, single-thread CPU runtime, cgroup CPU/RAM, dependency versions, model/voice hashes, and WAV hashes are recorded.
- Common Voice dataset IDs, filenames, sizes, checksums, licenses, official URLs, and the authorized download helper command are prepared under the 1 GB preparation ceiling.

Blocked:

- Common Voice audio bytes need a human-held Mozilla Data Collective API key and explicit terms acceptance through the official web interface.

## Sources

- Kokoro ONNX model: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
- Upstream Kokoro model: https://huggingface.co/hexgrad/Kokoro-82M
- Kokoro voices: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
- Misaki G2P: https://github.com/hexgrad/misaki
- MeloTTS model: https://huggingface.co/myshell-ai/MeloTTS-English
- MeloTTS code: https://github.com/myshell-ai/MeloTTS
- Common Voice dataset repository: https://github.com/common-voice/cv-dataset/
- Mozilla Data Collective API docs: https://mozilladatacollective.com/api-reference/docs
- Mozilla Data Collective API overview: https://mozilladatacollective.com/api-reference
- MDC Python SDK README: https://github.com/Mozilla-Data-Collective/datacollective-python
- Common Voice old-version FAQ: https://community.mozilladatacollective.com/faq-why-cant-i-download-old-versions-of-common-voice-datasets/
- Common Voice Spontaneous Speech 5.0 English: https://mozilladatacollective.com/datasets/cmu5nqn1h00vwmi07b4dbk085
- Common Voice Southern American English subset: https://mozilladatacollective.com/datasets/cmryvrc6i014vo90786kriix5
- Rejected third-party Common Voice mirror example: https://huggingface.co/datasets/fsicoli/common_voice_17_0/blob/main/audio/en/test/en_test_0.tar
- Rejected legacy direct-link discussion example: https://discourse.mozilla.org/t/updates-to-dataset-download-options/71470/11
