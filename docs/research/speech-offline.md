# Speech Offline Assets

Date: 2026-10-14.

This slice chooses the minimal speech/TTS assets needed before the next GPU window. It does not modify `configs/project-data-assets.json`, ticket state, or `plan/STATUS.md`.

## Recommendation

Use Kokoro 82M ONNX for the first synthetic speech pass. The selected artifact is `onnx-community/Kokoro-82M-v1.0-ONNX` at revision `1939ad2a8e416c0acfeecc08a694d14ef25f2231`, using `onnx/model_quantized.onnx` for the CPU canary plus `voices/af_heart.bin` and `voices/am_puck.bin`. `onnx/model_q8f16.onnx` is still staged and hash-verified, but it segfaulted under ONNX Runtime 1.31.0 CPU in this environment, so it is not the canary-proven model. The HF model card declares Apache-2.0 and the ONNX card points back to `hexgrad/Kokoro-82M` as its base model. The selected byte set is 179,453,458 bytes, far below the 1 GB ceiling.

Remote staged path:

```text
/root/autodl-tmp/lite-omnijev/artifacts/speech-tts/downloads/kokoro-onnx
```

Verified remote hashes:

| File | SHA-256 |
| --- | --- |
| `README.md` | `b8fd888b4782f4d4ae85e2f1587cafd727037f4f138f72429ef1bd5908c0e9d8` |
| `config.json` | `df34b4f930b23447cd4dc410fabfb42eb3f24e803e6c3f97d618fb359380a36f` |
| `tokenizer.json` | `77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34` |
| `tokenizer_config.json` | `be1cb066d6ef6b074b3f15e6a6dd21ac88ff3cdaedf325f0aaed686c70f75d20` |
| `onnx/model_q8f16.onnx` | `04c658aec1b6008857c2ad10f8c589d4180d0ec427e7e6118ceb487e215c3cd0` |
| `onnx/model_quantized.onnx` | `fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478` |
| `voices/af_heart.bin` | `d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b` |
| `voices/am_puck.bin` | `fcf73c989033e9233e0b98713eca600c8c74dcc1614b37009d5450ff4a2274a0` |

The CPU canary used direct ONNX Runtime, fixed token IDs from the Kokoro ONNX model card README, the `af_heart` voice embedding, `speed=1.0`, `ORT_DISABLE_TELEMETRY=1`, one ONNX Runtime CPU thread, and `CPUExecutionProvider`. It did not use a text phonemizer and did not load a PyTorch checkpoint or remote code. The produced WAV is an execution canary only; it is not a quality claim.

CPU canary evidence:

| Evidence | Value |
| --- | --- |
| Report | `reports/speech-kokoro-cpu-wav-canary.json` |
| Remote WAV | `/root/autodl-tmp/lite-omnijev/artifacts/speech-tts/canary/kokoro-af-heart-readme-token-canary.wav` |
| WAV SHA-256 | `861b1e8a2a70f3c78b5176555115b04e92c8bbd05b86f7548f58d4fee9a5ae71` |
| Duration | 8.225 seconds |
| Format | 24 kHz, mono, finite PCM waveform |
| Runtime | Python 3.12.3, ONNX Runtime 1.31.0, NumPy 2.3.4 |

The convenience `kokoro-onnx` Python package was not adopted as a project dependency in this slice. Its install path uses `espeakng-loader`, so production preparation should prefer direct ONNX Runtime plus Apache-2.0 Misaki G2P with fallback disabled unless legal review approves another runtime path. Misaki’s README shows an English path with `fallback=None`, while its eSpeak fallback is explicitly separate.

## Candidate Comparison

| Candidate | License Evidence | Artifact Shape | Decision |
| --- | --- | --- | --- |
| Kokoro 82M ONNX | ONNX HF card and upstream Kokoro card declare Apache-2.0. Voice files are shipped in the same Apache-2.0 model repository. | ONNX plus `.bin` voice embeddings; no PyTorch pickle load required. | Selected; quantized ONNX CPU canary passed. |
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

Both records are public Mozilla Data Collective records, declare CC0-1.0, and include audio formats. The page text and API UI state that downloads require API credentials and terms acceptance. I did not bypass that control and did not stage the Common Voice audio tarballs.

## Common Voice Public Alternative Check

No legitimate ungated original-publisher audio subset was found for this preparation slice. I rejected older direct S3 release links and third-party mirrors as substitutes because Mozilla's current Common Voice download path is Mozilla Data Collective, and its old-version FAQ says older releases are restricted through a request and terms flow so Mozilla can honor speaker deletion requests. A third-party Hugging Face mirror such as `fsicoli/common_voice_17_0` exposes direct CC0-tagged tar files, but it is not a current original-publisher download path and does not resolve the deletion-request stewardship constraint. The correct state remains: MDC metadata is pinned, audio bytes are account/terms blocked, and no older mirror is staged.

## Ready And Blocked

Ready:

- Kokoro q8f16 ONNX, quantized ONNX, and two voices are staged on remote under the dedicated speech artifact directory and hash-verified.
- Kokoro quantized ONNX generated a finite 8.225 second 24 kHz mono PCM WAV through CPU ONNX Runtime; report and WAV hash are recorded.
- Common Voice dataset IDs, filenames, sizes, checksums, licenses, and source URLs are pinned under the 1 GB preparation ceiling.

Blocked:

- Common Voice audio bytes need a Mozilla Data Collective API key and acceptance of the official download terms.

## Sources

- Kokoro ONNX model: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
- Upstream Kokoro model: https://huggingface.co/hexgrad/Kokoro-82M
- Kokoro voices: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
- Misaki G2P: https://github.com/hexgrad/misaki
- kokoro-onnx wrapper: https://github.com/thewh1teagle/kokoro-onnx
- MeloTTS model: https://huggingface.co/myshell-ai/MeloTTS-English
- MeloTTS code: https://github.com/myshell-ai/MeloTTS
- eSpeak NG license note: https://github.com/espeak-ng/espeak-ng
- Common Voice dataset repository: https://github.com/common-voice/cv-dataset/
- Common Voice old-version FAQ: https://community.mozilladatacollective.com/faq-why-cant-i-download-old-versions-of-common-voice-datasets/
- Common Voice Spontaneous Speech 5.0 English: https://mozilladatacollective.com/datasets/cmu5nqn1h00vwmi07b4dbk085
- Common Voice Southern American English subset: https://mozilladatacollective.com/datasets/cmryvrc6i014vo90786kriix5
- Rejected third-party Common Voice mirror example: https://huggingface.co/datasets/fsicoli/common_voice_17_0/blob/main/audio/en/test/en_test_0.tar
- Rejected legacy direct-link discussion example: https://discourse.mozilla.org/t/updates-to-dataset-download-options/71470/11
