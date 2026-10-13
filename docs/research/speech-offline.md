# Speech Offline Assets

Date: 2026-10-14.

This slice chooses the minimal speech/TTS assets needed before the next GPU window. It does not modify `configs/project-data-assets.json`, ticket state, or `plan/STATUS.md`.

## Recommendation

Use Kokoro 82M ONNX for the first synthetic speech pass. The selected artifact is `onnx-community/Kokoro-82M-v1.0-ONNX` at revision `1939ad2a8e416c0acfeecc08a694d14ef25f2231`, using `onnx/model_q8f16.onnx` plus `voices/af_heart.bin` and `voices/am_puck.bin`. The HF model card declares Apache-2.0 and the ONNX card points back to `hexgrad/Kokoro-82M` as its base model. The selected byte set is 87,092,342 bytes, far below the 1 GB ceiling.

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
| `voices/af_heart.bin` | `d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b` |
| `voices/am_puck.bin` | `fcf73c989033e9233e0b98713eca600c8c74dcc1614b37009d5450ff4a2274a0` |

The convenience `kokoro-onnx` Python package was not adopted as a project dependency in this slice. Its install path pulled `espeakng-loader`, so production preparation should prefer direct ONNX Runtime plus Apache-2.0 Misaki G2P with fallback disabled unless legal review approves another runtime path. Misaki’s README shows an English path with `fallback=None`, while its eSpeak fallback is explicitly separate.

## Candidate Comparison

| Candidate | License Evidence | Artifact Shape | Decision |
| --- | --- | --- | --- |
| Kokoro 82M ONNX | ONNX HF card and upstream Kokoro card declare Apache-2.0. Voice files are shipped in the same Apache-2.0 model repository. | ONNX plus `.bin` voice embeddings; no PyTorch pickle load required. | Selected. |
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

## Ready And Blocked

Ready:

- Kokoro q8f16 ONNX and two voices are staged on remote under the dedicated speech artifact directory and hash-verified.
- Common Voice dataset IDs, filenames, sizes, checksums, licenses, and source URLs are pinned under the 1 GB preparation ceiling.

Blocked:

- Common Voice audio bytes need a Mozilla Data Collective API key and acceptance of the official download terms.
- CPU WAV canary needs `onnxruntime` in the dedicated speech venv. The remote install of `onnxruntime==1.31.0` and `numpy==2.3.4` did not complete in the available window and was stopped. No unsafe PyTorch checkpoint was loaded.

## Sources

- Kokoro ONNX model: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
- Upstream Kokoro model: https://huggingface.co/hexgrad/Kokoro-82M
- Kokoro voices: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
- Misaki G2P: https://github.com/hexgrad/misaki
- MeloTTS model: https://huggingface.co/myshell-ai/MeloTTS-English
- MeloTTS code: https://github.com/myshell-ai/MeloTTS
- eSpeak NG license note: https://github.com/espeak-ng/espeak-ng
- Common Voice Spontaneous Speech 5.0 English: https://mozilladatacollective.com/datasets/cmu5nqn1h00vwmi07b4dbk085
- Common Voice Southern American English subset: https://mozilladatacollective.com/datasets/cmryvrc6i014vo90786kriix5
