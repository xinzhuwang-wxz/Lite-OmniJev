# Speech Offline Assets Runbook

Date: 2026-10-14.

## Python 3.11 text-to-WAV canary

Run this only in the dedicated speech env and serialize ONNX inference with the shared CPU lock:

```bash
cd /root/autodl-tmp/lite-omnijev
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
ORT_DISABLE_TELEMETRY=1 \
flock /tmp/omnijev-cpu-heavy.lock \
  artifacts/speech-tts/.venv311/bin/python scripts/kokoro_text_g2p_canary.py \
    --config configs/speech-tts-canary.json \
    --report reports/speech-kokoro-text-g2p-cpu-canary.json
```

Expected report status is `passed` with four WAVs under:

```text
/root/autodl-tmp/lite-omnijev/artifacts/speech-tts/canary/text-g2p
```

The helper fails before ONNX inference if Misaki emits unsupported tokenizer characters, if the token length exceeds 510 before pad tokens, or if the style index would exceed the voice embedding rows.

## Common Voice official download after user terms acceptance

Do not download Common Voice audio until the user has accepted the selected dataset terms in MDC and provided an API key outside chat/Git/logs.

Human steps:

1. Visit https://mozilladatacollective.com/datasets/cmu5nqn1h00vwmi07b4dbk085
2. Sign in and accept the dataset terms in the MDC web interface.
3. Create an API key from Profile > API.
4. Store it as `/root/.config/lite-omnijev/mdc.key` with mode `0600`.

Prepared command:

```bash
cd /root/autodl-tmp/lite-omnijev
.venv/bin/python scripts/mdc_download_dataset.py \
  --dataset-id cmu5nqn1h00vwmi07b4dbk085 \
  --key-file /root/.config/lite-omnijev/mdc.key \
  --output artifacts/speech-commonvoice/common-voice-spontaneous-speech-5-0-engl-97a82389.tar.gz \
  --expected-size-bytes 544267165 \
  --expected-sha256 390b5a54c9afe0cc01da039ad206248f85682f247dd2b27d4cc0ab9a68e860b6
```

The helper requests `POST https://mozilladatacollective.com/api/datasets/{datasetId}/download`, streams the returned presigned URL, and verifies size plus SHA-256. It does not accept terms automatically.
