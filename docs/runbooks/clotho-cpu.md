# Clotho-AQA CPU Audio Preparation

This runbook prepares only the CC0 and CC-BY audio subset from the authorized
Clotho-AQA source archive. It does not admit QA labels, hidden questions, or
training rows.

Pinned remote inputs:

```bash
/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207/audio_files.zip
/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207/clotho_aqa_metadata.csv
```

Expected archive size is `3140677494` bytes, MD5 is
`48c2c46ccd71dc06121228e2ed68d6b9`, and SHA-256 is
`4793d97781689f8bdb1a6669056386825bb0ec8022f2217acc8083abc4774d66`.

Run on the remote host under the shared CPU-heavy lock:

```bash
cd /root/autodl-tmp/lite-omnijev
flock /tmp/omnijev-cpu-heavy.lock env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -m scripts.prepare_clotho extract \
    --metadata artifacts/data/quarantine/clotho-aqa/6473207/clotho_aqa_metadata.csv \
    --archive artifacts/data/quarantine/clotho-aqa/6473207/audio_files.zip \
    --output artifacts/data/quarantine/clotho-aqa/6473207/approved-audio-002 \
    --report reports/clotho-aqa-audio-preparation-002.json
```

The script validates metadata hash, archive size, official MD5, and SHA-256
before extraction. It scans the complete zip member table and rejects unsafe
paths, links, special files, excessive member count, excessive member size, and
excessive total expansion before selecting approved audio.

Two fixed revision aliases are allowed because the official archive uses a
different punctuation spelling from the metadata for exactly two approved files:

- `souffle_me_tallique.wav` maps to archive member `souffle_me.tallique.wav`
- `Geese1 - (Apollonia&#39;s sPA) 18_44 05.10.wav` maps to archive member
  `Geese1 - (Apollonia&#39;s sPA) 18.44 05.10.wav`

The canonical metadata filename remains the output filename. No global fuzzy
matching is allowed; any alias collision fails the run.

Exit code `0` means the 1,659 CC0/CC-BY files were extracted or verified in
place, 332 non-approved rows were excluded, and a deterministic sample of up to
32 approved audio files fully decoded through ffmpeg on CPU. It does not mean
training admission, hidden-question access, GPU readiness, or model quality.
