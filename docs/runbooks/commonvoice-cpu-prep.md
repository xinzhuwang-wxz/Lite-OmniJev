# Common Voice CPU Preparation

This runbook validates the already-authorized Mozilla Data Collective Common
Voice Spontaneous Speech 5.0 English archive as source-only public data. It does
not download data, read MDC credentials, admit training samples, create a sealed
identity, or claim ASR/model quality.

The pinned archive is remote-only for this slice:

```bash
/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/commonvoice-spontaneous-5-en/archive.tar.gz
```

Expected size is `544267165` bytes and expected SHA-256 is
`390b5a54c9afe0cc01da039ad206248f85682f247dd2b27d4cc0ab9a68e860b6`.

Run under the CPU-heavy lock on the remote host. Keep the job single-threaded and
within the no-GPU/no-network preparation lane:

```bash
cd /root/autodl-tmp/lite-omnijev/source
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
uv run python -m scripts.prepare_commonvoice \
  --archive /root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/commonvoice-spontaneous-5-en/archive.tar.gz \
  --output-dir /root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/commonvoice-spontaneous-5-en/cpu-prep \
  --report reports/commonvoice-cpu-prep.json
```

The script first hashes the archive, then streams every tar member and rejects the
whole archive before extraction on absolute or parent-relative paths, links,
special files, duplicate canonical names, too many members, oversized members,
or excessive total expansion. It then writes only sanitized metadata schemas and
row counts into a new isolated output directory, replacing transcript and speaker
column names with redacted hashes. Raw TSV rows are not printed in the report.

Audio validation uses a deterministic sorted-name stride sample capped at 32
members and 128 MiB temporary extraction. Each selected audio file is extracted
to a numeric temporary filename, decoded fully through ffmpeg to EOF on CPU, and
then removed with the temporary directory. The report records aggregate codecs,
sample rates, channels, durations, and selected member name hashes only.

Exit code `0` means the full archive was readable, tar member safety checks
passed, metadata schema/count extraction completed, and the bounded real-audio
CPU decode sample completed. It does not mean the dataset is admitted for
training, licensed for every downstream use, or sufficient for model quality.
