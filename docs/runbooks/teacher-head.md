# Teacher Head Conversion Runbook

This runbook covers only the Jev-Omni `head.pt` admission step. It does not admit teacher quality, run GPU inference, or allow JEV-27B/JEV-27B-VL assets.

Pinned source:

- Repository: `akhilaaa3/Jev-Omni`
- Revision: `5addda86ddee081a68fb067477ea100c221b8917`
- Expected `head.pt` SHA-256: `8c81edecd733f7db327604803c14cf9ecbee53d2af055eeb09f78b64bcbf2638`
- Expected size: `3966079`

Probe the host boundary first. A ready result means the subprocess can install a Linux seccomp rule set that denies network syscalls; a blocked result must stay blocked until the host supports that boundary.

```bash
uv run python -m scripts.teacher_head probe-isolation --report reports/teacher-head-isolation.json
```

Convert only through the audited path. The command requires the pinned source metadata, the input hash, tensor-only `torch.load(..., weights_only=True)`, finite tensor checks, safetensors round-trip equality, and an output hash report.

```bash
uv run python -m scripts.teacher_head convert \
  --isolate \
  --source artifacts/quarantine/teacher/5addda86ddee081a68fb067477ea100c221b8917/head.pt \
  --output artifacts/teacher/jev-omni/5addda86ddee081a68fb067477ea100c221b8917/head.safetensors \
  --expected-sha256 8c81edecd733f7db327604803c14cf9ecbee53d2af055eeb09f78b64bcbf2638 \
  --expected-size 3966079 \
  --report reports/teacher-head-conversion.json
```

If an expected tensor spec exists, pass it with `--expected-spec path/to/spec.json`. The converter then rejects missing, extra, shape-drifted, or dtype-drifted tensors before admitting the output.

Normal inference code should consume only the generated `.safetensors` file and its report. Loading `head.pt` outside this conversion command is not an admitted inference path.
