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
The isolated path also requires a preapproved tensor spec derived from `decision_config.json` and `_Head256` in the pinned source before loading `head.pt`.

```bash
uv run python -m scripts.teacher_head convert \
  --isolate \
  --source artifacts/quarantine/teacher/5addda86ddee081a68fb067477ea100c221b8917/head.pt \
  --output artifacts/teacher/jev-omni/5addda86ddee081a68fb067477ea100c221b8917/head.safetensors \
  --expected-sha256 8c81edecd733f7db327604803c14cf9ecbee53d2af055eeb09f78b64bcbf2638 \
  --expected-size 3966079 \
  --expected-spec configs/teacher-head-jev-omni-spec.json \
  --report reports/teacher-head-conversion.json
```

The converter rejects missing, extra, shape-drifted, or dtype-drifted tensors before admitting the output. If the host cannot chroot to a jail containing only the copied input, expected spec, and output directory, `--isolate` must stay blocked and must not be treated as admitted.

Normal inference code should consume only the generated `.safetensors` file and its report. Loading `head.pt` outside this conversion command is not an admitted inference path.
