# RouterBench CPU pickle preparation

This runbook converts the pinned public RouterBench pickle files to bounded JSONL without loading pickle data in the normal project process.

Inputs are fixed to `withmartian/routerbench@784021482c3f320c6619ed4b3bb3b41a21424fcb`:

- `routerbench_0shot.pkl`, 99,567,659 bytes, SHA-256 `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`
- `routerbench_5shot.pkl`, 171,091,324 bytes, SHA-256 `fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe`

Run from a Linux root environment with a reader Python that has pandas, numpy, and pyarrow:

```bash
flock /tmp/omnijev-cpu-heavy.lock \
  python scripts/prepare_routerbench.py convert \
  --input-root /root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/decision-index-public0.3/raw/routerbench \
  --output-dir /root/autodl-tmp/lite-omnijev/artifacts/data/prepared/routerbench-jsonl \
  --report reports/routerbench-cpu.json \
  --reader-python /root/autodl-tmp/envs/offline-tools-canary-full/bin/python
```

The parent starts the child with a fixed minimal environment: explicit `PYTHONPATH` for this checkout, offline flags, empty CUDA visibility, one-thread BLAS settings, and no inherited `PYTHONHOME` or library path. The child preloads pandas/numpy/pyarrow, installs the teacher-head network-deny seccomp policy, chroots into a jail containing only copied input and output directories, enables no-new-privileges, drops to nobody, proves host-canary plus socket denial, applies CPU/file/FD/process/address-space limits, then calls `read_pickle`. JSONL is streamed row-by-row to avoid a full serialization buffer. The parent refuses output overwrite, copies child output into hidden staging with nofollow bounded reads, validates JSONL and a strict allowlisted child report, then atomically promotes the final JSONL. Reports include only counts, schema hashes, file hashes, and resource probes.

The resulting JSONL is public evaluation source material only. It is not admitted to training, does not touch sealed data, and does not use GPU.
