# Valen checkpoint conversion runbook

This runbook covers only the pinned CPU conversion of the nested training checkpoint from `Valen-Team/Valen-Preview-0923` revision `30096aaabb2689614c83a79f64f38c865c2ce1c1`. It converts the checkpoint's published `weights` delta payload to safetensors after deriving an allowlisted tensor spec from the official Valen source loader and the pinned Qwen base model shapes. It does not restore optimizer/RNG/training state and does not run Valen inference or quality admission.

Pinned inputs:

- Source checkpoint: `Valen-Team/Valen-Preview-0923@30096aaabb2689614c83a79f64f38c865c2ce1c1`
- Official source architecture/loader: `Liuziyu77/Valen@42a6a10e127e9248a6d7824299551b78af9fcb97`
- Base model shape source: `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`
- `checkpoint.pt` SHA256: `836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012`
- `checkpoint.pt` size: `1160272498`

Derive the trusted spec before conversion:

```bash
python -m scripts.valen_checkpoint derive-spec \
  --checkpoint-dir artifacts/quarantine/Valen-Team__Valen-Preview-0923/30096aaabb2689614c83a79f64f38c865c2ce1c1 \
  --base-dir artifacts/models/Qwen__Qwen3.5-2B/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --output reports/valen-checkpoint-spec.json \
  --report reports/valen-checkpoint-spec-report.json
```

Convert in the hardened jail:

```bash
python -m scripts.valen_checkpoint convert --isolate \
  --checkpoint artifacts/quarantine/Valen-Team__Valen-Preview-0923/30096aaabb2689614c83a79f64f38c865c2ce1c1/checkpoint.pt \
  --output artifacts/valen/Valen-Preview-0923/30096aaabb2689614c83a79f64f38c865c2ce1c1/weights.safetensors \
  --expected-sha256 836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012 \
  --expected-size 1160272498 \
  --expected-spec reports/valen-checkpoint-spec.json \
  --report reports/valen-checkpoint-conversion.json
```

The isolated conversion path preimports the loader dependencies, enters an empty chroot containing only the copied input checkpoint and preapproved spec, drops to `nobody`, denies network syscalls with seccomp, checks a host canary is inaccessible, and copies the child output back through a no-follow, nonblocking, regular-file-only bounded read. The report must show `status: converted`, `isolation.network_denied: true`, `isolation.host_canary_denied: true`, and the `expected_spec_sha256`, `config_sha256`, and `training_metadata_sha256` values used for admission.
