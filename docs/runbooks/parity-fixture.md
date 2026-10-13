# P0 parity fixture

`scripts.build_parity_fixture` creates two text-only program-truth batches for
numerical parity policy work. They are not training data, not a production
benchmark and not a holdout substitute.

```bash
uv run python -m scripts.build_parity_fixture --output-dir artifacts/data/parity-fixture-v1
```

The command writes:

- `calibration.jsonl`: 100 `calibration` pool samples.
- `confirmation.jsonl`: 100 `dev` pool samples.
- `manifest.json`: file SHA-256, row count, pool, generator families, effective
  sample families, generator source hashes, independence limits and quota counts.

The builder refuses to overwrite an existing `calibration.jsonl`,
`confirmation.jsonl` or `manifest.json`. Use a new output directory for each
regeneration so prior GPU evidence remains addressable by hash.

Both files use one question per sample and the existing `Sample` schema. The
quota is fixed per file: Choice 60 (2/3/5/10/20 options, 12 each), Noul 20 and
Score 20 (2/3/5/10 levels, 5 each). Missing or contradictory required facts are
explicit `insufficient` targets, so U behavior is testable.

`generator_family` is deliberately not equal to sample family. Calibration uses
`calibration-interval-register-v1`; confirmation uses
`confirmation-event-sequence-v1`. Each file has one combined generator family
with three task rules: Choice, Noul and Score. Each row still has a unique
`family_id` based on the rendered state, target-free question, case status and
generator identity. This supports 100 effective case families without pretending
there are 100 independent algorithms or statistically independent production
samples.

Use `data.parity_fixture.load_calibration(path)` as the public loader for the
calibration batch. It requires the basename `calibration.jsonl`, exactly 100
rows, text-only samples, one question per row and pool `calibration`.

## GPU calibration diagnostics

Use the training environment for `python -m scripts.parity_calibration` and the
isolated vLLM environment for serving. `--fixture-dir` always reads only
`calibration.jsonl`; it rejects train/dev/holdout samples even if the manifest
hash matches. Confirmation inference is deliberately not exposed here.

Run `hf-reference` for base and `--adapter` separately, followed by
`compare-vllm` against an operator-attested server record, as in the P0 parity
runbook. Required arguments are `--fixture-dir`, `--model-dir`,
`--asset-manifest`, and a fresh `--report` path; comparisons also require
`--reference` and `--server-record`. Records bind the same provenance, raw
logprobs mode, served model, dtype, and actual launch command. The remote
orchestration scripts and logs are experiment artifacts, not production jobs.

`--dtype bfloat16` is the default; `--dtype float16` is an explicitly recorded
alternative precision experiment. Both use the existing model and independent
LoRA. The original P0 command and strict threshold remain unchanged. Never
compare references across dtype or changed implementation hashes.

Reports include known-answer conditional metrics separately from legal-joint
accuracy/Brier including U, p(U) MSE by sufficient/insufficient truth, and Score
expected-value deltas. Argmax ties are not credited as a unique correct joint
answer. These are program-fixture results, not production calibration or a
release verdict. Strict numerical comparison failures are preserved even when
most labels agree. Follow with `scripts.parity_impact` for paired error slices.
