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
