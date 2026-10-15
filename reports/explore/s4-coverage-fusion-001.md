# S4 coverage delta midpoint: one bounded candidate

Status: CPU implementation verified; deferred by current user direction.
Do not generate the real fusion or run GPU screening now. Main session prioritizes
fuller development evaluation of existing coverage adapter 0deef, and no longer
treats 34 versus 35 on a 50-item screen as an automatic rejection. The historical
thresholds below document the optional bounded runner, not a current promotion
requirement. Final S4 quality gates remain unchanged.
This is public-dev method screening, not S4 acceptance or sealed evidence.

## Decision and evidence

The TRAIN-only diagnostic completed on GPU in 112.81 seconds (main-session
measurement). Video/cross gradient cosine was .0176, .0126, -.1109: negative in
only one of three steps. Total/cross cosine was .834, .749, .839: positive in
all three. The frozen projection trigger (video/cross negative in at least two
steps and total/cross negative in at least one) did not fire. No PCGrad is added.
These fixed-weight gold-CE gradients do not establish behavior of AdamW's
momentum/preconditioned updates.

Coverage adapter 0deef improved video from 99 to 108/150 but crossmodal original
fell from 35 to 34/50. One midpoint tests whether a smaller effective update can
retain the original crossmodal floor while achieving the video floor. This is a
new combination. The older failed fusion used video-CE adapter 14496 and is not
evidence that this combination has passed or failed.

## Frozen candidate

- First: `6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac`.
- Second: `0deef8cfd3d94f4dd3b42cfdbe995c825c388230721df7e6716e5963cb9dc818`.
- Alpha .5 only, seed 0. No .25 continuation, ratio search, retraining or engine change.
- Reuse `scripts/lora_fusion.py`: rank16 optimal effective-delta approximation,
  per-layer relative Frobenius error at most 1e-3. Approximation failure stops.
- Both source adapter configs must be explicitly SHA-pinned. Screen identity
  binds source/config/output hashes, fusion report, fusion code, shared evaluator,
  frozen source manifest/commands and runner. Resume rechecks prior stage evidence.

## Execution and stop condition

`.scratch/s4/screen-coverage-fusion.py` defaults to CPU fusion and preparation.
It imports the existing coverage screen's evaluator functions, fixed source pins,
order and thresholds. GPU loads only under explicit `--execute`.

Run from the immutable code snapshot root:

```sh
python .scratch/s4/screen-coverage-fusion.py \
  --expected-first-config-sha256 FIRST_CONFIG_SHA \
  --expected-second-config-sha256 SECOND_CONFIG_SHA
```

Default first/second source directories are the existing joint-pilot and
crossmodal-retention `adapter` directories. Output fusion is
`artifacts/s4-coverage-fusion050-001`, screen receipt is
`artifacts/s4-coverage-fusion050-screen-001/receipt.json` under the existing
remote project root. Explicit path overrides support CPU fixtures and immutable
remote layout; source weights remain fixed.

Add `--execute --resume` for video; after a passed video stage use
`--execute --resume --through text` to continue. Frozen order is video → original
crossmodal → supplemental → audio → image → screenshot → text. Minimum correct:
102/150 (temporal≥60, motion≥7), 35/50, 40/50, 37/50, 40/50, 50/50, 49/50.
Any failure stops immediately. No new candidate follows this one automatically.

## CPU verification

- Real CPU rank16 synthetic adapter fusion through the public entry: prepared
  receipt produced, no evaluator called; same-identity resume passed; changing
  fusion report alpha to .25 produced an execution_failed receipt and exit 1.
- Synthetic source hashes were substituted only in the temporary CPU test module;
  shipped source pins above are immutable. No synthetic output is quality evidence.
- Ruff check/format and CLI help passed. Existing fusion behavior tests: 9 passed (real CPU torch).
- No real source weights are available in this local worktree, so actual residual,
  output SHA, and public-dev quality remain pending remote execution.
