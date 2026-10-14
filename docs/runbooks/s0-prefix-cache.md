# S0:04 sequential prefix-cache evidence

Status: CPU protocol/mock implementation only. No server was started and no GPU
request was made by this change. This runner does not close S0:04 or its dependencies.

## Frozen input and server records

`eval.prefix_cache.CacheProtocol` defines the local sensitive protocol JSON:
one canonical PNG base64 and its SHA256, shared text, ten distinct question tails,
the official HF processor's expanded token IDs for every tail, the common expanded
prefix, complete unique answer token IDs including U, repeats (2–100), model warmup
request count, seed, and a frozen output-logprob tolerance (0–0.001).
`media_prefix_end` is the exclusive end of the image's final HF placeholder range;
derive it from the actual native processor canary, not an arbitrary positive hit.
The common prefix must cover that range. A warm hit must reach this endpoint;
partial text-only hits do not satisfy the media reuse check. Prefix-block alignment
may prevent this check from passing; preserve the failure rather than lower the endpoint.
PNG header/hash validation is not image decoding or HF processor validation.

`CacheServer` defines the operator-attested record: loopback HTTP URL, unique start
run ID, model alias/revision, weights-manifest hash, dtype, GPU identity, vLLM
0.26.0 or 0.31.0, the shared engine configuration hash excluding the cache switch,
exact credential-free launch argv, cache enabled flag, exclusive-local-test and
raw-logprobs confirmations, and evidence kind. Identity is an operator attestation,
not remote verification of weights. Keep real launch/environment records separately.
The operator must confirm model/GPU/runtime identity and that this test has exclusive
ownership. No production/shared endpoint is eligible for resets.

Start the same engine twice, sequentially, once with `--enable-prefix-caching` and
once with `--no-enable-prefix-caching`. Do not run both instances on one GPU at once.
Both starts need `--logprobs-mode raw_logprobs`,
`--enable-prompt-tokens-details`, `--enable-per-request-metrics`, statistics enabled,
and `VLLM_SERVER_DEV_MODE=1` for native reset routes. Keep all other settings fixed,
including media processor cache size, scheduler limits and GPU allocation. Record
start/end time, command, environment, GPU, model hashes, logs, cost and exit status.
The runner does not launch/stop servers or manage GPUs.

## CPU-only validation

Freeze actual inputs before any GPU request; do not hand-write expanded token IDs.
`scripts.freeze_prefix_cache` reads the unchanged 100-case public development manifest,
selects one single-image family and derives ten explicitly non-quality instruction/qid
variants with the same source metadata. Each variant runs the official local CPU
processor and canonical PNG transport. It extracts the longest common rendered text,
longest common expanded token prefix and last nonzero `mm_token_type_ids` position;
it rejects different canonical bytes/mappings or a prefix that omits the image.
The returned native payload is checked against every actual prepared HTTP message.
It writes sensitive native protocol and public gateway request only to caller-selected
local artifact paths. Keep these under ignored `artifacts/`, never commit them.
Do not use probe variants as training data or quality/annotation evidence.

```bash
uv run python -m scripts.freeze_prefix_cache --manifest tests/fixtures/s0-prepare-public-001/manifest.json --asset-root artifacts/public-media --model-dir artifacts/pinned-processor --protocol-output artifacts/cache/native.json --gateway-output artifacts/cache/gateway.json --report reports/cache-freeze.json
```

Run with an already installed official processor/CPU media dependency environment;
the command never installs packages, downloads weights, or starts a GPU server.
Outputs must not already exist. A failed first attempt is kept as a separate report.
Real Mac evidence in `reports/s0-prefix-cache-official-freeze-003.json` confirms ten
probes from `clevrer-train-00000`, common expanded prefix309 and media endpoint265;
it is processor-only evidence, not native vLLM forward or cache performance.

```bash
uv run python -m scripts.s0_prefix_cache validate --protocol artifacts/cache-protocol.json --server-record artifacts/cache-on-server.json --report reports/cache-protocol-validation.json
uv run pytest tests/test_prefix_cache.py
```

`validate` never creates an HTTP client. Inputs remain local; reports contain input
hashes rather than media/text/expanded token lists. Synthetic MockTransport tests
are timing arithmetic and failure-path evidence only, not native cache performance.

## Future explicitly authorized GPU collection

Each invocation only collects one already-running isolated server. CLI `run`
requires a GPU evidence record and `--allow-reset`; without both exclusive attestation
and this flag it sends no network requests. Proxy environment and redirects are disabled.
Default action selection is explicit; there is no automatic server invocation.

```bash
uv run python -m scripts.s0_prefix_cache run --protocol artifacts/cache-protocol.json --server-record artifacts/cache-on-server.json --allow-reset --report reports/cache-on-run.json
uv run python -m scripts.s0_prefix_cache run --protocol artifacts/cache-protocol.json --server-record artifacts/cache-off-server.json --allow-reset --report reports/cache-off-run.json
uv run python -m scripts.s0_prefix_cache compare --on reports/cache-on-run.json --off reports/cache-off-run.json --report reports/cache-comparison.json
```

Do not execute these collection commands before the project's GPU authorization and
prerequisite gates. Failed invocations leave a new blocked report with safe error type;
do not overwrite old evidence. A mid-run failure does not yield a completed CacheRun.

Warmup requests happen outside measurements. For each repeat, independently prepare
and measure cold-single, cold-ten, warm-single and warm-ten. Each preparation resets
KV prefix, media processor and encoder caches while the prior request is finished.
Warm conditions add one completed fixed primer before the measured sequence. The
single measurement never becomes the ten measurement's primer. Every ten batch is
ten awaited sequential HTTP calls, with no reset/metric polling inside the timer.
All ten share the canonical image/text prefix; only the question tail varies.

Cold first requests must have zero KV cached tokens. On/cold requests 2–10 and
on/warm all requests must report cached tokens reaching the media prefix endpoint.
Off requests must report zero. Each response passes the shared strict native chat
parser: exact expanded IDs, all answer IDs+U, one generated token, unclipped finite
logprobs and exact usage. All repeated/on/off raw outputs must remain within the
frozen tolerance. Nonfinite values, missing metrics, reset failures, missing trials,
changed input identities and output drift cannot become a successful measurement.

Each condition reports all paired `T10/T1`, median and maximum. Every cache-on
paired repeat must be **<=2**, inclusively. Cache-off is a reported control; it is
not expected to attain the cache-on optimization gate. Never divide a warm ten batch
by a cold single. Compare requires distinct start IDs and identical shared engine,
GPU, precision, model, protocol and evidence type. A measured result still sets
`s0_acceptance=false`: S0 dependencies, provenance and full integration are separate.

## Timing and cache boundaries

Per-request HTTP wall and payload JSON preparation are recorded separately; batch
wall spans the complete ordered sequence including local parsing/preparation. Native
`metrics.time_to_first_token_ms` measures scheduled-to-first-token; queue is separate;
`generation_time_ms` measures first-to-last-token decode only and may be zero for
one-token requests. Neither is end-to-end HTTP wall or pure preprocessing time.
The runner records gateway/media-processor durations as null, not invented
subtractions: project gateway and actual media preparation instrumentation remain
integration work before closing the ticket. Report cost explicitly; CLI defaults
to unknown rather than claiming a zero-cost GPU run.

Native reset is engine-wide and could affect others. Both versions return
`{"success": bool}` for `/reset_prefix_cache`; check the body, not just HTTP 200.
Keep `reset_running_requests=false&reset_external=false`. The native media/encoder
reset routes acknowledge HTTP 200 but explicitly do not check internal reset success;
cold media state is an operator/API-acknowledged setup, not independently verified
by that status code. KV cached tokens are `usage.prompt_tokens_details.cached_tokens`.
Media processor/encoder caches and GPU KV prefix caching are different layers.
The native processor cache can be separately disabled with `--mm-processor-cache-gb 0`.

Prometheus offers aggregate token-based prefix queries/hits and item-based media
queries/hits. If captured, retain raw before/after exposition and account for stats
flush intervals; do not insert polling into timed sequences or equate media hits
with prefix hits. Per-request native cached-token evidence is the runner's primary
source. No custom headers or invented response metrics are used.

## Official fixed-version sources

- [0.26 reset routes](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/serve/dev/cache/api_router.py#L20-L66) and [0.31 reset routes](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/serve/dev/cache/api_router.py#L20-L64).
- [0.26 development-route registration](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/openai/api_server.py#L237-L240) and [0.31 registration](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/launchers/api_server/routers.py#L34-L37).
- [0.26 native usage mapping](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/entrypoints/openai/chat_completion/serving.py#L90-L109) and [0.31 mapping](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/chat_completion/serving.py#L1162-L1167).
- [0.26 prefix counters](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/metrics/loggers.py#L584-L602) and [0.31 prefix/media counters](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/metrics/loggers.py#L599-L670).
- [0.31 timing formulas](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/generate/base/serving.py#L50-L92).
- [0.26 CLI](https://docs.vllm.ai/en/v0.26.0/cli/serve/) and [0.31 CLI](https://docs.vllm.ai/en/v0.31.0/cli/serve/).
- [0.26 processor cache size](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/config/multimodal.py#L123-L131).
- [0.31 CPU processor test interface](https://github.com/vllm-project/vllm/blob/v0.31.0/tests/models/multimodal/processing/test_gemma4.py#L218-L240). For both versions use `MULTIMODAL_REGISTRY.create_processor(ModelConfig(...))`, `processor.info.parse_mm_data(...)`, then `processor(prompt, items, hf_processor_mm_kwargs={})`. Compare `result["prompt_token_ids"]` and modality tensors with HF. `cache=` is a 0.31 addition and must be omitted on 0.26. These calls do not load model weights or run model forward; actual CPU execution evidence is required.
