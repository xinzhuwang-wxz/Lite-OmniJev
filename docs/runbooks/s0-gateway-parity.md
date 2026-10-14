# S0 gateway parity runbook

`scripts/s0_gateway_parity.py` is the final post-HF/vLLM gateway harness for the S0 public 100-case cohort. It requires a prior passed `scripts.s0_parity compare-vllm` report and compares the gateway's public `/v1/systemone` response semantics against that frozen non-holdout report.

The gateway does not expose full-vocabulary raw logprobs or K+1 joint distributions. This harness checks only conditional answer probabilities and `p_insufficient`; raw/K+1 evidence must come from the referenced frozen non-holdout HF/vLLM report hash.

## Real run after GPU/server setup

Prepare a gateway operator record that includes:

- `operator_attested: true`
- `model_version`, matching `/health`
- `gateway_launch_command`
- `s0_hf_vllm_report_sha256`, matching the passed HF/vLLM report file
- `upstream_server_record_sha256`, matching the `server.record_sha256` inside the HF/vLLM report
- `provenance`, exactly matching the HF/vLLM report provenance

Then run:

```bash
PYTHONPATH=/path/to/Lite-OmniJev python -m scripts.s0_gateway_parity compare-gateway \
  --manifest artifacts/s0/manifest-100.json \
  --hf-vllm-report reports/s0-parity-vllm-001.json \
  --gateway-record artifacts/s0/gateway-record.json \
  --gateway-url http://127.0.0.1:8000 \
  --output reports/s0-gateway-parity-001.json
```

The command refuses to overwrite reports, checks `/health` for `canonical-chat`, `inference_configured=true`, and the expected `model_version`, then sends each canonical case as a gateway request. Choice option order, score order, and noul labels are derived from the original `Question`; prompts are not reauthored by this script. It runs the full cohort twice and requires identical responses excluding transport timing headers. Conditional probabilities and `p_insufficient` must match the referenced HF/vLLM report with max delta strictly below `0.001`.

## CPU manifest and synthetic controls

```bash
PYTHONPATH=/path/to/Lite-OmniJev python -m scripts.s0_gateway_parity manifest-control \
  --manifest tests/fixtures/s0-prepare-public-001/manifest.json \
  --canonical-report reports/s0-canonical-http-cpu-003.json \
  --output reports/s0-gateway-parity-cpu-001.json
```

This validates the real 100-case development manifest through the public gateway request model and `core_question`, preserving state, media, video frame count, choice option order, score order, noul labels, source family metadata, and answer count against the existing canonical CPU summary. It is still CPU-only and is not S0 acceptance.

```bash
PYTHONPATH=/path/to/Lite-OmniJev python -m scripts.s0_gateway_parity synthetic-control \
  --output reports/s0-gateway-parity-synthetic-control.json
```

The synthetic control uses `httpx.MockTransport` and three synthetic text cases to verify health checks, repeatability comparison, strict probability parsing, report shape, source hashes, and no-overwrite behavior. It is not S0 acceptance.

`compare-gateway` only accepts an HTTP loopback origin such as `http://127.0.0.1:8000` or `http://localhost:8000`; do not point it at external hosts.
