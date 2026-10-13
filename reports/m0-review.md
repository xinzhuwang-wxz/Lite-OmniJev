# M0 independent review — 2026-10-13

Reviewed commit: `0a8671a5ad907b17c2d08e155bcc98dca4f7e30e`.
Reviewer worktree: `/Users/bamboo/Githubs/Lite-OmniJev/.worktrees/review`.
Scope: existing M0 schema, template/readout, metrics/gate, assets CLI, text gateway, local queue and documentation. The concurrently written ten-step training runner is outside this snapshot. This report writes no application code and does not claim GPU/model acceptance.

Verdict: **changes required** before M0 closes. Two P1 and four P2 findings reproduce below. No P0 finding. All 75 existing tests passed, so the counterexamples need behavior regressions rather than a rerun of unchanged tests alone.

## Findings

### F1 · P1 · Automation coverage can pass by cutting through a tied-confidence group

Location: `eval/metrics.py:55–60`; current regression `tests/test_core.py::test_metrics_exact_definitions_and_confidence_boundaries` also locks in the issue.

Public reproduction: `evaluate([[0.8,0.2]] * 100, [0]*65 + [1]*35)["coverage_at_error"]` returns `0.68`; reversing the labels returns `0.0`. All 100 examples have identical confidence, so a confidence threshold can accept either all (35% error) or none. A 68% coverage claim at 5% error is not implementable and would exceed the S5 minimum 65% on this counterexample. The documented stable-sort diagnostic does not make it suitable for the project's automation-coverage gate.

Fix: evaluate eligibility only after entire equal-confidence groups; continue scanning later groups because a longer prefix can recover after a violation. Keep the warning that selecting a deployment threshold on holdout labels is prohibited. Regressions should assert permutation invariance within ties and retain the recovery-after-violation case.

### F2 · P1 · Full asset verification reports ready with no model weights

Locations: `scripts/assets.py:113–132` and `scripts/assets.py:204–210`.

Public reproduction: create a valid pinned manifest containing only `tokenizer.json`, content `{}`, size 2, and its correct SHA-256; place that file at `model_directory(root, model)`. `prepare_assets(manifest, root, metadata_only=False)` returns:

```json
{"status":"all_files_verified","files_verified":1,"bytes_verified":2,"weights_ready":true}
```

`plan_assets` checks some completeness, but `download`/`verify` ingest persisted JSON independently and do not repeat it. A partial/edited inventory can therefore trigger the no-card → GPU-ready transition despite absent config/weights. A sharded inventory also needs confirmation that every index-referenced shard is present; verifying only the listed surviving shard cannot establish completeness.

Fix: validate manifest structure and the required model artifact set on every entry point, including model weights and tokenizer/config requirements; reject missing/duplicate shard entries and ensure index references match the downloaded set before `weights_ready=true`. Keep metadata-only verification explicitly distinct from model readiness.

### F3 · P2 · Score legend changes the caller's rubric values

Location: `serve/gateway/app.py:206–209`.

For Score criteria `["low","high"]`, the HTTP response is:

```json
{"legend":{"0":{"description":"low","value":0},"1":{"description":"high","value":1}}}
```

The [TypeSafe API](https://docs.typesafe.ai/api.md) and pinned [OneJev schema](https://github.com/OmniJev/OneJev/blob/6122519058af23504a33d944f0e63b2d868c8a4b/qev/schema.py) use level-index → original criterion entry. Expected here: `{"0":"low","1":"high"}`. A client displaying/looking up the supplied text instead receives invented objects. ADR-0002 discloses a confidence formula difference, not a legend contract change.

Fix: return each original criterion under its stringified index and assert exact rubric round-trip via the HTTP interface. This does not change the frozen Score confidence formula.

### F4 · P2 · Whitespace-normalized duplicate options raise HTTP 500

Locations: `serve/gateway/app.py:45–53` and `serve/gateway/app.py:151`.

An otherwise valid request with Choice criteria `{"a":" A","b":"A"}` passes the outer raw-string distinctness check. `core_question` constructs `Question`, whose `Nonempty` strips whitespace; both descriptions become `A`, and its duplicate validator raises an uncaught ValidationError before the gateway's try block. Against ASGI with `raise_app_exceptions=False`, the request returns `500 Internal Server Error`.

Fix: use the same whitespace normalization at the public validation boundary, or deliberately translate internal request validation failures to 422. Verify malformed criteria are rejected before any backend call.

### F5 · P2 · Asset prohibition accidentally rejects the approved OneJev-27B teacher

Location: `scripts/assets.py:52`.

`plan_assets({"models":{"teacher":{"repo_id":"OmniJev/OneJev-27B","accepted_licenses":["apache-2.0"]}}}, ["teacher"], mock_api)` immediately raises `ValueError("Unresolved or prohibited model: teacher")`, and `mock_api.model_info` is never called. The substring `JEV-27B` matches `ONEJEV-27B`, so filling the deliberately unresolved teacher configuration can never select the allowed teacher.

Fix: compare normalized model basename against the explicit forbidden `JEV-27B` / `JEV-27B-VL` names (or exact reviewed repository IDs), not a substring that also matches OneJev. Preserve license/provenance review and the intentional disabled-by-default teacher configuration.

### F6 · P2 · Positive numbers are accepted as raw log probabilities

Location: `serve/gateway/app.py:174–178`.

A backend response with selected-token values `{A:3.0, B:2.0, U:1.0}` passes the raw-logprob checks and produces HTTP 200 with ordinary-looking probabilities. Actual normalized log probabilities cannot be positive; these numbers can instead be logits or a misconfigured/protocol-broken backend. Finiteness and the `-9999` lower sentinel alone do not validate the declared contract.

Fix: reject selected values greater than zero (or an explicitly justified tiny numerical tolerance), in addition to current missing/nonfinite/clipped failures. Test this at the HTTP boundary. The operator raw-mode confirmation and later HF/vLLM parity gate remain required; this range check does not prove provenance by itself.

## Additional accounting observation

The upstream Completion API samples one token and may report `completion_tokens=1`; the gateway returns `usage.output_tokens=0` and discards that backend cost. Zero product output tokens can be an intentional typed-decision API convention, so this is not a demand to add M0 production billing. Preserve backend input/output token accounting separately, or document that the field measures product output only, before using these responses in cost/capacity reports.

## Verification performed

- `PYTHONPATH=. /Users/bamboo/Githubs/Lite-OmniJev/.venv/bin/python -m pytest -q`: **75 passed in 0.43s**.
- Same interpreter with `-m scripts.loop validate`: valid, 11 tickets, no ready frontier, bootstrap:01 claimed; current P0 and FINAL dependencies remain blocked.
- Executed focused public-interface Python probes for F1/F2/F5 and ASGI HTTP probes for F3/F4/F6. They used deterministic fixtures and temporary files, no live model/network/GPU.
- Inspected actual pinned local Gemma tokenizer metadata: rendering includes `<bos>`. The downloaded tokenizers runtime produced identical token IDs with add_special_tokens true/false for the tested rendered prompt; no unsupported double-BOS finding is asserted. CUDA/Transformers/vLLM token parity still needs its planned real probe.
- Root instructions are within the requested short project contract; CLAUDE delegates to AGENTS. STATUS, research and browser reports distinguish CPU/fixture checks from real model quality. Existing browser evidence covers `/docs`, health and explicit unconfigured-backend failure, which is the current M0 scope.
- Production authentication, multimodal processing, final Playground, GPU numerical parity, teacher scores, human holdout agreement, and three-platform release acceptance are explicitly deferred; their absence is not counted as an M0 defect here.

Parent is implementing fixes. This snapshot review is not a post-fix approval: record follow-up commit and fresh targeted/batch evidence before changing the verdict.

## Post-fix verification addendum — 2026-10-13

Verified fix commit: `06a9cdca93d9f39887846c8dc2d95992d97052e0`. This addendum supersedes the initial changes-required verdict **for F1–F6 and the reviewed M0 snapshot only**. All six findings are resolved; no remaining blocker was observed in this focused recheck. The concurrent training runner is still outside this review.

Independent rerun: `PYTHONPATH=. /Users/bamboo/Githubs/Lite-OmniJev/.venv/bin/python -m pytest -q` → **84 passed in 0.41s**. Then reran the original public-interface counterexamples without editing tests or application code:

| Finding | Fresh result |
|---|---|
| F1 | 100 equal-confidence rows with 65 correct/35 incorrect report coverage `0.0` in either order. The 20-row case recovering to 5% error still reports `1.0`. |
| F2 | Metadata-only inventory raises incomplete-inventory ValueError. Additional probe with both correctly named/checksummed shards present but an index referring to only one shard also raises index-mismatch ValueError. Existing tests confirm metadata-only download still reports `weights_ready=false`. |
| F3 | Score criteria `["low","high"]` round-trip as `legend={"0":"low","1":"high"}` through HTTP 200. |
| F4 | Whitespace-equivalent Choice descriptions return HTTP 422 with no additional backend call. |
| F5 | `OmniJev/OneJev-27B` reaches the mocked metadata client; exact prohibited basenames `JEV-27B` and `JEV-27B-VL` fail before any client call. License/source review is still required. |
| F6 | Positive raw logprobs return HTTP 502 and no fabricated answer. |

The accounting observation is also addressed: the HTTP probe preserves product `usage.output_tokens=0` while independently returning `backend_usage.output_tokens=1` for the actual backend work.

No live GPU, teacher, human annotation, external leaderboard, complete SDK, multimodal or final release acceptance claim follows from these tests. Parent remains responsible for integration evidence and the separately reviewed training runner before marking the whole milestone complete.
