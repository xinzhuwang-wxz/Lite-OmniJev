# Single-arm polarity launcher: CPU validated, GPU pending

Explicit `--arm control|polarity` is required. Default validates only; `--execute` runs one arm and one fixed original temporal DEV50. No second arm, queue loop, quality promotion or training-set complement score is scheduled. Root chooses whether to run polarity after reviewing control.

Both real remote CPU dry validations passed on core3fe0ab2/archive9132c2d…: frozen prototype report and all chosen-arm outputs, explicit TRAIN ledger120, canonical schema/config, existing isolated inventory receipt, f2db parent weight/config and terminal source training report, official model chat_template SHA, trusted core Python files against archive bytes, fixed DEV runner and baseline code/records/source receipt hashes. No complete historical inventory re-audit, GPU query/model load, key read or telemetry submission occurs during dry validation. Eleven launcher CPU tests pass (budget/objective/data/arm/isolation/overwrite boundaries); builder's nine tests remain passed. Ruff/mypy/CLI checks pass. Companion JSON records actual immutable validation paths/hashes and argv.

Root-reviewed control execution command (not executed by this task):

```sh
PYTHONPATH=/root/autodl-tmp/s4-claim-pair-code-001 \
 /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python \
 /root/autodl-tmp/s4-video-polarity-launch-code-002/launch-video-polarity-pilot.py \
 --arm control --execute
```

Polarity requires a separate identical invocation with `--arm polarity`. Dry reproduction omits `--execute`; optional `--validation-report NEW_PATH` refuses overwrite. Execution refuses existing run/report/dev outputs and a nonterminal other-arm launcher. Preflight is capped180s; training240s via timeout INT/20s cleanup, checkpoint5/final10; fixed seed42/120draw/accum12/CE0/refKL0/Brier0. Both arms initialize from f2db. SwanLab uses core real step metrics in Lite-OmniJev/maxen; the key is read only for real execution from the existing outside-repository file and never included in serialized argv/receipts. GPU busy causes failure, not waiting or a queue.

After completed10 training, a separate process invokes the unchanged fixed DEV50 runner: same original50 questions/inputs, bf16,16uniform frames/video softtoken70, one pass. It performs CPU frame-manifest preparation and evaluation inside a single outer180s deadline (no extra unbounded preparation time); underlying progress.json updates at25/50. Original baseline identity and each record's id/input_sha/input_summary_sha/answer mapping must match. Candidate and parent metrics are reported for root review; no threshold or stage gate is applied. TRAIN complement consistency is not evaluated or presented as generalization.

Exposure workflow: after each arm is terminal, reconcile that arm's runtime data/canonical Samples, actual init, training receipt/steps and source allocation ledger into the latest inventory before proceeding to the next arm. Any failed training with steps>0 still creates real exposure, even if DEV fails. Launcher retains actual terminal/partial training receipt hashes and marks exposure reconciliation pending; it does not implement or claim that reconciliation. Each arm's source inventory is independently isolated; shared replay IDs mean the two prepared manifests must not simply be concatenated as current. Root should mark the previous arm's actual exposure historical and append the next one. Shared30 video families are a matched training variable, not an unseen/generalization claim. Original DEV50 and later original DEV quality remain adoption criteria.

Remote outputs use `s4-video-polarity-{arm}-train-001`, matching training/preflight/launch reports, and `s4-video-polarity-{arm}-dev50-001`. New files are private; failures retain launch receipt and existing partial evaluator receipts. No root STATUS/ticket, trainer, builder or other coverage helper was modified. GPU training/evaluation and live SwanLab remain untested by this CPU delivery.
