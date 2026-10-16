# Final step100 public500 operational preparation

CPU preflight succeeded for image/audio/text/screenshot and both original/supplemental crossmodal sources, each with actual500unique public DEV originals. Single-media runs took about.19seconds each, cross sources about.4seconds. No media bytes or GPU were read during preparation; small model/tokenizer/template assets, selected data, actual adapter and fixed source were hash-verified. Base model weights are rehashed on explicit execution. All six receipt hashes/source counts/protocols are in the aggregate JSON. No new datasets, private/calibration/holdout reads or training occurred.

The runner fixes actual completed100step report421b9506… and adaptere4655014… at `artifacts/s4-program-stratified-pool-train-001/adapter-invocation-step-0100`. Ordinary cohorts reuse the exact cached passedS3data/commands with one pass, original questions, bf16,1120image budget and monitored official runtime74e96e2/archiveb8c5…. Those S3reports supply protocol/data identity only: their adapter1beb…is not the working f2db parent and their scores are not inherited. Root already has a separately measured f2db image500410/500reference. This helper does not claim historical full tensor identity where the S1sidecars lack it.

Crossmodal original source really has500rows at `/root/autodl-tmp/s0-crossmodal-development-001/artifacts/crossmodal-dev-001-a/dev.jsonl` (SHA8099d5c7…), and supplemental really has500 at `artifacts/s4-gold-inventory-001/crossmodal-dev-002/dev.jsonl` (SHA1d2d8e15…). Both media roots exist. There is no missing-full-source gap in this environment. The wrapper SHA-pins and reuses the existing full_forward function, preserves its official input/template/image1120/answer-token mapping/raw probabilities, and passes all500sorted samples directly; it does not invoke the old default100slice or a screening threshold.

After execution, complete500original IDs/targets and adapter/data/report/sidecar bindings are checked. It preserves existing accuracy/Brier/ECE15/coverage/count and computes conditional NLL plus full legal-answer+UjointNLL from raw scores, without truncation or dropping U. Raw logits/probabilities remain in ignored output sidecars. S1statuspassedmeans the evaluator completed, not S4quality acceptance; wrapper stage_gatefalse/quality_inheritancefalse. Actual formal quality decisions remain root-owned.

One explicit cohort/source per invocation; `--execute` is required for GPU and a busy GPU is rejected. Default600second evaluator limit, optional420, no auto-all-cohort runner. Ordinary CLI uses timeoutINTwith20second kill grace; fullcross has its bounded signal deadline. Failures preserve wrapper receipts and evaluator logs/partial evidence; public progress is every25originals. Both routes validate complete legal+Ureadouts. Two CPU guards cover real500/public-only scope and conditional/jointNLLincludingU; Ruff check/format passed. No GPU result is claimed here.

Root-only first image invocation:

```sh
cd /root/autodl-tmp/s4-monitored-eval-code-001
PYTHONPATH=. /root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python /root/autodl-tmp/s4-native-event-cpu-001/evaluate-program-stratified-public500.py --cohort image --output /root/autodl-tmp/s1-text-lora-001/artifacts/s4-program-public500-image-001 --execute
```

Choose exactly one `--cohort audio|text|screenshot` with a new output directory for another ordinary job. For one full cross source use `--cohort crossmodal --cross-source original` or `supplemental`, again a new output. Root queues serially after video selection. Without `--execute` the same entry only validates CPU bindings and writes a preparation receipt. Runtime/source remains74e96e2for all jobs; fixed cross helpers are separately bound at7176d167…/0c9c616f… . No core/default evaluator or thresholds were edited.
