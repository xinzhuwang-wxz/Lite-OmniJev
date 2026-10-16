# Native-course DEV50 mechanism diagnostic

The native curriculum does not repair the event-absence error on this fixed public DEV50. Overall33/50 remains unchanged (1correction,1regression). Brier improves .447486→.437525, but the improvement concentrates in event-present relation probabilities while absent-event and motion probabilities worsen. This small paired slice does not support adopting the candidate or expanding evaluation solely on aggregate Brier.

| Official program / camera proxy group | Count | Correct parent→native | Corrected / regressed | Yes false positives | Brier parent→native | NLL parent→native |
|---|---:|---:|---:|---:|---:|---:|
| Native relations |22|13→13|1 /1|6→7|.515852→.451852|.710532→.634441|
| Target transition absent proxy |8|5→4|0 /1|3→4|.515486→.603399|.709843→.824365|
| Target transition present proxy |14|8→9|1 /0|3→3|.516062→.365253|.710925→.525912|
| Motion |28|20→20|0 /0|8→8|.393770→.426268|.569789→.615129|

All motion predictions are unchanged, so the regression there is probability quality rather than additional classification errors. Overall yes predictions36→38 and yes false positives14→15; native relation yes predictions14→16. The original absence-related yes bias persists, despite26proxy-absent official TRAIN examples in the course. These observations describe this fixed development slice, not a proof of general model incapacity or a proposed threshold fit.

Verification: both completed receipts and record-file hashes checked; adapters bind f2db parent and f5feb native candidate. Evaluation identity is identical except adapter weights/config: same code hashes, source, selected IDs, bf16, official template,16uniform frames and media budget. Each of50ordered IDs, target mappings, prompt/payload hashes, input hashes, complete tensor summaries and their hashes match. Official cached TRAIN question/program/media bindings for these public DEV samples and annotation ZIP were reverified. Grouping reuses the existing bounded postfix/annotation diagnostic;22relation/8absence/28motion counts were asserted. “Absence” remains a camera-transition proxy, not official executor or pixel visibility proof. Original official gold is authoritative.

The Brier and NLL columns use the existing legal yes/no distribution conditioned on excluding U, reconstructed against raw saved logprobs. Exact ties retain the saved legal argmax choice. No threshold changes, new model forward, private pool inspection or new experiment occurred. CPU completed exit0 in5.43seconds; cost null. An initial local-analysis attempt rejected a valid exact tie before output creation; the corrected argmax validation has a regression test.

Artifacts: remote `/root/autodl-tmp/s1-text-lora-001/reports/native-event-condition-dev50-diagnostic-001.json`. Machine report records both receipt and sidecar SHA256 plus official source and helper hashes. Raw50sidecars were not copied into Git. Three targeted CPU tests and Ruff check/format passed. Main task retains the decision whether a150sample retention check is justified; this diagnostic recommends retaining f2db based on the absence regression and unchanged total accuracy.
