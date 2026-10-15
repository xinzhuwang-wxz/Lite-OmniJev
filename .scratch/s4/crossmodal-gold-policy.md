# S4 crossmodal gold construction policy

Frozen before real composition. This is construction, not gold or stage admission.

- Preserve original 500 crossmodal dev items and component ledger byte-for-byte.
- Pair existing same-pool VQA Choice gold with Clotho Noul / LibriSpeech Choice gold. No source relabeling, pool reassignment, teacher gold, or repeated proposal variants to inflate counts.
- Calibration: 3000 unique image question items paired one-to-one with 1400 Clotho and 1600 speech question items. Holdout: 1000 unique image items with 500 of each audio source. Hash-order seed42; independently hash-order pairs before assigning 50% both true, 25% image false, 25% audio false.
- Choice claims carry full options and proposed option text; false claims use another legal answer. Output gold is exact conjunction of source truth. Original source samples are unchanged.
- Counting follows the existing S4 assessment: frozen original question identities, with family/media/dependency counts separately. Same-pool shared components do not establish IID observations. Report dependency groups, maximum reuse and effective sample size as unestimated. Cross-pool family/media/component overlap rejects construction. Final independent-holdout acceptance remains to be verified; do not turn these counts into IID confidence intervals.
- Keep original crossmodal dev quality visible. Before S4 quality acceptance, construct/freeze source-matched supplemental development diagnostics and compare matched teachers on each source slice; changed source mix cannot hide regression on the original cohort.

## Integration sequence

1. Complete VQA plan002 selected downloads; preserve failed/budget-limited receipts and resume only missing bytes. Materialize with the bound plan, complete download manifest and explicit media root. No partial corpus admission.
2. Merge VQA-added artifacts into GUI002 combined inventory (both were independently based on audio inventory); do not discard GUI/audio/video history or component ledgers. Audit merged manifest.
3. Bundle ONLY selected VQA/Clotho/speech calibration+holdout source artifacts into one trusted media output using explicit per-artifact roots. Clotho may have legacy relative and new absolute file URIs, so use declared root union; ambiguity rejects. The bundler's inventory is a subset, never a substitute for global history.
4. In a derived combined manifest, retain each original source artifact as historical exposure and replace its current reference with the rewritten artifact; preserve every other entry and every component ledger. Bind mapping/receipt/output hashes; rerun global isolation. URI relocation changes content fingerprint but preserves original ID, family, media and source-hash aliases.
5. Compose crossmodal gold against that derived full inventory and the bundled current sources, using the bundle output as sole media root. Audit final global component graph. All private data remains remote; export aggregates only.
6. Finish source admission, source-matched dev, controlled job permission/log/error rejection evidence, and full gate acceptance before S4 expansion. Existing metadata audit/build success alone cannot close s0:05.
