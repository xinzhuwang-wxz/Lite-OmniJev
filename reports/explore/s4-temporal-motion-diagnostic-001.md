# Temporal curriculum and motion sampling: paired CPU diagnosis

Both analyses use completed public-dev outputs; no GPU work was launched here.
The JSON reports bind source/reports/sidecars/code SHA and preserve evidence gaps.

## Temporal curriculum versus CE0 on video150

Source150, IDs/order, hard targets, model/evaluator/precision/media budgets match.
Historical s1 sidecars have no input/tensor hashes. Their matching family_hash is
not an input hash; per-tensor input equivalence remains unverified.

| Group | CE0 correct | Curriculum correct | Corrected/regressed | NLL delta | Brier delta |
|---|---:|---:|---:|---:|---:|
| All150 |109|110|4/3|+.031358|+.023098|
| Temporal97 |68|70|4/2|+.015101|+.012141|
| Motion14 |11|10|0/1|+.014812|+.017447|
| Existence39 |30|30|0/0|+.077729|+.052376|

Total yes predictions rose79→82; temporal yes49→53. NLL improved on51rows and
worsened on90. The +1 is a small net decision-boundary redistribution, including
+2 temporal and -1 motion, accompanied by worse probability quality in every
subgroup. This does not establish a robust curriculum benefit or a model limit.
A single same-data/same-init CE+Brier arm is reasonable to test whether the
probability deterioration can be reduced, with existing fixed budgets and no
weight sweep. It is supervised calibration-sensitive training, not RL; success
must include probability losses and retention, not just one additional argmax.

Evidence: `reports/s4-temporal-curriculum-paired-002.json`.

## Motion versus fresh uniform16 on fixed temporal50, same CE0

Both30/50: one corrected and one regressed. NLL .655922→.670081 (+.014159),
Brier .472287→.476211 (+.003924). NLL improved21/worsened26; yes/no counts stayed
25/25. This provides no reason to expand or tune motion sampling now.

Prompt/payload/media/gold/answer mapping and attention/modality mask hashes match.
Video pixel hashes differ50/50. Expanded input IDs differ48/50, consistent with
processor expansion from changed video indices/time metadata; token-level causes
cannot be localized from hashes. Video-position hashes differ0/50. All tensor
shapes/dtypes match; both policies preserve source FPS/count/duration and their
CPU-manifest indices. This is a controlled index-policy comparison, not a claim
that the full processed tensors should be identical across policies.

Evidence: `reports/s4-motion-coverage-paired-001.json`.

## Next diagnostic boundary

S3 exposure evidence records200unique actual video families/draws among1000total
training draws/100steps, not the entire8299video pool. Short-pilot +1 cannot
identify a backbone capacity ceiling. Before further broad optimization, a
fixed balanced12TRAIN-family repeated120draw test can separate failure to learn
those examples from failure to generalize. It must use audited TRAIN selection,
complete before/after input hashes and a separately bounded dev check. No such
new training or GPU execution is authorized by this report itself.
