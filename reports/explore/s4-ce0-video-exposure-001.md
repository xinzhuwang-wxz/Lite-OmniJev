# CE0 verified video exposure and coverage50 paired diagnosis

Weight receipts verify S3stable1beb → firstjoint6bb → crosscoverage0deef → CE0
4b8e. The segment begins at video introduction in S3; its db45 upstream adapter
is anchored by the S3 init receipt, not recursively audited here. Only saved
TRAIN inputs and fixed seed42 shuffle/modulo actual completed draws were used.

| Stage | Total draws | Video draws | Video families | New segment families |
|---|---:|---:|---:|---:|
| S3 stable |1000|200|200|200|
| First joint |300|50|50|0|
| Cross coverage |360|150|150|150|
| CE0 |120|40|40|0|

Union: **440video draws,350unique video IDs/families**. Union subgroup families:
216temporal,50motion,84existence. Firstjoint's actual video exposure was50,
not150; the later coverage run contributed150new families. The full8299admitted
pool and discarded experimental branches are not counted as CE0 training.

The new60coverage videos intersect CE0's verified segment in only2S3 families
(clevrer:5334,clevrer:8188), leaving58new-to-segment families. They exclude the
old temporal curriculum60, as promised, without claiming all-history exclusion.

Fresh fixed temporalDEV50 on identical full input hashes gives30→33/50:
7corrected,4regressed; NLL .655922→.631715, Brier .472287→.447486. Yet yes
predictions25→36 on balanced25yes/25no gold, so the net gain is accompanied by a
large yes shift (yes recall15→22/25, no recall15→11/25). This is useful but not
sufficient evidence for promotion or another training segment. Larger video150
and old-modal/cross retention are the root-owned next checks. A second segment
may be CPU-prepared only, excluding the verified350plus first60; parent selection
and GPU execution remain pending those outcomes.

Evidence: sibling `.json` lineage/source/receipt bindings and `.dev50.json`
complete-input paired probability metrics. CPU analysis only; no GPU launched.
