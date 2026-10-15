"""Assemble observed data admission; model evaluation gate remains a separate obligation."""
import hashlib,json
from pathlib import Path
names=['s4-selected-gold-audit-001.json','s4-source-evidence-chain-001.json','s4-source-code-closure-001.json','s4-text-gold-source-replay-001.json','s4-clotho-preserved-source-001.json','s4-media-normalization-attestation-001.json','s4-known-exposure-reconciliation-001.json','s4-observed-image-rgb-audit-002.json','s4-private-inventory-job-003.json','s4-private-failure-probes-002.json','s4-private-identity-negative-probes-001.json']
r={n:json.loads((Path('reports')/n).read_bytes()) for n in names}
assert all(v['exit_code']==0 for v in r.values())
selection=r[names[0]];assert selection['outcome']=='selected_counts_and_targets_verified'
assert r[names[1]]['outcome']=='mechanical_source_bindings_verified' and len(r[names[1]]['sources'])==7
assert len(r[names[2]]['jobs'])==8 and all(j['mismatched_or_missing']==0 for j in r[names[2]]['jobs'])
assert r[names[3]]['outcome']=='matched_original_gold'
assert r[names[4]]['outcome']=='original_mapping_verified' and r[names[4]]['preserved_items']==120
assert r[names[5]]['outcome']=='uri_only_changes_verified'
assert r[names[6]]['outcome']=='known_exposures_reconciled'
assert r[names[7]]['outcome']=='isolated' and r[names[7]]['receipt']['isolation']['overlapping_components']==0
job=r[names[8]];assert job['outcome']=='isolated' and job['ordinary_identity_boundary'] and job['source_manifest_sha256']==selection['inventory_sha256']
assert r[names[9]]['passed'] and all(r[names[9]]['checks'].values()) and r[names[9]]['code_sha256']==job['code_sha256']
assert all(r[names[10]][k] for k in ('live_unregistered_identity_rejected','used_identity_rejected','retained_runs_root_only','prior_workers_denied_retained_runs'))
manifest=Path('.scratch/s4/selected-gold-manifest-001.json');assert hashlib.sha256(manifest.read_bytes()).hexdigest()==selection['manifest_sha256']
assert r[names[6]]['inventory_sha256']==job['source_manifest_sha256']==r[names[7]]['receipt']['manifest_sha256']
for cohort,counts in selection['selected_question_counts'].items():assert counts['calibration']==3000 and counts['holdout']==1000 and selection['original_development'][cohort]==500
assert len(selection['selected_question_counts'])==6
bindings={str(Path('reports')/n):hashlib.sha256((Path('reports')/n).read_bytes()).hexdigest() for n in names}
evidence_sha=hashlib.sha256(json.dumps(bindings,sort_keys=True,separators=(',',':')).encode()).hexdigest()
result={'schema_version':'s4-gold-data-admission-v1','status':'data_admitted','scope':'Selected existing-gold data and observed ordinary-identity data-audit workflow; not private model execution or stage quality acceptance','selected_manifest_sha256':selection['manifest_sha256'],'inventory_sha256':selection['inventory_sha256'],'evidence_bundle_sha256':evidence_sha,'evidence':bindings,'counts':selection['selected_question_counts'],'original_development':selection['original_development'],'source_gold':'mechanically_verified','known_exposure_coverage':'reconciled_saved_training_and_inspected_dev','pool_isolation':'verified_including_decoded_images','namespace_enforced':False,'network_isolation':'not_enforced','root_visible_workflow_boundary':True,'iid_claim':False,'model_evaluation_entrypoint':'pending_real_integration','stage_quality':'not_evaluated','s0_05_complete':False,'kappa':'not_applicable','limitations':['Counts are distinct question identities, not independent family observations; source dependence remains reported.','Root-visible trusted operator workflow is not external security attestation.','Old AOK/video reserve artifacts remain preserved source diagnostics outside primary1000 quotas.','Each milestone still needs matched-source model quality and heldout/development comparison.'],'exit_code':0}
Path('reports/s4-gold-data-admission-001.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'evidence_bundle_sha256':evidence_sha,'s0_05_complete':False}))
