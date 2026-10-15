import hashlib,json
from pathlib import Path
from scripts.s0_development_baselines import score_records,answer_tokens_for_record
root=Path('/root/autodl-tmp/s1-text-lora-001');reports={};rows={}
for kind in ('candidate','base'):
 reportpath=root/f'reports/s4-crossmodal-matched-{kind}-001.json';r=json.loads(reportpath.read_bytes());assert r['exit_code']==0 and r['status']=='completed'
 path=root/f'artifacts/s4-crossmodal-matched-{kind}-001/records.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==r['records_sha256']
 rs=[json.loads(x) for x in path.read_text().splitlines()];assert len(rs)==50 and score_records(rs)==r['metrics'];reports[kind]=r;rows[kind]=rs
assert reports['candidate']['selected_ids_sha256']==reports['base']['selected_ids_sha256']
counts={'candidate_only_correct':0,'base_only_correct':0,'both_correct':0,'both_wrong':0}
for a,b in zip(rows['candidate'],rows['base'],strict=True):
 for key in ('id','input_sha256','input_summary_sha256','target_index'):assert key in a and a[key]==b[key]
 ca=a['prediction']==answer_tokens_for_record(a)[a['target_index']];cb=b['prediction']==answer_tokens_for_record(b)[b['target_index']]
 counts['both_correct' if ca and cb else 'candidate_only_correct' if ca else 'base_only_correct' if cb else 'both_wrong']+=1
out={'scope':'fixed50 supplemental public-development matched-input diagnostic only','samples':50,'identical_prepared_inputs':True,'metrics':{k:r['metrics'] for k,r in reports.items()},'paired_counts':counts,'total_wall_seconds':sum(r['elapsed_seconds'] for r in reports.values()),'records_sha256':{k:r['records_sha256'] for k,r in reports.items()},'stage_gate_pass':False,'exit_code':0}
with (root/'reports/s4-crossmodal-matched-paired-001.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
