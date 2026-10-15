"""Compare private records and export only aggregate repeatability evidence."""
import json,hashlib,stat,time
from pathlib import Path
root=Path('/root/autodl-tmp/s1-text-lora-001');runs=root/'artifacts/s4-private-stage-eval-001';start=time.monotonic()
report={'scope':'two dev-only real model executions through fixed private entrypoint; not stage quality','exit_code':1,'outcome':'rejected'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
 receipts=[json.loads((root/f'reports/s4-private-stage-engineering-00{i}.json').read_bytes()) for i in (1,2)]
 for r in receipts:assert r['mode']=='engineering' and r['execution_status']=='complete' and r['exit_code']==0 and r['records']==50
 keys=('spec_sha256','data_sha256','code_sha256','base_revision','model_manifest_sha256','adapter_manifest_sha256','tokenizer_template_sha256','answer_map_sha256','config_sha256','canonicalized_input_sha256','seed','source_rows','records','metrics')
 assert all(receipts[0][k]==receipts[1][k] for k in keys)
 datasets=[];permissions=[]
 for r in receipts:
  matches=[]
  for d in runs.iterdir():
   if not d.is_dir() or not (d/'receipt.json').is_file():continue
   local=json.loads((d/'receipt.json').read_bytes())
   if local['records_sha256']==r['records_sha256']:matches.append(d)
  assert len(matches)==1;d=matches[0];p=d/'records.jsonl';assert sha(p)==r['records_sha256']
  datasets.append([json.loads(line) for line in p.read_bytes().splitlines()])
  permissions.append(stat.S_IMODE(d.stat().st_mode)==0o700 and all(stat.S_IMODE(f.stat().st_mode)==0o600 for f in d.iterdir() if f.is_file()))
 assert all(permissions) and all(len(rs)==50 for rs in datasets)
 fields=('id','input_sha256','input_summary_sha256','raw_logprobs','conditional_without_u','p_U','prediction','target_index','answer_token_ids')
 mismatches=sum(any(a.get(k)!=b.get(k) for k in fields) for a,b in zip(*datasets,strict=True))
 assert mismatches==0
 report.update(outcome='repeatable',exit_code=0,records_compared=50,record_mismatches=mismatches,private_permissions_verified=True,binding_fields_compared=list(keys),records_sha256=[r['records_sha256'] for r in receipts],metrics=receipts[0]['metrics'],total_job_wall_seconds=sum(r['wall_seconds'] for r in receipts),code_sha256=receipts[0]['code_sha256'],spec_sha256=receipts[0]['spec_sha256'])
except Exception:report['failure']='repeatability_or_permission_rejected'
report.update(runner_sha256=sha(Path(__file__)),elapsed_seconds=round(time.monotonic()-start,3))
with (root/'reports/s4-private-stage-repeatability-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(report['exit_code'])
