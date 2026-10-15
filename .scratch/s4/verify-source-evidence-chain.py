"""Read-only source receipt/input/output hash-chain verification; aggregate export."""
import hashlib,json,time
from pathlib import Path
root=Path('/root/autodl-tmp/s1-text-lora-001');start=time.monotonic();checked={};jobs=[]
def verify(a):
 p=Path(a['path']);assert p.is_absolute();key=str(p)
 if key not in checked:
  with p.open('rb') as f:checked[key]=hashlib.file_digest(f,'sha256').hexdigest()
 assert checked[key]==a['sha256']
 return p
def walk(v):
 if isinstance(v,dict):
  if isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str):verify(v)
  for value in v.values():walk(value)
 elif isinstance(v,list):
  for value in v:walk(value)
p=Path('/root/autodl-tmp/source-evidence-bindings-001.json');assert hashlib.sha256(p.read_bytes()).hexdigest()=='a2937ed6d0a2bfd5c4618ba47cc951f9315392a79ce02ed5ff16bd35e7ab5278'
closure=Path('/root/autodl-tmp/source-code-closure-001.json');assert hashlib.sha256(closure.read_bytes()).hexdigest()=='53e44915c0a0a1b4541eaae602d20e7a44679e87f5d615a1fcb8ba92e6c1a8c6'
report={'scope':'mechanical successful-build source/code/output bindings, not full stage acceptance','exit_code':1,'outcome':'rejected','full_gold_gate':'not_validated'}
try:
 for c in json.loads(closure.read_bytes())['jobs']:
  for f in c['files'].values():verify(f)
 for spec in json.loads(p.read_bytes())['sources']:
  r=json.loads(verify(spec['receipt']).read_bytes());assert r['exit_code']==0
  if spec['source'] in ('gui','clotho','speech','video'):assert r['mode']=='real' and r['outcome']=='built' and r['real_source_verified'] is True
  elif spec['source']=='vqa':assert r['status']=='materialized' and r['synthetic_canary'] is False
  else:assert r['status']=='built' and r['result']['synthetic_canary'] is False
  m=json.loads(verify(spec['input_manifest']).read_bytes());walk(m)
  for a in spec['outputs']:verify(a)
  jobs.append({'source':spec['source'],'receipt_sha256':spec['receipt']['sha256'],'input_manifest_sha256':spec['input_manifest']['sha256'],'output_artifacts':len(spec['outputs']),'status':'verified'})
 report.update(exit_code=0,outcome='mechanical_source_bindings_verified')
except Exception:report['failure']='source_chain_rejected'
report.update(sources=jobs,verified_unique_artifacts=len(checked),binding_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),code_closure_manifest_sha256=hashlib.sha256(closure.read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=round(time.monotonic()-start,3))
with (root/'reports/s4-source-evidence-chain-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(report['exit_code'])
