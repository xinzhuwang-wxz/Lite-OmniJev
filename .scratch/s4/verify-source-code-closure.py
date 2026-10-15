import json,hashlib,time
from pathlib import Path
start=time.monotonic();p=Path('/root/autodl-tmp/source-code-closure-001.json');assert hashlib.sha256(p.read_bytes()).hexdigest()=='53e44915c0a0a1b4541eaae602d20e7a44679e87f5d615a1fcb8ba92e6c1a8c6'
config=json.loads(p.read_bytes());jobs=[]
for job in config['jobs']:
 failed=0
 for a in job['files'].values():
  try:actual=hashlib.sha256(Path(a['path']).read_bytes()).hexdigest()
  except OSError:actual=None
  failed+=int(actual!=a['sha256'])
 jobs.append({'source':job['source'],'code_files':len(job['files']),'mismatched_or_missing':failed})
r={'scope':config['scope'],'jobs':jobs,'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'exit_code':int(any(j['mismatched_or_missing'] for j in jobs)),'elapsed_seconds':round(time.monotonic()-start,3)}
with Path('/root/autodl-tmp/s1-text-lora-001/reports/s4-source-code-closure-001.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r));raise SystemExit(r['exit_code'])
