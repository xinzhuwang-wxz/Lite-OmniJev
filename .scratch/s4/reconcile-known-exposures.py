"""Reconcile known saved training/dev evidence with the latest full inventory."""
import hashlib,json,time
from pathlib import Path
root=Path('/root/autodl-tmp/s1-text-lora-001');base=Path('/root/autodl-tmp/lite-omnijev');private=root/'artifacts/s4-gold-inventory-001';start=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report={'scope':'known saved update-bearing runs and frozen inspected development versions, not unknowable external jobs','outcome':'rejected','exit_code':1}
try:
 ip=private/'crossmodal-dev-002/inventory-manifest.json';assert sha(ip)=='6eb3224906c96ad04042b019f41acfcc533789b8e6b5105c1a1c490a20b53769'
 inv=json.loads(ip.read_bytes());hashes={a['sha256'] for a in inv['samples']};idxp=private/'historical-training-index.json';assert sha(idxp)=='0f6dff35cf4c14f84ecb88b1abee74a9ac3b95558c43bf54693b9f4eb4562b29';idx=json.loads(idxp.read_bytes());assert not idx['unresolved']
 indexed=set();bindings=0
 for a in idx['files']:
  assert sha(Path(a['path']))==a['sha256'] and a['sha256'] in hashes;indexed.add(a['sha256'])
  for b in a['report_bindings']:
   p=Path(b['path']);assert sha(p)==b['sha256'];r=json.loads(p.read_bytes());assert r['data_sha256']==a['sha256'];bindings+=1
 fresh=[]
 paths=list((root/'reports').glob('*.json'))+list((root/'artifacts').rglob('run.json'))+list((base/'artifacts/training').rglob('run.json'))
 for p in set(paths):
  try:r=json.loads(p.read_bytes())
  except (OSError,ValueError):continue
  if isinstance(r,dict) and isinstance(r.get('steps_completed'),int) and r['steps_completed']>0 and r.get('data_sha256'):
   assert r['data_sha256'] in indexed and r['data_sha256'] in hashes;fresh.append(sha(p))
 exposure_reports=[]
 for name in ('s2-training-exposure-audit-001.json','s3-training-exposure-audit-001.json'):
  p=root/'reports'/name;r=json.loads(p.read_bytes());assert r['status']=='passed'
  for run in r['runs']:
   assert run['data_sha256'] in hashes and sha(Path(run['report']))==run['report_sha256']
  exposure_reports.append(sha(p))
 metadata=json.loads((root/'reports/s4-gold-selection-metadata-001.json').read_bytes())
 olddev=[a for a in metadata['artifacts'] if a['pool']=='dev' and '/artifacts/s4-gold-inventory-001/' not in a['artifact_path']]
 assert len(olddev)==7
 for a in olddev:assert a['artifact_sha256'] in hashes and sha(Path(a['artifact_path']))==a['artifact_sha256']
 report.update(outcome='known_exposures_reconciled',exit_code=0,indexed_input_files=len(idx['files']),bound_training_reports=bindings,fresh_update_bearing_reports=len(fresh),frozen_original_dev_versions=len(olddev),exposure_report_sha256=exposure_reports,inventory_sha256=sha(ip),index_sha256=sha(idxp))
except Exception:report['failure']='known_exposure_mismatch'
report.update(elapsed_seconds=round(time.monotonic()-start,3),runner_sha256=sha(Path(__file__)))
with (root/'reports/s4-known-exposure-reconciliation-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(report['exit_code'])
