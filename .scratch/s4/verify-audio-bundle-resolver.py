"""Verify actual rewritten references through the unchanged training resolver."""
import hashlib,json,time
from pathlib import Path
from data.schema import Sample
from train.multimodal import resolve_media
start=time.monotonic();root=Path('/root/autodl-tmp/s1-text-lora-001');bundle=root/'artifacts/s4-gold-inventory-001/audio-media-bundle-001'
p=bundle/'inventory-manifest.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='4f3318d84df49d556f1f4193f9e4d42b30385720f684d1eb179b410c6b1cea1a'
report={'outcome':'rejected','exit_code':1,'verified_references':0,'full_gold_gate':'not_validated'}
try:
 for a in json.loads(p.read_bytes())['samples']:
  payload=Path(a['path']).read_bytes();assert hashlib.sha256(payload).hexdigest()==a['sha256']
  for line in payload.splitlines():
   s=Sample.model_validate_json(line)
   for m in s.prefix.media:
    resolve_media(bundle,m.uri,m.sha256,max_bytes=50*1024*1024);report['verified_references']+=1
 assert report['verified_references']==4000
 report.update(outcome='verified',exit_code=0)
except Exception:report['failure']='resolver_rejected'
report.update(manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),elapsed_seconds=round(time.monotonic()-start,3),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (root/'reports/s4-audio-bundle-resolver-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(report['exit_code'])
