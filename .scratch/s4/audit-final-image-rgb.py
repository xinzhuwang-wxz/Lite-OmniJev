"""Audit decoded aliases of actual supplied historical images, not unused source frames."""
import concurrent.futures,hashlib,json,os,time
from pathlib import Path
from PIL import Image,__version__ as pillow_version
from scripts import gold_inventory as gi
root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001';start=time.monotonic()
report={'scope':'actual supplied current/historical image samples plus decoded RGB aliases','exit_code':1,'outcome':'rejected'}
try:
 index_path=private/'image-history-resolved-004.json';payload=index_path.read_bytes()
 resolution=json.loads((root/'reports/s4-image-history-resolution-004.json').read_bytes())
 assert resolution['exit_code']==0 and resolution['unresolved']==0
 assert hashlib.sha256(payload).hexdigest()==resolution['resolved_index_sha256']
 index=json.loads(payload);unique={r['sha256']:r for r in index}
 def decode(row):
  p=Path(row['path']);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256']
  with Image.open(p) as im:
   im.load();rgb=im.convert('RGB');raw=rgb.tobytes();width,height=rgb.size
  raw_rgb=hashlib.sha256(raw).hexdigest()
  alias=hashlib.sha256(f'rgb:{width}:{height}:{raw_rgb}'.encode()).hexdigest()
  return row['sha256'],{'sha256':row['sha256'],'width':width,'height':height,'decoded_rgb_sha256':raw_rgb,'isolation_alias_sha256':alias}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:decoded=dict(pool.map(decode,unique.values()))
 cache=private/'observed-image-rgb-cache-002.json'
 fd=os.open(cache,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as h:json.dump({'schema_version':'observed-image-rgb-cache-v1','pillow_version':pillow_version,'source_index_sha256':hashlib.sha256(payload).hexdigest(),'media':decoded},h,sort_keys=True)
 original=gi.audit_pool_isolation
 def enriched(metadata):
  for row in metadata['rows']:
   row['media_sha256'] += [decoded[sha]['isolation_alias_sha256'] for sha in list(row['media_sha256']) if sha in decoded]
  return original(metadata)
 gi.audit_pool_isolation=enriched
 manifest=private/'crossmodal-dev-002/inventory-manifest.json'
 assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='6eb3224906c96ad04042b019f41acfcc533789b8e6b5105c1a1c490a20b53769'
 receipt=gi.audit_manifest(manifest)
 report.update(outcome=receipt.outcome,exit_code=int(receipt.outcome!='isolated'),unique_images_decoded=len(decoded),receipt=receipt.model_dump(mode='json'),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),pillow_version=pillow_version)
except Exception as exc:report['failure_category']=type(exc).__name__
report['elapsed_seconds']=round(time.monotonic()-start,3)
report['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with (root/'reports/s4-observed-image-rgb-audit-002.json').open('x') as h:json.dump(report,h,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!='receipt'}))
if 'receipt' in report:print(json.dumps(report['receipt']['isolation']))
raise SystemExit(report['exit_code'])
