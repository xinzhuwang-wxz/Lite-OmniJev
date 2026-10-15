"""Verify only media URIs changed across every consumed normalization mapping."""
import json,hashlib,time
from pathlib import Path
from gold_media_bundle_97da7e9 import media_path
root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001';start=time.monotonic();checked={};totals=[];report={'outcome':'rejected','exit_code':1}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_media(p,expected):
 stat=p.stat();key=(str(p),stat.st_size,stat.st_mtime_ns)
 if key not in checked:checked[key]=sha(p)
 assert checked[key]==expected
try:
 cases=[('audio-media-bundle-001','audio-media-bundle-input-001.json','s4-audio-media-bundle-001.json'),('crossmodal-source-bundle-001','crossmodal-source-bundle-input-001.json','s4-crossmodal-source-bundle-001.json'),('crossmodal-dev-source-bundle-002','crossmodal-dev-bundle-input-002.json','s4-crossmodal-dev-source-bundle-002.json')]
 for dirname,inputname,reportname in cases:
  out=private/dirname;receipt=json.loads((root/'reports'/reportname).read_bytes());assert receipt['outcome']=='bundled'
  ip=private/inputname;assert sha(ip)==receipt['manifest_sha256'];specs=json.loads(ip.read_bytes())['samples'];mapping=json.loads((out/'mapping.json').read_bytes());assert len(specs)==len(mapping)
  count=0
  for spec,pair in zip(specs,mapping,strict=True):
   assert pair['original']=={'path':spec['path'],'sha256':spec['sha256']}
   old=Path(spec['path']);new=Path(pair['rewritten']['path']);assert sha(old)==spec['sha256'] and sha(new)==pair['rewritten']['sha256']
   a=old.read_bytes().splitlines();b=new.read_bytes().splitlines();assert len(a)==len(b)
   roots=[Path(p) for p in spec.get('media_roots',[spec.get('media_root')])]
   for l,r in zip(a,b,strict=True):
    left=json.loads(l);right=json.loads(r)
    lm=left['prefix'].get('media',[]);rm=right['prefix'].get('media',[]);assert len(lm)==len(rm)
    for x,y in zip(lm,rm,strict=True):
     assert x['sha256']==y['sha256'];verify_media(media_path(roots,x['uri']),x['sha256']);verify_media(media_path([out],y['uri']),y['sha256']);y['uri']=x['uri']
    assert left==right;count+=1
  totals.append({'bundle':dirname,'samples':count,'mapping_sha256':sha(out/'mapping.json'),'input_manifest_sha256':sha(ip),'receipt_sha256':sha(root/'reports'/reportname)})
 report.update(outcome='uri_only_changes_verified',exit_code=0,bundles=totals,verified_media_paths=len(checked))
except Exception:report['failure']='normalization_mismatch'
report.update(scope='exact semantic preservation plus original/rewritten media bytes; not full gold admission',elapsed_seconds=round(time.monotonic()-start,3),runner_sha256=sha(Path(__file__)))
with (root/'reports/s4-media-normalization-attestation-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(report['exit_code'])
