"""Hash the pinned-revision cache privately; export only aggregate binding."""
import hashlib,json,os,time
from pathlib import Path
start=time.monotonic()
root=Path('/root/autodl-tmp/s1-text-lora-001')
private=root/'artifacts/s4-gold-inventory-001'
config=json.loads((private/'visual-capacity-config-001.json').read_bytes())
planpath=Path(config['plan']['path'])
assert hashlib.sha256(planpath.read_bytes()).hexdigest()==config['plan']['sha256']
plan=json.loads(planpath.read_bytes())
index={'schema_version':'gui-gold-source-index-v1','revision':'61632d0f3f4d51d7e9561ce4f84347dd06b2019d','license':'CC-BY-4.0','annotations':{},'media':{}}
for entry in plan['selected_episodes']:
 eid=entry['episode_id']; assert eid.isdigit() and len(eid)==16
 annotation=Path(config['annotation_root'])/(eid+'.json')
 assert not annotation.is_symlink()
 payload=annotation.read_bytes(); index['annotations'][eid]=hashlib.sha256(payload).hexdigest()
 data=json.loads(payload); assert str(data['episode_id'])==eid
 for step in data['steps']:
  name=step['screenshot']; assert name==f"{eid}_{step['step']}.png"
  path=Path(config['media_root'])/'screenshots'/name
  assert not path.is_symlink()
  with path.open('rb') as h: sha=hashlib.file_digest(h,'sha256').hexdigest()
  if name in index['media']: assert index['media'][name]==sha
  index['media'][name]=sha
out=private/'gui-source-index-001.json'
fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as h:json.dump(index,h,sort_keys=True)
report={'scope':'hash binding of previously staged pinned-revision cache; not independent remote re-download attestation','revision':index['revision'],'plan_sha256':config['plan']['sha256'],'annotations':len(index['annotations']),'media':len(index['media']),'index_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':round(time.monotonic()-start,3),'exit_code':0}
with (root/'reports/s4-gui-source-index-001.json').open('x') as h:json.dump(report,h,indent=2)
print(json.dumps(report))
