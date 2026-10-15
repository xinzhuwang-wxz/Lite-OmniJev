"""Resolve bound historical images inside a private job; export only coverage."""
import hashlib,json,os,time
from pathlib import Path
from urllib.parse import urlparse,unquote
from scripts.gold_inventory import Manifest,bound_lines
from data.schema import Sample
start=time.monotonic();root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001'
config=json.loads((private/'vqa-capacity-config-001.json').read_bytes())
config['inventory']={'path':str(private/'crossmodal-dev-002/inventory-manifest.json'),'sha256':'6eb3224906c96ad04042b019f41acfcc533789b8e6b5105c1a1c490a20b53769'}
manifest_path=Path(config['inventory']['path'])
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==config['inventory']['sha256']
manifest=Manifest.model_validate_json(manifest_path.read_bytes())
roots=set(config['media_roots'])
roots.update([str(private),'/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/gui-odyssey-source-audit/media/app_split_train_under19_5gb'])
roots.add('/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine')
roots.update(['/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/vqa2-new-images-001','/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/vqa2-new-images-replacement-001'])
configpaths=[root/'.scratch/s3/main-config.json',root/'.scratch/s3/probe-config.json']
for path in configpaths:
 if path.exists(): roots.update(json.loads(path.read_bytes()).get('media_roots',[]))
for path in (root/'reports').glob('s*-dev-eval-*.json'):
 try:
  report=json.loads(path.read_bytes());cmd=report.get('command',[])
  if '--media-root' in cmd: roots.add(cmd[cmd.index('--media-root')+1])
 except (ValueError,IndexError): pass
bundle=json.loads((root/'reports/s0-development-bundle-001.json').read_bytes())
for value in bundle.values():
 if isinstance(value,list):
  for item in value:
   if isinstance(item,dict) and isinstance(item.get('media_root'),str): roots.add(item['media_root'])
images=set()
for artifact in manifest.samples:
 parent=Path(artifact.path).parent
 for path in (parent,parent/'media',parent.parent/'media'): roots.add(str(path))
 for line in bound_lines(artifact):
  sample=Sample.model_validate_json(line)
  images.update((m.uri,m.sha256) for m in sample.prefix.media if m.type=='image')
roots={str(Path(p)) for p in roots if Path(p).is_dir() and not Path(p).is_symlink()}
cache={};missing=0;resolved=[]
for uri,sha in sorted(images):
 parsed=urlparse(uri); direct=Path(unquote(parsed.path)) if parsed.scheme=='file' else Path(uri)
 paths=[direct] if direct.is_absolute() else [Path(p)/direct for p in sorted(roots)]
 matches=[]
 for path in paths:
  if not path.is_file() or path.is_symlink():continue
  if path not in cache:
   with path.open('rb') as h:cache[path]=hashlib.file_digest(h,'sha256').hexdigest()
  if cache[path]==sha:matches.append(path)
 if not matches:missing+=1
 else:resolved.append({'uri':uri,'sha256':sha,'path':str(matches[0])})
# No source data or memberships are changed. Original paths remain private.
fd=os.open(private/'image-history-resolved-004.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as h:json.dump(resolved,h,sort_keys=True)
config['media_roots']=sorted(roots)
fd=os.open(private/'image-audit-config-004.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as h:json.dump(config,h,sort_keys=True)
report={'scope':'SHA-bound historical image resolution only, RGB decode remains materializer responsibility','image_references':len(images),'resolved':len(resolved),'unresolved':missing,'roots':len(roots),'config_sha256':hashlib.sha256((private/'image-audit-config-004.json').read_bytes()).hexdigest(),'resolved_index_sha256':hashlib.sha256((private/'image-history-resolved-004.json').read_bytes()).hexdigest(),'elapsed_seconds':round(time.monotonic()-start,3),'exit_code':int(missing>0)}
with (root/'reports/s4-image-history-resolution-004.json').open('x') as h:json.dump(report,h,indent=2)
print(json.dumps(report))
raise SystemExit(report['exit_code'])
