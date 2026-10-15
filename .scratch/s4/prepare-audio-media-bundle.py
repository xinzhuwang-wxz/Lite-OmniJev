"""Prepare bound metadata only; no sample payload output."""
import hashlib,json,os
from pathlib import Path
private=Path('/root/autodl-tmp/s1-text-lora-001/artifacts/s4-gold-inventory-001')
inventory=private/'gui-gold-002/inventory-manifest.json'
assert hashlib.sha256(inventory.read_bytes()).hexdigest()=='a80bf03ff68508cf2c32c7108991e6d649191de207e2db3f26b1410e749611af'
selected=[]
for artifact in json.loads(inventory.read_bytes())['samples']:
 if artifact.get('cohort')!='audio' or artifact['exposure']!='current' or artifact['expected_pool'] not in ('calibration','holdout'):continue
 parent=Path(artifact['path']).parent
 assert parent in (private/'clotho-gold-001',private/'speech-gold-001')
 roots=[str(parent)]
 if parent.name=='clotho-gold-001':roots.append('/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207/approved-audio-002')
 selected.append({**artifact,'media_roots':roots})
assert len(selected)==4
payload=json.dumps({'schema_version':'gold-media-bundle-manifest-v1','samples':selected},sort_keys=True).encode()
fd=os.open(private/'audio-media-bundle-input-001.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'wb') as f:f.write(payload)
print(json.dumps({'artifacts':len(selected),'manifest_sha256':hashlib.sha256(payload).hexdigest()}))
