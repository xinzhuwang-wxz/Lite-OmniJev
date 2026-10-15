"""Keep original exposure and bind relocated current artifacts; aggregate-only."""
import hashlib,json,os
from pathlib import Path
from scripts.gold_inventory import audit_manifest
root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001'
def bound(path,sha):
 payload=path.read_bytes();assert hashlib.sha256(payload).hexdigest()==sha;return json.loads(payload)
inventory=bound(private/'gui-gold-002/inventory-manifest.json','a80bf03ff68508cf2c32c7108991e6d649191de207e2db3f26b1410e749611af')
mapping=bound(private/'audio-media-bundle-001/mapping.json','ffd6bcaa276585553846a012c3d0bb11b7970ae0fac6559f82b85f560634f181')
relocated=bound(private/'audio-media-bundle-001/inventory-manifest.json','4f3318d84df49d556f1f4193f9e4d42b30385720f684d1eb179b410c6b1cea1a')
assert len(mapping)==len(relocated['samples'])==4
for oldnew in mapping:
 matches=[a for a in inventory['samples'] if a['path']==oldnew['original']['path'] and a['sha256']==oldnew['original']['sha256'] and a['exposure']=='current']
 assert len(matches)==1
 original=matches[0];original['exposure']='historical'
 replacements=[a for a in relocated['samples'] if a['path']==oldnew['rewritten']['path'] and a['sha256']==oldnew['rewritten']['sha256']]
 assert len(replacements)==1
 replacement=replacements[0]
 assert replacement['expected_pool']==original['expected_pool'] and replacement['cohort']==original['cohort']
 inventory['samples'].append(replacement)
path=private/'with-audio-relocated-inventory-001.json'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:json.dump(inventory,f,sort_keys=True)
receipt=audit_manifest(path)
with (root/'reports/s4-audio-relocated-isolation-001.json').open('x') as f:f.write(receipt.model_dump_json())
print(json.dumps({'outcome':receipt.outcome,'manifest_sha256':receipt.manifest_sha256,'isolation':receipt.isolation.model_dump() if receipt.isolation else None}))
raise SystemExit(0 if receipt.outcome=='isolated' else 1)
