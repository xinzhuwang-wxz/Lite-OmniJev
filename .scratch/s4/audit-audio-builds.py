"""Join built private audio inventories; export only counts and isolation receipt."""
import json,os,hashlib
from collections import Counter
from pathlib import Path
from scripts.gold_inventory import audit_manifest, Manifest, bound_lines, extract
from data.schema import Sample
root=Path('/root/autodl-tmp/s1-text-lora-001')
private=root/'artifacts/s4-gold-inventory-001'
base=None; source_receipts={}
for name in ('clotho','speech'):
 receipt_path=root/f'reports/s4-{name}-gold-build-001.json'
 receipt=json.loads(receipt_path.read_bytes())
 assert receipt['outcome']=='built' and receipt['real_source_verified']
 path=private/f'{name}-gold-001/inventory-manifest.json'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['output_sha256']['inventory-manifest.json']
 parsed=json.loads(path.read_bytes())
 if base is None: base=parsed
 else:
  base['samples'].extend(parsed['samples']);base['component_ledgers'].extend(parsed['component_ledgers'])
 source_receipts[name]=hashlib.sha256(receipt_path.read_bytes()).hexdigest()
path=private/'with-audio-gold-manifest-001.json'
if path.exists():
 assert json.loads(path.read_bytes())==base
else:
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as h:json.dump(base,h,sort_keys=True)
receipt=audit_manifest(path)
counts=Counter()
for artifact in Manifest.model_validate(base).samples:
 if artifact.exposure=='current':
  for line in bound_lines(artifact):
   sample=Sample.model_validate_json(line)
   if extract(sample,artifact).cohort=='audio': counts[artifact.expected_pool]+=1
assert counts['calibration']==3000 and counts['holdout']==1000
report={'scope':'source-bound audio builds plus supplied historical metadata isolation','source_receipt_sha256':source_receipts,'inventory_receipt':receipt.model_dump(mode='json'),'audio_current_counts':dict(counts),'speech_dev_is_supplemental':True,'source_alias_ledgers':{'speech':{'path':str(private/'speech-gold-001/allocation.jsonl'),'sha256':hashlib.sha256((private/'speech-gold-001/allocation.jsonl').read_bytes()).hexdigest()},'clotho':{'path':str(private/'clotho-gold-001/allocation.jsonl'),'sha256':hashlib.sha256((private/'clotho-gold-001/allocation.jsonl').read_bytes()).hexdigest()}},'full_gold_gate':'not_validated','exit_code':0 if receipt.outcome=='isolated' else 1}
with (root/'reports/s4-audio-gold-isolation-002.json').open('x') as h:json.dump(report,h,indent=2)
print(json.dumps({'outcome':receipt.outcome,'audio_counts':dict(counts),'manifest_sha256':receipt.manifest_sha256,'exit_code':report['exit_code']}))
raise SystemExit(report['exit_code'])
