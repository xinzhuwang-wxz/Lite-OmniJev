"""Private inventory metadata inventory; no sample IDs, inputs or answers exported."""
import json,hashlib
from collections import Counter
from pathlib import Path
from scripts.gold_inventory import Manifest,bound_lines
from data.schema import Sample
root=Path('/root/autodl-tmp/s1-text-lora-001');p=root/'artifacts/s4-gold-inventory-001/crossmodal-dev-002/inventory-manifest.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='6eb3224906c96ad04042b019f41acfcc533789b8e6b5105c1a1c490a20b53769'
manifest=Manifest.model_validate_json(p.read_bytes());items=[]
for a in manifest.samples:
 if a.expected_pool=='train' or not a.cohort:continue
 if a.expected_pool!='dev' and a.exposure!='current':continue
 samples=[Sample.model_validate_json(s) for s in bound_lines(a)]
 items.append({'artifact_sha256':a.sha256,'artifact_path':a.path,'cohort':a.cohort,'pool':a.expected_pool,'exposure':a.exposure,'samples':len(samples),'questions':sum(len(s.questions) for s in samples),'families':len({s.meta.family_id for s in samples}),'media':len({m.sha256 for s in samples for m in s.prefix.media}),'datasets':dict(Counter(s.meta.source_dataset for s in samples)),'target_sources':dict(Counter(q.target.source if q.target else 'missing' for s in samples for q in s.questions)),'licenses':dict(Counter(s.meta.license for s in samples))})
report={'scope':'artifact-level metadata only, selection not yet approved','inventory_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'artifacts':items,'exit_code':0}
with (root/'reports/s4-gold-selection-metadata-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({'artifact_count':len(items),'exit_code':0}))
