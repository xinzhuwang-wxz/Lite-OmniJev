"""Construct private SHA-bound build manifests; do not print their payloads."""
import json
import os
from pathlib import Path
from scripts.clotho_gold_pools import PINS
from scripts.gold_inventory import Manifest
root=Path('/root/autodl-tmp/s1-text-lora-001/artifacts/s4-gold-inventory-001')
source=Path('/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207')
inventory=root/'with-clevrer-gold-manifest-001.json'
m=Manifest.model_validate_json(inventory.read_bytes())
reserves=[{'path':a.path,'sha256':a.sha256} for a in m.samples if a.exposure=='current' and a.cohort=='audio' and a.expected_pool in ('calibration','holdout')]
assert len(reserves)==2
payload={'schema_version':'clotho-gold-build-v1','audio_root':str(source/'approved-audio-002'),'current_reserves':reserves,'inventory':{'path':str(inventory),'sha256':'3bff87d13b41244636dacccf61114db363f26544bc759f9884c920c632482ea7'},'quarantine':{'path':str(root/'quarantined-reserves/exclusions.json'),'sha256':'8901483c170a7b1b1d04cf339eb4fbc6fc9e9722b8c2beb9cde0dc56b437ba7f'}}
for name,sha in PINS.items():
    path={'metadata':source/'clotho_aqa_metadata.csv','license_plan':Path('/root/autodl-tmp/s1-text-lora-001/reports/clotho-aqa-license-plan.json'),'archive':source/'audio_files.zip'}.get(name,source/f'clotho_aqa_{name}.csv')
    payload[name]={'path':str(path),'sha256':sha}
fd=os.open(root/'clotho-build-manifest-001.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as h: json.dump(payload,h,sort_keys=True)
print(json.dumps({'status':'prepared','current_reserve_artifacts':len(reserves)}))
