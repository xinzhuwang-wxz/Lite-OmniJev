import json,os
from pathlib import Path
root=Path('/root/autodl-tmp/s1-text-lora-001/artifacts/s4-gold-inventory-001')
source=Path('/root/autodl-tmp/lite-omnijev/source-audit/librispeech-s4-candidate-001/mirror-002')
receipt=json.loads(Path('/root/autodl-tmp/s1-text-lora-001/reports/s4-clotho-gold-build-001.json').read_bytes())
assert receipt['outcome']=='built' and receipt['real_source_verified']
manifest={'schema_version':'librispeech-gold-build-v1','dev_clean':{'path':str(source/'dev-clean.tar.gz'),'sha256':'76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3'},'test_clean':{'path':str(source/'test-clean.tar.gz'),'sha256':'39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23'},'inventory':{'path':str(root/'clotho-gold-001/inventory-manifest.json'),'sha256':receipt['output_sha256']['inventory-manifest.json']},'quarantine':{'path':str(root/'quarantined-reserves/exclusions.json'),'sha256':'8901483c170a7b1b1d04cf339eb4fbc6fc9e9722b8c2beb9cde0dc56b437ba7f'},'alias_ledgers':[]}
fd=os.open(root/'speech-build-manifest-001.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as h: json.dump(manifest,h,sort_keys=True)
