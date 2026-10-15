import json,os
from pathlib import Path
root=Path('/root/autodl-tmp/s1-text-lora-001/artifacts/s4-gold-inventory-001')
config=json.loads((root/'visual-capacity-config-001.json').read_bytes())
inventory=root/'with-audio-gold-manifest-001.json'
current=json.loads(inventory.read_bytes())
reserves=[{'path':a['path'],'sha256':a['sha256']} for a in current['samples'] if a.get('cohort')=='screenshot' and a['exposure']=='current' and a['expected_pool'] in ('calibration','holdout')]
assert len(reserves)==2
manifest={'schema_version':'gui-gold-build-v1','plan':config['plan'],'source_index':{'path':str(root/'gui-source-index-001.json'),'sha256':'35398d33d0121f4f6cc13cbb5c492cca5d263b27901d69303c198e9ebed8d658'},'inventory':{'path':str(inventory),'sha256':'0d583307d050423ab40c34ba129aefd7a7345355992367e0518f07083e6eba6a'},'quarantine':config['quarantine'],'current_reserves':reserves,'annotation_root':config['annotation_root'],'media_root':config['media_root']}
manifest['historical_rgb_cache']={'path':str(root/'observed-image-rgb-cache-001.json'),'sha256':'5cb45603b2d955bb8323f1218796e6a6b8d32e9ccd701a20da6a3b55ce347496'}
fd=os.open(root/'gui-build-manifest-002.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as h:json.dump(manifest,h,sort_keys=True)
print(json.dumps({'status':'prepared','current_reserve_artifacts':len(reserves)}))
