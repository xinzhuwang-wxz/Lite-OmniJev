"""Bind existing TRAIN-only sources for the bounded 200-pair pilot."""
import hashlib
import json
from pathlib import Path

from scripts.gold_media_bundle import private_write

root = Path('/root/autodl-tmp/s1-text-lora-001')
private = root / 'artifacts/s4-gold-inventory-001'
inventory = private / 'crossmodal-dev-002/inventory-manifest.json'
assert hashlib.sha256(inventory.read_bytes()).hexdigest() == '6eb3224906c96ad04042b019f41acfcc533789b8e6b5105c1a1c490a20b53769'
roots = {
    'image': '/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/stage2-sources/aokvqa/coco-ccby-publicdomain-images',
    'audio': '/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207/approved-audio-002',
}
sources = []
for modality, media_root in roots.items():
    matches = [a for a in json.loads(inventory.read_bytes())['samples']
               if a.get('expected_pool') == 'train' and a.get('cohort') == modality]
    assert len(matches) == 1
    a = matches[0]
    sources.append({'path': a['path'], 'sha256': a['sha256'], 'modality': modality, 'media_root': media_root})
output = private / 'crossmodal-train-input-001.json'
assert not output.exists()
private_write(output, json.dumps({'inventory': {'path': str(inventory), 'sha256': hashlib.sha256(inventory.read_bytes()).hexdigest()}, 'sources': sources}, sort_keys=True))
print(json.dumps({'manifest_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'sources': len(sources)}))
