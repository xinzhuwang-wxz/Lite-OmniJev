"""Combine SHA-bound private inventories and emit only isolation aggregates."""
import hashlib
import json
import os
from pathlib import Path
from scripts.gold_inventory import main

root = Path('/root/autodl-tmp/s1-text-lora-001/artifacts/s4-gold-inventory-001')
build = json.loads((root / 'clevrer-build-manifest-001.json').read_bytes())
base_path = Path(build['inventory']['path'])
assert hashlib.sha256(base_path.read_bytes()).hexdigest() == build['inventory']['sha256']
base = json.loads(base_path.read_bytes())
new_path = root / 'clevrer-validation-gold-001/inventory-manifest.json'
receipt = json.loads(Path('/root/autodl-tmp/s1-text-lora-001/reports/s4-clevrer-gold-build-001.json').read_bytes())
assert receipt['outcome'] == 'built' and receipt['real_source_verified']
assert hashlib.sha256(new_path.read_bytes()).hexdigest() == receipt['output_sha256']['inventory-manifest.json']
new = json.loads(new_path.read_bytes())
base['samples'].extend(new['samples'])
base['component_ledgers'].extend(new['component_ledgers'])
combined = root / 'with-clevrer-gold-manifest-001.json'
fd = os.open(combined, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as handle:
    json.dump(base, handle, sort_keys=True)
raise SystemExit(main(['--manifest', str(combined), '--report', '/root/autodl-tmp/s1-text-lora-001/reports/s4-clevrer-gold-isolation-001.json']))
