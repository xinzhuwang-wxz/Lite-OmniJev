"""Bind private source paths without emitting payloads; execute on the rental host."""
import hashlib
import json
import os
from pathlib import Path

root = Path('/root/autodl-tmp/s1-text-lora-001/artifacts/s4-gold-inventory-001')
source = Path('/root/autodl-tmp/lite-omnijev/source-audit/clevrer-official')
inputs = {
    'questions': source / 'files/questions/validation.json',
    'videos': source / 'files/videos/validation/video_validation.zip',
    'readme': source / 'files/README.txt',
    'inventory': root / 'with-development-clean-reserves-manifest.json',
    'quarantine': root / 'quarantined-reserves/exclusions.json',
}
manifest = {'schema_version': 'clevrer-gold-build-v1'}
for name, path in inputs.items():
    with path.open('rb') as handle:
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest[name] = {'path': str(path), 'sha256': sha}
path = root / 'clevrer-build-manifest-001.json'
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as handle:
    json.dump(manifest, handle, sort_keys=True)
print(json.dumps({'outcome': 'prepared', 'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
