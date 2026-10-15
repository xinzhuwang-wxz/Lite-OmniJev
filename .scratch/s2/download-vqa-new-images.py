"""Stage only planned COCO images; never admit samples or replace existing media."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

p=argparse.ArgumentParser();p.add_argument('--plan-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=False)
candidates=a.plan_dir/'candidates.jsonl'
rows=[json.loads(x) for x in candidates.read_text().splitlines()]
assert len(rows)<=600 and len({r['image_id'] for r in rows})==len(rows)
started=time.monotonic()
def download(row):
    url=row['url'];parsed=urlparse(url)
    assert parsed.hostname=='images.cocodataset.org' and parsed.scheme in {'http','https'}
    path=a.output_dir/f"{row['image_id']:012d}.jpg"
    error=None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url,timeout=30) as response:
                raw=response.read(20*1024*1024+1)
            if len(raw)>20*1024*1024: raise ValueError('image exceeds 20MiB staging cap')
            with path.open('xb') as f:f.write(raw)
            return {'image_id':row['image_id'],'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'url':url,'bytes':len(raw)}
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            if path.exists(): break
    return {'image_id':row['image_id'],'url':url,'error':error}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(download,rows))
good=[r for r in results if 'sha256' in r];bad=[r for r in results if 'error' in r]
(a.output_dir/'download-manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in good))
report={'status':'staged' if not bad else 'failed','training_admitted':False,'planned':len(rows),'downloaded':len(good),'bytes':sum(r['bytes'] for r in good),'failed':bad,'seconds':time.monotonic()-started,'candidates_sha256':hashlib.sha256(candidates.read_bytes()).hexdigest()}
(a.output_dir/'download-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
raise SystemExit(bool(bad))
