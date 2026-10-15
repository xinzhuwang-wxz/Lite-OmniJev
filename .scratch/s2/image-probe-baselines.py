import hashlib
import json
from pathlib import Path
root=Path('/root/autodl-tmp/s1-text-lora-001')
subset=root/'artifacts/s2-text-lora-001/dev-image-150.jsonl'
ids={json.loads(line)['id'] for line in subset.read_text().splitlines()}
results={}
for tag,directory in [('v1','dev-eval-image'),('v2','dev-eval-image-v2-corrected-001')]:
 p=root/'artifacts/s2-text-lora-001'/directory/'records.jsonl'
 rows=[r for r in map(json.loads,p.read_text().splitlines()) if r['pass']==0 and r['variant']=='original']
 selected=[r for r in rows if r['id'] in ids]
 assert len(selected)==150
 results[tag]={'full_accuracy':sum(r['prediction']==r['target'] for r in rows)/len(rows),'probe_accuracy':sum(r['prediction']==r['target'] for r in selected)/150,'probe_correct':sum(r['prediction']==r['target'] for r in selected),'probe_samples':150,'sidecar_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
output={'status':'passed','exit_code':0,'scope':'matched dev150 subset baseline for checkpoint probe, no new GPU calls','subset_sha256':hashlib.sha256(subset.read_bytes()).hexdigest(),'results':results}
(root/'reports/s2-image-probe-baselines-001.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
