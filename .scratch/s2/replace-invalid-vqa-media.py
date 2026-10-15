"""Replace decode-rejected families deterministically before training admission."""
import json
from pathlib import Path
from scripts.vqa2_new_family import plan
root=Path('/root/autodl-tmp/s1-text-lora-001');old=root/'artifacts/s2-new-family-plan-003';new=root/'artifacts/s2-new-family-plan-004'
d=json.loads((old/'plan.json').read_text())['input_paths']
r=plan(**{k:Path(v) for k,v in d.items() if not k.startswith('ledger_')},allocation_ledger=[Path(v) for k,v in d.items() if k.startswith('ledger_')]+[root/'artifacts/s2-new-family-bad-media-exclusion-001.jsonl'],output_dir=new)
existing={x['image_id'] for x in map(json.loads,(old/'candidates.jsonl').read_text().splitlines())}
rows=[x for x in map(json.loads,(new/'candidates.jsonl').read_text().splitlines()) if x['image_id'] not in existing]
assert len(rows)==1
subset=root/'artifacts/s2-new-family-replacement-001';subset.mkdir(exist_ok=False)
(subset/'candidates.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
print({'sample_count':r['sample_count'],'replacements':len(rows),'candidates_sha256':r['candidates_sha256']})
