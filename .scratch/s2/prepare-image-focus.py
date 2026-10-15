"""Freeze a small image-focused pool with old-modality replay, without new data."""
import hashlib
import json
from pathlib import Path

from scripts.train_s1 import verify_data_is_train_pool

root=Path('/root/autodl-tmp/s1-text-lora-001')
out=root/'artifacts/s2-image-focus-001'
out.mkdir(exist_ok=False)
cohorts={'image':('s2-media-pools-image-001',600),'text':('s1-text-pools-v1-001',200),'screenshot':('s2-media-pools-screenshot-001',200)}
rows=[]
inputs={}
ledgers=[]
for cohort,(directory,count) in cohorts.items():
 source=root/'artifacts'/directory/'run-a'
 p=source/'train.jsonl'
 digest=hashlib.sha256(p.read_bytes()).hexdigest()
 assert digest==json.loads((source/'report.json').read_text())['outputs']['train_jsonl_sha256']
 candidates=list(map(json.loads,p.read_text().splitlines()))
 selected=sorted(candidates,key=lambda r:hashlib.sha256(('s2-focus:42:'+r['id']).encode()).hexdigest())[:count]
 rows.extend(selected)
 ledgers.append(source/'allocation-ledger.jsonl')
 inputs[cohort]={'sha256':digest,'selected':count}
 if cohort=='image':
  (out/'train-image-150.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in selected[:150]))
data=out/'train.jsonl'
data.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
checks=verify_data_is_train_pool(data,ledgers)
config=json.loads((root/'configs/s2-joint-v2.json').read_text())
config.update(sample_count=1000,steps=100,accumulation=10,learning_rate=2e-5,midway_step=50,checkpoint_every=50)
(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
report={'status':'passed','exit_code':0,'inputs':inputs,'train_rows':1000,'ledger_check':checks,'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((out/'config.json').read_bytes()).hexdigest(),'holdout_content_accessed':False,'scope':'experiment preparation, not training or stage acceptance'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
