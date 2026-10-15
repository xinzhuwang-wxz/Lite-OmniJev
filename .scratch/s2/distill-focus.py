"""Bounded gold-first teacher supervision on an already frozen training slice."""
import hashlib
import json
import time
from pathlib import Path
from data.schema import Sample
from scripts.s0_development_baselines import default_processor_factory, real_forward_factory, prepare_sample, make_record
from scripts.train_s1 import verify_data_is_train_pool
root=Path('/root/autodl-tmp/s1-text-lora-001')
source=root/'artifacts/s2-image-focus-001'
output=root/'artifacts/s2-image-distill-001'
output.mkdir(exist_ok=False)
start=time.time()
report_path=root/'reports/s2-image-distill-labels-001.json'
report={'status':'failed','exit_code':1,'holdout_content_accessed':False,'cost':None}
cleanup=None
try:
 teacher_evidence=root/'reports/s0-development-image-onejev-4b-001.json'
 evidence=json.loads(teacher_evidence.read_text())['model']
 assert evidence['repo_id']=='OmniJev/OneJev-4B' and evidence['license']=='apache-2.0'
 assert evidence['revision']=='c88e18653ceb7a8770716287f55fdefc79d6b588'
 model=Path(evidence['model_dir'])
 for name,expected in evidence['metadata_hashes'].items():
  assert hashlib.sha256((model/name).read_bytes()).hexdigest()==expected,name
 rows=[json.loads(line) for line in (source/'train.jsonl').read_text().splitlines()]
 ledgers=[root/'artifacts'/d/'run-a/allocation-ledger.jsonl' for d in ['s1-text-pools-v1-001','s2-media-pools-image-001','s2-media-pools-screenshot-001']]
 verify_data_is_train_pool(source/'train.jsonl',ledgers)
 processor=default_processor_factory(model)
 forward,cleanup=real_forward_factory(model,'onejev-4b','bfloat16')
 agreed=total=0
 sidecar=output/'teacher-records.jsonl'
 old='aokvqa-train-image-choice-s2-pools-v1';new='aokvqa-train-image-choice-s2-teacher-soft-v1'
 with sidecar.open('x') as stream:
  for row in rows:
   if row['meta']['data_version']!=old: continue
   sample=Sample.model_validate(row); assert sample.meta.pool=='train'
   prepared=prepare_sample(sample,processor,Path('/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/stage2-sources/aokvqa/coco-ccby-publicdomain-images'))
   record=make_record(prepared,pass_index=1,forward=forward)
   stream.write(json.dumps(record,sort_keys=True)+'\n');stream.flush()
   q=row['questions'][0];gold=q['target']['value']
   if record['prediction']==gold:
    p=record['conditional_without_u'];tokens=[chr(ord('A')+i) for i in range(len(q['options']))]
    values=[.5*p[t]+(.5 if t==gold else 0.) for t in tokens]
    q['target']={'kind':'soft','probs':values,'source':'gold_plus_admitted_OneJev-4B_0.5;parent_gold_preserved'}
    agreed+=1
   row['meta']['data_version']=new
   Sample.model_validate(row)
   total+=1
   if total==20 or total%100==0: print(json.dumps({'annotated':total,'teacher_gold_agreement':agreed/total}),flush=True)
 assert total==600
 report.update(teacher_model=evidence,teacher_evidence_sha256=hashlib.sha256(teacher_evidence.read_bytes()).hexdigest(),metadata_rehashed=True,weights_hashes_reused_from_s0=True,teacher_gold_agreement=agreed/total,annotated=total,soft_rows=agreed,hard_gold_fallback_rows=total-agreed,teacher_records_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest())
 assert agreed/total>=.7,'teacher agreement below pilot threshold; do not train'
 data=output/'train.jsonl';data.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
 verify_data_is_train_pool(data,ledgers)
 config=json.loads((source/'config.json').read_text());config['expected_data_versions']=[new if v==old else v for v in config['expected_data_versions']]
 (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 report.update(status='passed',exit_code=0,data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),source_sha256=hashlib.sha256((source/'train.jsonl').read_bytes()).hexdigest(),config_sha256=hashlib.sha256((output/'config.json').read_bytes()).hexdigest())
except Exception as exc:
 report['error']=f'{type(exc).__name__}: {exc}'
finally:
 if cleanup: cleanup()
 report['duration_seconds']=time.time()-start
 report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='teacher_model'}),flush=True)
raise SystemExit(report['exit_code'])
