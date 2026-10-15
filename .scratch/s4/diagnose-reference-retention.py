"""Aggregate same-input public-dev changes; no new model runs or training data."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from scripts.s0_development_baselines import answer_tokens_for_record
root=Path('/root/autodl-tmp/s1-text-lora-001')
paths={
 'retained':root/'artifacts/s4-pilot-crossmodal-original-001/records.jsonl',
 'video_ce':root/'artifacts/s4-video-pilot-crossmodal-original-001/records.jsonl',
 'reference_kl':root/'artifacts/s4-reference-retention-screen-001/original/records.jsonl',
}
records={name:[json.loads(x) for x in path.read_text().splitlines()] for name,path in paths.items()}
indexes={name:{r['id']:r for r in rows} for name,rows in records.items()}
assert all(len(rows)==50 and set(rows)==set(indexes['retained']) for rows in indexes.values())
report={'scope':'50 fixed original public-dev paired diagnosis, not stage acceptance','sources':{name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in paths.items()},'models':{},'paired':{}}
def prediction(row):
 labels=answer_tokens_for_record(row)
 return labels[max(range(len(labels)),key=lambda i:row['conditional_without_u'][labels[i]])]
for name,rows in records.items():
 confusion=Counter();correct=0;yes=[]
 for r in rows:
  labels=answer_tokens_for_record(r);assert labels==('yes','no')
  gold=labels[r['target_index']];pred=prediction(r)
  confusion[gold+'->'+pred]+=1;correct+=pred==gold;yes.append(r['conditional_without_u']['yes'])
 report['models'][name]={'correct':correct,'count':50,'confusion':dict(confusion),'mean_p_yes':math.fsum(yes)/50}
for name in ['video_ce','reference_kl']:
 corrected=regressed=unchanged=0
 for key,a in indexes['retained'].items():
  b=indexes[name][key]
  for field in ['input_sha256','payload_sha256','answer_token_ids','target_index']:assert a[field]==b[field],field
  gold=answer_tokens_for_record(a)[a['target_index']]
  x=prediction(a)==gold;y=prediction(b)==gold
  corrected+=not x and y;regressed+=x and not y;unchanged+=prediction(a)==prediction(b)
 report['paired'][name]={'corrected':corrected,'regressed':regressed,'unchanged_predictions':unchanged}
report['kl_vs_ce_changed_predictions']=sum(prediction(indexes['video_ce'][k])!=prediction(indexes['reference_kl'][k]) for k in indexes['retained'])
train=json.loads((root/'reports/s4-reference-retention-training-001.json').read_text())
metrics=train['step_metrics'];report['training']={'mean_gold_ce':math.fsum(r['gold_ce'] for r in metrics)/len(metrics),'mean_reference_kl':math.fsum(r['reference_kl'] for r in metrics)/len(metrics),'enabled_draws':sum(r['reference_enabled_count'] for r in metrics),'weight':.5}
report['interpretation']='This coefficient did not improve aggregate crossmodal retention; bounded experiment stops. Confusion and paired corrections identify the next hypothesis, not evidence for changing acceptance thresholds.'
with (root/'reports/s4-reference-retention-diagnostic-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
