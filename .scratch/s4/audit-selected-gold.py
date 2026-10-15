"""Private selected-item counts and hard-target validation, not source attestation."""
import hashlib,json,time
from collections import Counter,defaultdict
from pathlib import Path
from data.schema import Sample,HardTarget,answer_tokens
from scripts.gold_inventory import Manifest,bound_lines,Artifact
root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001';start=time.monotonic()
p=private/'selected-gold-manifest-001.json';payload=p.read_bytes();assert hashlib.sha256(payload).hexdigest()=='e073a601eab235644f91f058a99dd331c568de2560660441192778553429876f'
config=json.loads(payload);ip=Path(config['inventory']['path']);assert hashlib.sha256(ip.read_bytes()).hexdigest()==config['inventory']['sha256']
inventory=Manifest.model_validate_json(ip.read_bytes());registered={(a.path,a.sha256,a.expected_pool) for a in inventory.samples}
summary={'scope':'bound selected-item and target validation only','full_gold_gate':'not_validated','source_authenticity':'requires_source_replay_binding','exit_code':1,'outcome':'rejected'}
try:
 groups=defaultdict(list);original=Counter();counts=[];licenses=Counter()
 for spec in config['selected']:
  assert (spec['path'],spec['sha256'],spec['pool']) in registered
  samples=[Sample.model_validate_json(line) for line in bound_lines(Artifact(path=spec['path'],sha256=spec['sha256']))]
  assert len(samples)==spec['expected_questions']
  seen=set();families=set();media=set();sources=Counter()
  for s in samples:
   assert s.meta.pool==spec['pool'] and s.meta.usage=='commercial' and len(s.questions)==1
   q=s.questions[0];assert isinstance(q.target,HardTarget) and isinstance(q.target.value,str) and q.target.value in answer_tokens(q) and q.target.value!='U'
   assert q.target.source in spec['expected_target_sources']
   identity=(s.id,q.qid);assert identity not in seen;seen.add(identity)
   groups[(spec['cohort'],spec['pool'])].append(identity);families.add(s.meta.family_id);media.update(m.sha256 for m in s.prefix.media);sources[q.target.source]+=1;licenses[s.meta.license]+=1
  if spec['frozen_original_dev']:original[spec['cohort']]+=len(samples)
  counts.append({'cohort':spec['cohort'],'pool':spec['pool'],'artifact_sha256':spec['sha256'],'questions':len(samples),'unique_questions':len(seen),'families':len(families),'media':len(media),'target_sources':dict(sources),'frozen_original_dev_questions':len(samples) if spec['frozen_original_dev'] else 0})
 for cohort in ('text','image','screenshot','audio','video','crossmodal'):
  assert original[cohort]==500
  for pool in ('dev','calibration','holdout'):
   items=groups[(cohort,pool)];assert len(items)==len(set(items))
   assert len(items)>=500 if pool=='dev' else len(items)==(3000 if pool=='calibration' else 1000)
 summary.update(outcome='selected_counts_and_targets_verified',exit_code=0,artifacts=counts,licenses=dict(licenses),original_development=dict(original),selected_question_counts={c:{p:len(groups[(c,p)]) for p in ('dev','calibration','holdout')} for c in original})
except Exception:summary['failure']='selection_rejected'
summary.update(manifest_sha256=hashlib.sha256(payload).hexdigest(),inventory_sha256=config['inventory']['sha256'],runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=round(time.monotonic()-start,3))
with (root/'reports/s4-selected-gold-audit-001.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps({k:v for k,v in summary.items() if k not in ('artifacts','licenses')}));raise SystemExit(summary['exit_code'])
