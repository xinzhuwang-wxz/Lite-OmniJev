"""Actual official processor comparison on public dev; no model/GPU load."""
import hashlib,json,time
from pathlib import Path
import torch
from scripts import s0_development_baselines as baseline
from train.trainer import prepare_media_example,MEDIA_FORWARD_KEYS
from train.multimodal import configure_image_processor
start=time.monotonic()
root=Path('/root/autodl-tmp/s1-text-lora-001')
source=Path('/root/autodl-tmp/s0-crossmodal-development-001/artifacts/crossmodal-dev-001-a/dev.jsonl')
assert baseline.sha256_file(source)=='8099d5c78af26fd0848dd36d06a52b68906e6789cb2c55f7c62d8773485298af'
model=Path('/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2')
processor=baseline.default_processor_factory(model)
configure_image_processor(processor,1120)
samples=sorted(baseline.load_samples(source),key=lambda s:hashlib.sha256(('42:s4-crossmodal:'+s.id).encode()).hexdigest())[:10]
keys=set()
for sample in samples:
 reference=baseline.prepare_sample(sample,processor,source.parent/'media')
 current=prepare_media_example(sample,processor,source.parent/'media',max_length=8192)
 assert current['input_ids']==reference.inputs['input_ids'][0].tolist()
 assert current['answer_token_ids']==reference.answer_token_ids
 assert current['target'][reference.target_index]==1.0
 for key in MEDIA_FORWARD_KEYS:
  assert (key in current)==(key in reference.inputs)
  if key in current: assert torch.equal(current[key],reference.inputs[key]); keys.add(key)
 extras=set(reference.inputs)-set(MEDIA_FORWARD_KEYS)-{'input_ids','attention_mask'}
 if extras:
  print(json.dumps({'extra_keys':{k:{'shape':list(reference.inputs[k].shape),'unique':reference.inputs[k].unique().tolist()} for k in extras}}))
  raise SystemExit(2)
 assert bool(reference.inputs['attention_mask'].eq(1).all())
report={'status':'passed','scope':'real official CPU processor parity for existing public crossmodal dev, not training or model quality','samples':len(samples),'exact_input_ids_answer_mapping_and_media_tensors':True,'tensor_keys':sorted(keys),'image_max_soft_tokens':1120,'data_sha256':baseline.sha256_file(source),'trainer_sha256':baseline.sha256_file(Path('train/trainer.py')),'baseline_sha256':baseline.sha256_file(Path('scripts/s0_development_baselines.py')),'runner_sha256':baseline.sha256_file(Path(__file__)),'elapsed_seconds':round(time.monotonic()-start,3),'exit_code':0}
with (root/'reports/s4-crossmodal-preparation-parity-001.json').open('x') as h:json.dump(report,h,indent=2)
print(json.dumps(report))
