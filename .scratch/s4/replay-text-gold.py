"""Private CLINC source replay; emits only aggregate evidence."""
import contextlib
import hashlib
import io
import json
import platform
import time
from pathlib import Path
from scripts import admit_clinc as source
from scripts import s0_text_dev as dev
from scripts import s1_text_pools as pools

start=time.monotonic()
root=Path('/root/autodl-tmp/s1-text-lora-001')
report={'schema_version':'clinc-gold-source-replay-v1','outcome':'rejected','exit_code':1,'full_gold_gate':'not_validated','private_job_permissions':'not_validated'}
def digest(data):return hashlib.sha256(data).hexdigest()
def encoded(rows):return ''.join(s.model_dump_json()+'\n' for s in rows).encode()
try:
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  path=Path('/root/autodl-tmp/lite-omnijev/artifacts/data/raw/clinc-clinc_oos')/source.REVISION/'train.parquet'
  assert source.sha256_file(path)==source.EXPECTED_TRAIN_SHA256
  rows,labels=source.read_clinc_parquet(path)
  admitted,_=source.build_samples(rows,labels,source_sha256=source.EXPECTED_TRAIN_SHA256)
  assert digest(encoded(admitted))==dev.EXPECTED_ADMITTED_SHA256
  development,remaining=dev.split_samples(admitted,dev_count=500)
  assert digest(encoded(remaining))==pools.EXPECTED_SOURCE_SHA256
  assert digest(encoded(development))==pools.EXPECTED_DEV_SHA256
  held,cal,train=pools.split_pools(remaining)
  expected={'holdout':('holdout-reserve.jsonl','6bba594d544fdb67dc7dd95d37ce22de3570ad91d2db5dd0e633828eb87542c6',held),'calibration':('calibration-reserve.jsonl','6d3878168fbe135bb38bf1253ba8534d88b00b71654235c0bf619a2481a8a0c9',cal),'train':('train.jsonl','75206358039678400da9c52302b854468320562117f72041b1d01238006135a9',train)}
  for name,(filename,sha,items) in expected.items():
   assert digest(encoded(items))==sha
   assert digest((root/'artifacts/s1-text-pools-v1-001/run-a'/filename).read_bytes())==sha
  report.update(outcome='matched_original_gold',exit_code=0,counts={k:len(v[2]) for k,v in expected.items()},dev_count=len(development),source_sha256=source.EXPECTED_TRAIN_SHA256,output_sha256={k:v[1] for k,v in expected.items()},source_revision=source.REVISION,license=source.LICENSE,rule='Pinned CLINC ClassLabel -> original Choice mapping -> frozen dev exclusion -> frozen S1 family split',code_sha256={m.__name__:digest(Path(m.__file__).read_bytes()) for m in (source,dev,pools)})
except Exception:
 report['failure']='source_replay_rejected'
report.update(elapsed_seconds=round(time.monotonic()-start,3),runner_sha256=digest(Path(__file__).read_bytes()),python=platform.python_version())
with (root/'reports/s4-text-gold-source-replay-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
raise SystemExit(report['exit_code'])
