"""Bind the already-pinned E4B and S3 adapter for a public-dev engineering run."""
import hashlib,json,os,time
from pathlib import Path
root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001';base=Path('/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2');adapter=root/'artifacts/s3-stable-probe-train-001/adapter';start=time.monotonic()
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def files(directory):
 result={}
 for p in sorted(directory.rglob('*')):
  if p.is_file() and p.suffix in ('.json','.safetensors','.jinja','.model'):
   assert not p.is_symlink();result[str(p.relative_to(directory))]=sha(p)
 return result
mf=files(base);af=files(adapter);assert af['adapter_model.safetensors']=='1bebae31309f799395d01499fb9deec5e44622599d0c082473ea8b5a014e5755'
data=private/'crossmodal-dev-002/dev.jsonl';assert sha(data)=='1d2d8e151a3bc282099764a5ce7d04e8190fd212684ca010582511c5b25cfde1'
spec={'mode':'engineering','cohort':'crossmodal','data':{'path':str(data),'sha256':sha(data)},'pool':'dev','media_root':str(private/'crossmodal-dev-source-bundle-002'),'model_dir':str(base),'model_id':'gemma4-e4b-it','base_revision':base.name,'model_files':mf,'adapter_dir':str(adapter),'adapter_files':af,'expected_rows':500,'subset_limit':50,'seed':42,'image_budget':1120,'video_frames':16,'output_root':str(root/'artifacts/s4-private-stage-eval-001'),'timeout_seconds':600}
p=private/'private-stage-engineering-input-001.json';fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:json.dump(spec,f,sort_keys=True)
report={'scope':'bound local model assets and publicdev engineering spec only','model_files':len(mf),'adapter_files':len(af),'spec_sha256':sha(p),'data_sha256':sha(data),'elapsed_seconds':round(time.monotonic()-start,3),'exit_code':0}
with (root/'reports/s4-private-stage-engineering-prep-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
