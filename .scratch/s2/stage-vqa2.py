import hashlib
import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
root=Path('/root/autodl-tmp/s1-text-lora-001')
out=Path('/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/vqa2-annotations-20261015')
out.mkdir(exist_ok=True)
files={'v2_Annotations_Train_mscoco.zip':21708861,'v2_Annotations_Val_mscoco.zip':10518930,'v2_Questions_Train_mscoco.zip':7239401,'v2_Questions_Val_mscoco.zip':3494929}
start=time.time();rows=[]
def stage(item):
 name,size=item
 url='https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/'+name;p=out/name
 if not p.exists():
  partial=out/(name+'.partial')
  subprocess.run(['aria2c','--continue=true','--auto-file-renaming=false','--file-allocation=none','--max-connection-per-server=8','--split=8','--min-split-size=1M','--connect-timeout=15','--timeout=30','--max-tries=2','--console-log-level=warn','--summary-interval=0','--download-result=hide','--dir',str(out),'--out',partial.name,url],check=True,timeout=240)
  assert partial.stat().st_size==size
  partial.rename(p)
 assert p.stat().st_size==size
 return {'url':url,'path':str(p),'bytes':size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with ThreadPoolExecutor(max_workers=4) as pool:
 rows=list(pool.map(stage,files.items()))
report={'status':'staged','exit_code':0,'training_admitted':False,'scope':'official public VQA v2 train/val annotations/questions only; no new media, no test','files':rows,'duration_seconds':time.time()-start,'license_evidence':'https://visualqa.org/terms.html','license':'CC-BY-4.0 annotations only'}
(root/'reports/s2-vqa2-staging-001.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
