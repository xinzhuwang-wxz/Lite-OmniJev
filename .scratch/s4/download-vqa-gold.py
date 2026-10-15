"""Bounded official-image downloader; all per-item manifests stay private."""
import concurrent.futures,hashlib,json,os,re,threading,time,urllib.request,urllib.error
from collections import Counter
from pathlib import Path
start=time.monotonic();root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001'
planpath=private/'vqa-gold-plan-002/plan.json'
assert hashlib.sha256(planpath.read_bytes()).hexdigest()=='748dc460c65953c30f6d7cd21a37034ce3c49c4eddf235a2c584fd6559b9bb1f'
plan=json.loads(planpath.read_bytes());requests_path=planpath.parent/'download-requests.jsonl'
assert hashlib.sha256(requests_path.read_bytes()).hexdigest()==plan['requests_sha256']
requests=[json.loads(line) for line in requests_path.read_text().splitlines()]
assert len(requests)==4500 and len({r['image_id'] for r in requests})==4500
candidates_path=planpath.parent/'candidates.jsonl'
assert hashlib.sha256(candidates_path.read_bytes()).hexdigest()==plan['candidates_sha256']
candidates={r['image_id']:r for r in (json.loads(line) for line in candidates_path.read_text().splitlines())}
assert len(candidates)==4500
for row in requests:
 transport=candidates[row['image_id']]['attribution']['coco_url']
 assert transport==row['url'].replace('https:', 'http:', 1)
 row['transport_url']=transport
out=private/'vqa-downloads-003';out.mkdir(mode=0o700)
for row in requests:
 assert row['file_name']==f"{row['image_id']:012d}.jpg"
 assert row['url']=='https://images.cocodataset.org/train2017/'+row['file_name']
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs): raise ValueError('redirect_rejected')
stop=threading.Event();lock=threading.Lock();failed=0

def fetch(row):
 global failed
 if stop.is_set() or time.monotonic()-start>1200: return {'image_id':row['image_id'],'error':'budget_or_circuit_stop'}
 category='network'
 for attempt in range(2):
  try:
   opener=urllib.request.build_opener(NoRedirect())
   with opener.open(urllib.request.Request(row['transport_url'],headers={'User-Agent':'Lite-OmniJev-source-audit'}),timeout=25) as response:
    if response.geturl()!=row['transport_url']: raise ValueError('redirect_rejected')
    payload=response.read(16*1024*1024+1)
   if not 0<len(payload)<=16*1024*1024 or not payload.startswith(b'\xff\xd8'):raise ValueError('invalid_image_bytes')
   path=out/row['file_name']
   fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
   with os.fdopen(fd,'wb') as h:h.write(payload)
   return {'image_id':row['image_id'],'path':str(path),'url':row['url'],'transport_url':row['transport_url'],'final_url':row['transport_url'],'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
  except ValueError: category='redirect_or_invalid_bytes';break
  except urllib.error.HTTPError as exc:
   category='http_'+str(exc.code)
   if exc.code not in (408,429,500,502,503,504):break
  except (OSError,urllib.error.URLError,TimeoutError): category='network'
 with lock:
  failed+=1
  if failed>=8:stop.set()
 return {'image_id':row['image_id'],'error':category}

results=[fetch(row) for row in requests[:3]]
if not any('error' in row for row in results):
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  results.extend(pool.map(fetch,requests[3:]))
else: stop.set()
passed=[r for r in results if 'error' not in r]; errors=[r for r in results if 'error' in r]
def write(name,rows):
 fd=os.open(out/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as h:
  for row in rows:h.write(json.dumps(row,sort_keys=True)+'\n')
write('download-manifest.jsonl',passed);write('failures.jsonl',errors)
report={'scope':'selected official VQA2 images only, no gold admission','requested':len(requests),'attempted':len(results),'downloaded':len(passed),'failed':len(errors),'errors':dict(Counter(r['error'] for r in errors)),'stored_bytes':sum(r['bytes'] for r in passed),'plan_sha256':hashlib.sha256(planpath.read_bytes()).hexdigest(),'download_manifest_sha256':hashlib.sha256((out/'download-manifest.jsonl').read_bytes()).hexdigest(),'failure_manifest_sha256':hashlib.sha256((out/'failures.jsonl').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':round(time.monotonic()-start,3),'exit_code':int(len(passed)!=4500)}
with (root/'reports/s4-vqa-gold-download-003.json').open('x') as h:json.dump(report,h,indent=2)
print(json.dumps(report))
raise SystemExit(report['exit_code'])
