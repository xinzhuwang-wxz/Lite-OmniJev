"""Add source-matched public dev diagnostics, keeping the original cohort intact."""
import contextlib,hashlib,io,json,os,time
from pathlib import Path
from scripts.gold_inventory import audit_manifest
from scripts.gold_media_bundle import bundle
from crossmodal_gold_pools_04a5239 import build
root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001'
start=time.monotonic()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(v,f,sort_keys=True)
 return sha(p)
summary={'scope':'supplemental source-matched dev only; original500 preserved','status':'rejected','exit_code':1}
try:
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  ip=private/'crossmodal-gold-001/inventory-manifest.json';assert sha(ip)=='9c3e02ccbd45b577301a49e1f889a607869bd22de143511675126112ff08037e'
  inventory=json.loads(ip.read_bytes())
  config={str(private/'vqa-gold-001/dev.jsonl'):('VQA-v2',private),str(private/'speech-gold-001/dev.jsonl'):('LibriSpeech',private/'speech-gold-001'),'/root/autodl-tmp/s0-audio-dev-001/artifacts/audio-dev-001-a/dev.jsonl':('Clotho-AQA',Path('/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/clotho-aqa/6473207/approved-audio-002'))}
  selected=[a for a in inventory['samples'] if a['path'] in config and a['expected_pool']=='dev'];assert len(selected)==3
  bp=private/'crossmodal-dev-bundle-input-001.json';write(bp,{'schema_version':'gold-media-bundle-manifest-v1','samples':[{**a,'media_root':str(config[a['path']][1])} for a in selected]})
  out=private/'crossmodal-dev-source-bundle-001';br=bundle(bp,out);write(root/'reports/s4-crossmodal-dev-source-bundle-001.json',br.model_dump(mode='json'));assert br.outcome=='bundled'
  mapping=json.loads((out/'mapping.json').read_bytes());rewritten=json.loads((out/'inventory-manifest.json').read_bytes());sources=[]
  for m in mapping:
   originals=[a for a in inventory['samples'] if a['path']==m['original']['path'] and a['sha256']==m['original']['sha256']];assert len(originals)==1
   originals[0]['exposure']='historical'
   new=[a for a in rewritten['samples'] if a['path']==m['rewritten']['path'] and a['sha256']==m['rewritten']['sha256']];assert len(new)==1
   new=new[0];new['exposure']='current';inventory['samples'].append(new)
   sources.append({'path':new['path'],'sha256':new['sha256'],'pool':'dev','dataset':config[m['original']['path']][0]})
  cp=private/'with-crossmodal-dev-sources-inventory-001.json';h=write(cp,inventory);audit=audit_manifest(cp);assert audit.outcome=='isolated'
  cm=private/'crossmodal-dev-input-001.json';write(cm,{'inventory':{'path':str(cp),'sha256':h},'sources':sources})
  result=build(cm,out,private/'crossmodal-dev-001',mode='supplemental-dev')
  summary.update(status='built',exit_code=0,result=result)
except Exception:summary['failure']='assembly_rejected'
summary.update(elapsed_seconds=round(time.monotonic()-start,3),runner_sha256=sha(Path(__file__)))
write(root/'reports/s4-crossmodal-dev-assembly-001.json',summary)
print(json.dumps(summary));raise SystemExit(summary['exit_code'])
