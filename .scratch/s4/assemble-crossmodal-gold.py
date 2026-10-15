"""Assemble already-built private artifacts; export allowlisted stage receipts only."""
import contextlib,hashlib,io,json,os,time
from pathlib import Path
from scripts.gold_inventory import audit_manifest
from scripts.gold_media_bundle import bundle
from scripts.crossmodal_gold_pools import build
root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001'
start=time.monotonic()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path,expected):
 assert sha(path)==expected
 return json.loads(path.read_bytes())
def write(path,value):
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True)
 return sha(path)
def stage_report(name,value):write(root/'reports'/name,value)
summary={'scope':'private crossmodal assembly; not full gold gate','status':'rejected','exit_code':1}
try:
 with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
  vr=json.loads((root/'reports/s4-vqa-gold-build-001.json').read_bytes())
  assert vr['exit_code']==0 and vr['status']=='materialized' and not vr['synthetic_canary']
  vi=load(private/'vqa-gold-001/inventory-manifest.json',vr['output_sha256']['inventory'])
  basepath=private/'with-audio-relocated-inventory-001.json'
  merged=load(basepath,'48d41081a4b516782a1ee598d3c4976d4123f6501d5206f3a0323a0ce0cd9947')
  added=[a for a in vi['samples'] if Path(a['path']).parent==private/'vqa-gold-001']
  assert len(added)==3 and {a['expected_pool'] for a in added}=={'dev','calibration','holdout'}
  merged['samples'].extend(added)
  mp=private/'with-vqa-gui-audio-inventory-001.json';write(mp,merged)
  audit=audit_manifest(mp);stage_report('s4-vqa-gui-audio-isolation-001.json',audit.model_dump(mode='json'))
  assert audit.outcome=='isolated'
  inputs=[];dataset_by_old={}
  amap=load(private/'audio-media-bundle-001/mapping.json','ffd6bcaa276585553846a012c3d0bb11b7970ae0fac6559f82b85f560634f181')
  for m in amap:
   dataset_by_old[m['rewritten']['path']]='Clotho-AQA' if Path(m['original']['path']).parent.name=='clotho-gold-001' else 'LibriSpeech'
  for a in merged['samples']:
   if a['exposure']!='current' or a['expected_pool'] not in ('calibration','holdout'):continue
   if Path(a['path']).parent==private/'vqa-gold-001':
    inputs.append({**a,'media_root':str(private)});dataset_by_old[a['path']]='VQA-v2'
   elif a['path'] in dataset_by_old:inputs.append({**a,'media_root':str(private/'audio-media-bundle-001')})
  assert len(inputs)==6
  bm=private/'crossmodal-source-bundle-input-001.json'
  write(bm,{'schema_version':'gold-media-bundle-manifest-v1','samples':inputs})
  out=private/'crossmodal-source-bundle-001';br=bundle(bm,out)
  stage_report('s4-crossmodal-source-bundle-001.json',br.model_dump(mode='json'))
  assert br.outcome=='bundled'
  mapping=json.loads((out/'mapping.json').read_bytes());rewritten=json.loads((out/'inventory-manifest.json').read_bytes())
  assert len(mapping)==len(rewritten['samples'])==6
  sources=[]
  for m in mapping:
   old=[a for a in merged['samples'] if a['path']==m['original']['path'] and a['sha256']==m['original']['sha256'] and a['exposure']=='current'];assert len(old)==1
   old[0]['exposure']='historical'
   new=[a for a in rewritten['samples'] if a['path']==m['rewritten']['path'] and a['sha256']==m['rewritten']['sha256']];assert len(new)==1
   new=new[0];assert new['expected_pool']==old[0]['expected_pool'] and new['cohort']==old[0]['cohort']
   merged['samples'].append(new)
   sources.append({'path':new['path'],'sha256':new['sha256'],'pool':new['expected_pool'],'dataset':dataset_by_old[m['original']['path']]})
  cp=private/'with-crossmodal-sources-inventory-001.json';ch=write(cp,merged)
  ca=audit_manifest(cp);stage_report('s4-crossmodal-source-isolation-001.json',ca.model_dump(mode='json'));assert ca.outcome=='isolated'
  cm=private/'crossmodal-gold-input-001.json';write(cm,{'inventory':{'path':str(cp),'sha256':ch},'sources':sources})
  result=build(cm,out,private/'crossmodal-gold-001')
  summary.update(status='built',exit_code=0,result=result,source_inventory_sha256=ch,runner_sha256=sha(Path(__file__)),code_sha256={name:sha(Path('scripts')/(name+'.py')) for name in ('crossmodal_gold_pools','gold_media_bundle','gold_inventory')})
except Exception:summary['failure']='assembly_rejected'
summary['elapsed_seconds']=round(time.monotonic()-start,3)
stage_report('s4-crossmodal-gold-assembly-001.json',summary)
print(json.dumps(summary));raise SystemExit(summary['exit_code'])
