import json,hashlib
from pathlib import Path
base='/root/autodl-tmp/s1-text-lora-001/';private=base+'artifacts/s4-gold-inventory-001/'
configs=[('gui','s4-gui-gold-build-002.json','gui-gold-002','gui-gold-002/build-manifest.json'),('clotho','s4-clotho-gold-build-001.json','clotho-gold-001','clotho-gold-001/build-manifest.json'),('speech','s4-speech-gold-build-001.json','speech-gold-001','speech-build-manifest-001.json'),('video','s4-clevrer-gold-build-001.json','clevrer-validation-gold-001','clevrer-build-manifest-001.json'),('vqa','s4-vqa-gold-build-001.json','vqa-gold-001','vqa-gold-plan-002/plan.json'),('crossmodal','s4-crossmodal-gold-assembly-001.json','crossmodal-gold-001','crossmodal-gold-input-001.json'),('crossmodal_dev','s4-crossmodal-dev-assembly-002.json','crossmodal-dev-002','crossmodal-dev-input-002.json')]
items=[]
for name,report,folder,manifest in configs:
 p=Path('reports')/report;raw=p.read_bytes();r=json.loads(raw);detail=r.get('result',r)
 expected=detail.get('input_sha256',{}).get('build_manifest',detail.get('input_sha256',{}).get('manifest'))
 if name=='vqa':expected='748dc460c65953c30f6d7cd21a37034ce3c49c4eddf235a2c584fd6559b9bb1f'
 assert expected
 outputs=[]
 for k,v in detail['output_sha256'].items():
  filename={'calibration':'calibration.jsonl','holdout':'holdout.jsonl','dev':'dev.jsonl','inventory':'inventory-manifest.json','aliases':'alias-ledger.jsonl','components':'components.jsonl'}.get(k,k)
  outputs.append({'path':private+folder+'/'+filename,'sha256':v})
 items.append({'source':name,'receipt':{'path':base+'reports/'+report,'sha256':hashlib.sha256(raw).hexdigest()},'input_manifest':{'path':private+manifest,'sha256':expected},'outputs':outputs})
p=Path('.scratch/s4/source-evidence-bindings-001.json');p.write_text(json.dumps({'sources':items,'scope':'Successful executed original-gold builders and composed program truth; source metadata strings alone are insufficient.'},indent=2)+'\n');print(hashlib.sha256(p.read_bytes()).hexdigest())
