"""Bounded S4 reference-retention phases; preserves the original gold and draws."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('phase',choices=['preflight','reference','train'])
a=parser.parse_args()
root=Path('/root/autodl-tmp/s1-text-lora-001')
source=Path('/root/autodl-tmp/s4-reference-code-fix-001')
python='/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python'
prior=root/'artifacts/s4-video-pilot-001'
first=root/'artifacts/s4-joint-pilot-001/train.jsonl'
out=root/'artifacts/s4-reference-retention-001'
out.mkdir(exist_ok=True,mode=0o700)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
base=[python,'-m','scripts.train_reference','--config',str(prior/'config-30.json'),'--data',str(prior/'train.jsonl'),'--ledger',str(prior/'allocation-ledger.jsonl'),'--model-dir','/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2','--asset-manifest','/root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json','--reference-adapter',str(root/'artifacts/s4-joint-pilot-train-001/adapter'),'--reference-adapter-sha256','6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac','--replay-data',str(first),'--replay-data-sha256',sha(first)]
receipt={'phase':a.phase,'source_commit':'44214af','source_archive_sha256':sha(Path('/root/autodl-tmp/omnijev-s4-reference-fix.tar')),'prior_config_sha256':sha(prior/'config-30.json'),'data_sha256':sha(prior/'train.jsonl'),'ledger_sha256':sha(prior/'allocation-ledger.jsonl'),'replay_sha256':sha(first),'quality_passed':False,'cost_usd':None}
report_path=root/f'reports/s4-reference-retention-{a.phase}-001.json'
assert not report_path.exists()
if a.phase=='preflight':
    command=base+['--output',str(out/'reference-preflight.json')]
    limit=120
elif a.phase=='reference':
    before=json.loads((out/'reference-preflight.json.report.json').read_text())
    assert before['status']=='preflight_verified' and before['exit_code']==0
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    command=base+['--output',str(out/'reference.json'),'--execute']
    limit=240
else:
    before=json.loads((out/'reference.json.report.json').read_text())
    assert before['status']=='completed' and before['draws']==300
    assert sha(out/'reference.json')==before['sidecar_sha256']
    config=json.loads((prior/'config-30.json').read_text())
    config.update(reference_sidecar=str(out/'reference.json'),reference_sidecar_sha256=sha(out/'reference.json'),reference_ledger=str(prior/'allocation-ledger.jsonl'),reference_ledger_sha256=sha(prior/'allocation-ledger.jsonl'),reference_kl_weight=0.5,reference_replay_data=str(first),reference_replay_sha256=sha(first))
    config_path=out/'config-30.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'existing config differs'
    else:
        with config_path.open('x') as handle:json.dump(config,handle,indent=2)
    command=json.loads((prior/'report.json').read_text())['preflight_argv'][3:]
    command=[python]+command
    for flag,value in [('--config',str(config_path)),('--output-dir',str(root/'artifacts/s4-reference-retention-train-001')),('--report',str(root/'reports/s4-reference-retention-training-001.json'))]:
        command[command.index(flag)+1]=value
    preflight=list(command)
    preflight[preflight.index('--report')+1]=str(root/'reports/s4-reference-retention-training-preflight-002.json')
    result=subprocess.run(preflight,cwd=source,env=env,capture_output=True,text=True,timeout=120)
    (out/'training-preflight-002.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,'training preflight failed'
    command[command.index('preflight')]='run'
    env.update(SWANLAB_API_KEY=Path('/root/.config/lite-omnijev/swanlab.key').read_text().strip(),LITE_OMNIJEV_SWANLAB_MODE='online')
    limit=480
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
receipt['command']=command
start=time.monotonic()
with (out/(a.phase+'.log')).open('x') as log:
    result=subprocess.run(['timeout','--signal=INT','--kill-after=20s',str(limit),*command],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
receipt.update(exit_code=result.returncode,wall_seconds=time.monotonic()-start)
with report_path.open('x') as handle:json.dump(receipt,handle,indent=2)
print(json.dumps(receipt))
raise SystemExit(result.returncode)
