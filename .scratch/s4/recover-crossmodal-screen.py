"""Recover a completed public GPU sidecar after its receipt finalizer failed."""
import hashlib
import json
from pathlib import Path
from scripts import s0_development_baselines as b
from train.readout import read_probabilities
root=Path('/root/autodl-tmp/s1-text-lora-001')
data=Path('/root/autodl-tmp/s0-crossmodal-development-001/artifacts/crossmodal-dev-001-a/dev.jsonl')
assert b.sha256_file(data)=='8099d5c78af26fd0848dd36d06a52b68906e6789cb2c55f7c62d8773485298af'
selected=sorted(b.load_samples(data),key=lambda s:hashlib.sha256(('42:s4-crossmodal:'+s.id).encode()).hexdigest())[:50]
sidecar=root/'artifacts/s4-crossmodal-screen-001/records.jsonl'
rows=[json.loads(line) for line in sidecar.read_text().splitlines()]
assert len(rows)==50 and [r['id'] for r in rows]==[s.id for s in selected]
for sample,row in zip(selected,rows,strict=True):
    assert row['pass_index']==1 and row['target_index']==b.target_index(sample.questions[0])
    tokens=b.answer_tokens_for(sample.questions[0])
    readout=read_probabilities(row['raw_logprobs'],tokens)
    assert readout.probabilities==row['conditional_without_u'] and readout.p_insufficient==row['p_U'] and readout.prediction==row['prediction']
report={'status':'completed_records_recovered','scope':'single-pass fixed50 publicdevelopment diagnostic only; not stage/quality acceptance','samples':50,'passes':1,'seed':42,'dtype':'bfloat16','image_max_soft_tokens':1120,'metrics':b.score_records(rows),'data_sha256':b.sha256_file(data),'selected_ids_sha256':hashlib.sha256('\n'.join(s.id for s in selected).encode()).hexdigest(),'records_sha256':b.sha256_file(sidecar),'adapter_sha256':b.sha256_file(root/'artifacts/s3-stable-probe-train-001/adapter/adapter_model.safetensors'),'adapter_config_sha256':b.sha256_file(root/'artifacts/s3-stable-probe-train-001/adapter/adapter_config.json'),'base_revision':'ee0ef6023621cff504d758262d4e04895a5af4a2','source_code_commit':'f1ad912c1ea64267258c55fa6d4a234a677e5635','code_sha256':{p:b.sha256_file(Path(p)) for p in b.REQUIRED_CODE_FILES},'original_runner_sha256':b.sha256_file(Path('/root/autodl-tmp/screen-crossmodal-candidate.py')),'recovery_runner_sha256':b.sha256_file(Path(__file__)),'failure_log_sha256':b.sha256_file(Path('/root/autodl-tmp/s4-crossmodal-screen-001.log')),'original_process_exit_code':1,'failure':'receipt_finalizer_called_code_hashes_with_argument','recovery_exit_code':0,'stage_gate_pass':False,'prepare_seconds_sum':sum(r['prepare_wall_seconds'] for r in rows),'forward_seconds_sum':sum(r['forward_wall_seconds'] for r in rows),'elapsed_seconds':None,'peak_cuda_allocated_bytes':None,'cost_usd':None,'limitations':['Original receipt finalizer failed after all50records; no inference rerun.','Total wall time and peak CUDA memory were lost with process and are not reconstructed.','Source model files previously verified; this diagnostic did not independently rehash every base shard.']}
with (root/'reports/s4-crossmodal-screen-recovered-001.json').open('x') as h: json.dump(report,h,indent=2)
print(json.dumps(report))
