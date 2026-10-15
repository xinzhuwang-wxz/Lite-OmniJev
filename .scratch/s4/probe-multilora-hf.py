"""Same synthetic prompt HF reference for the bounded native routing probe."""
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from peft import PeftModel
root = Path('/root/autodl-tmp/s1-text-lora-001')
report_path = root / 'artifacts/s4-native-multilora-probe-003/hf-reference.json'
assert not report_path.exists()
model_dir = '/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2'
r = json.loads((report_path.parent / 'report.json').read_text())
assert r['exit_after_cleanup'] == 0
start = time.monotonic()
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
payload = json.dumps({'state': 'The red box contains two apples.', 'question': 'Does the red box contain apples?', 'answers': {'A': 'yes', 'B': 'no'}, 'U': 'Insufficient evidence to decide', 'response': 'Return exactly one answer token, or U if evidence is insufficient.'}, separators=(',', ':'))
prompt = tok.apply_chat_template([{'role':'user','content':payload}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
ids = r['answer_token_ids']
base = Gemma4ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
model = PeftModel.from_pretrained(base, str(root/'artifacts/s4-joint-pilot-train-001/adapter'), adapter_name='general', local_files_only=True)
model.load_adapter(str(root/'artifacts/s4-video-pilot-train-001/adapter'), adapter_name='video', local_files_only=True)
model.eval()
records=[]
with torch.inference_mode():
    for name in ['general','video','general','video']:
        model.set_adapter(name)
        logits=model(**inputs).logits[0,-1].float()
        logprobs=logits.log_softmax(-1)[ids].tolist()
        records.append({'adapter':name,'raw_logprobs':logprobs})
report={'scope':'single_synthetic_text_only_not_stage_parity','records':records,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'input_tokens':inputs.input_ids.shape[-1],'answer_token_ids':ids,'wall_seconds':time.monotonic()-start,'adapter_sha256':r['adapter_sha256']}
report_path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
