"""Validate official image-budget preprocessing without loading model weights."""
import json,time,hashlib
from pathlib import Path
from transformers import AutoProcessor
from data.schema import Sample
from train.trainer import prepare_media_example
root=Path('/root/autodl-tmp/s1-text-lora-001');model=Path('/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2')
data=root/'artifacts/s2-media-pools-image-001/run-a/train.jsonl';sample=Sample.model_validate_json(data.read_text().splitlines()[0]);processor=AutoProcessor.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
results=[]
for budget in [280,1120]:
 processor.image_processor.max_soft_tokens=budget;start=time.monotonic()
 x=prepare_media_example(sample,processor,Path('/root/autodl-tmp/lite-omnijev/artifacts/data/quarantine/stage2-sources/aokvqa/coco-ccby-publicdomain-images'),max_length=2048)
 results.append({'budget':budget,'prompt_tokens':len(x['input_ids']),'image_tokens':x['input_ids'].count(processor.tokenizer.image_token_id),'pixel_shape':list(x['pixel_values'].shape),'seconds':time.monotonic()-start})
assert results[1]['image_tokens']>results[0]['image_tokens']
r={'status':'passed','scope':'single train example CPU official processing only; not GPU quality evidence','data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'measurements':results};(root/'reports/s2-image-budget-preflight-001.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
