"""Research-only official image-budget override; never changes model assets."""
import hashlib
import json
import sys
from pathlib import Path
import transformers
from scripts import s1_dev_eval

BUDGET=1120
original=transformers.AutoProcessor.from_pretrained
seen=[]
def configured(*args,**kwargs):
    processor=original(*args,**kwargs)
    before=processor.image_processor.max_soft_tokens
    processor.image_processor.max_soft_tokens=BUDGET
    seen.append({'original_max_soft_tokens':before,'max_soft_tokens':processor.image_processor.max_soft_tokens})
    return processor
transformers.AutoProcessor.from_pretrained=staticmethod(configured)
args=s1_dev_eval.parse_args(sys.argv[1:])
code=s1_dev_eval.main(sys.argv[1:])
r=json.loads(args.report.read_text());r['research_input_profile']={'scope':'official variable-resolution pilot; not stage acceptance','image_max_soft_tokens':BUDGET,'processor_overrides':seen,'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'official_reference':'https://ai.google.dev/gemma/docs/capabilities/vision/image'}
args.report.write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(code)
