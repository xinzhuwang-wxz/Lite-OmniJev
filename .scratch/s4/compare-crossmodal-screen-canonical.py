import json,hashlib
from collections import Counter
from pathlib import Path
import s0_development_baselines_corrected as b
root=Path('/root/autodl-tmp/s1-text-lora-001')
reports=[json.loads((root/f'reports/{name}').read_bytes()) for name in ('s4-crossmodal-screen-recovered-002.json','s4-crossmodal-base-screen-001.json')]
for key in ('data_sha256','selected_ids_sha256','dtype','image_max_soft_tokens','base_revision','code_sha256'): assert reports[0][key]==reports[1][key]
paths=[root/'artifacts'/name/'records.jsonl' for name in ('s4-crossmodal-screen-001','s4-crossmodal-base-screen-001')]
rows=[]
for p,r in zip(paths,reports,strict=True):
 assert b.sha256_file(p)==r['records_sha256'];rows.append([json.loads(line) for line in p.read_text().splitlines()])
counts=Counter(); preds=[Counter(),Counter()]
for a,c in zip(*rows,strict=True):
 for key in ('id','target_index','prompt_sha256','payload_sha256','input_sha256','input_summary_sha256','answer_token_ids'): assert a[key]==c[key]
 tokens=b.answer_tokens_for_record(a); gold=tokens[a['target_index']]
 ac=a['prediction']==gold; bc=c['prediction']==gold
 counts['both_correct' if ac and bc else 'adapter_only_correct' if ac else 'base_only_correct' if bc else 'both_wrong']+=1
 preds[0][a['prediction']]+=1;preds[1][c['prediction']]+=1
out={'scope':'paired fixed50 publicdev diagnostic, population/regression and stage acceptance unproven','identical_prepared_input_evidence':True,'samples':len(rows[0]),'adapter_accuracy':reports[0]['metrics']['accuracy'],'base_accuracy':reports[1]['metrics']['accuracy'],'paired_counts':dict(counts),'adapter_prediction_counts':dict(preds[0]),'base_prediction_counts':dict(preds[1]),'source_receipt_sha256':[b.sha256_file(root/'reports'/name) for name in ('s4-crossmodal-screen-recovered-002.json','s4-crossmodal-base-screen-001.json')],'exit_code':0,'next_action':'Keep multi-input training/eval support on S4 plan; small-screen difference is not full quality proof'}
with (root/'reports/s4-crossmodal-paired-screen-002.json').open('x') as h:json.dump(out,h,indent=2)
print(json.dumps(out))
