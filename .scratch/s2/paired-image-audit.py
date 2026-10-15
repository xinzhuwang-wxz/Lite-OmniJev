"""Summarize paired development errors using existing GPU prediction files only."""
import hashlib
import json
from pathlib import Path
root=Path('/root/autodl-tmp/s1-text-lora-001')
sidecar=root/'artifacts/s2-text-lora-001/dev-eval-image-v2-corrected-001/records.jsonl'
student={r['id']:r for r in map(json.loads,sidecar.read_text().splitlines()) if r['variant']=='original' and r['pass']==0}
assert len(student)==500
results={}
for name in ['gemma4-e4b-it','onejev-4b','onejev-27b']:
 report_path=root/f'reports/s0-development-image-{name}-001.json'
 report=json.loads(report_path.read_text())
 path=Path(report['records_sidecar']['path'])
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 expected=report.get('verified_sidecar_sha256') or report['records_sidecar']['sha256']
 assert digest==expected,(name,digest,expected)
 records=list(map(json.loads,path.read_text().splitlines()))
 first=min(r['pass_index'] for r in records)
 teacher={r['id']:r for r in records if r['pass_index']==first}
 assert teacher.keys()==student.keys()
 both_correct=student_only=reference_only=both_wrong=0
 for key,s in student.items():
  t=teacher[key]
  target=chr(ord('A')+t['target_index'])
  assert s['target']==target
  sc=s['prediction']==target;tc=t['prediction']==target
  both_correct+=sc and tc;student_only+=sc and not tc;reference_only+=tc and not sc;both_wrong+=not sc and not tc
 results[name]={'reference_sidecar_sha256':digest,'both_correct':both_correct,'student_only_correct':student_only,'reference_only_correct':reference_only,'both_wrong':both_wrong}
wrong=[r for r in student.values() if r['prediction']!=r['target']]
result={'status':'passed','exit_code':0,'scope':'paired dev error audit, no new GPU run and no holdout access','student_sidecar_sha256':hashlib.sha256(sidecar.read_bytes()).hexdigest(),'samples':500,'student_errors':len(wrong),'errors_with_max_probability_ge_09':sum(max(r['probabilities'].values())>=.9 for r in wrong),'paired':results}
(root/'reports/s2-paired-image-audit-001.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
