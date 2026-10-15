"""Fetch only missing pinned annotations for already cached GUI media."""
import hashlib,json,time
from pathlib import Path
from scripts import s0_gui_dev as gui
start=time.monotonic()
root=Path('/root/autodl-tmp/s1-text-lora-001')
config=json.loads((root/'artifacts/s4-gold-inventory-001/visual-capacity-config-001.json').read_bytes())
path=Path(config['plan']['path']); content=path.read_bytes()
assert hashlib.sha256(content).hexdigest()==config['plan']['sha256']
plan=json.loads(content); annotation_root=Path(config['annotation_root'])
missing=[gui.safe_episode_id(row['episode_id']) for row in plan['selected_episodes'] if not (annotation_root/(row['episode_id']+'.json')).exists()]
assert len(missing)<=200
receipts=gui.ensure_annotations(missing,annotation_root,download_missing=True)
counts={state:sum(r['result']==state for r in receipts) for state in ('downloaded','present','failed')}
private=root/'artifacts/s4-gold-inventory-001/gui-annotation-download-001.json'
with private.open('x') as h: json.dump(receipts,h)
report={'scope':'missing pinned annotation metadata for existing cached media','source_revision':gui.REVISION,'requested':len(missing),'counts':counts,'downloaded_bytes':sum(r.get('bytes',0) for r in receipts),'receipt_sha256':hashlib.sha256(private.read_bytes()).hexdigest(),'plan_sha256':config['plan']['sha256'],'elapsed_seconds':round(time.monotonic()-start,3),'exit_code':int(bool(counts['failed'])),'pool_assignment_written':False}
with (root/'reports/s4-gui-annotation-download-001.json').open('x') as h: json.dump(report,h,indent=2)
print(json.dumps(report))
