"""Private exact original-source check for preserved Clotho reserve metadata."""
import hashlib,json,time
from pathlib import Path
from data.schema import Sample
from scripts.clotho_gold_pools import BuildManifest,bound_json,verify
from scripts.s0_audio_dev import load_license_plan,load_metadata,load_train_groups,candidate_rows,sample_for,FIXED_ARCHIVE_ALIASES
from scripts.gold_inventory import bound_lines
root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-gold-inventory-001';start=time.monotonic();report={'outcome':'rejected','exit_code':1,'scope':'preserved120 exact original train question and metadata mapping'}
try:
 p=private/'clotho-gold-001/build-manifest.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='8f89a79e41a8d99b0fb30368a51da4614930336baa0913cc04779644130638a2'
 m=BuildManifest.model_validate_json(p.read_bytes())
 for name in ('train','metadata','license_plan'):verify(getattr(m,name))
 approved=load_license_plan(Path(m.license_plan.path),metadata_sha256=m.metadata.sha256)
 metadata=load_metadata(Path(m.metadata.path),metadata_sha256=m.metadata.sha256,approved=approved)
 groups=load_train_groups(Path(m.train.path),train_sha256=m.train.sha256)
 rows,_,_=candidate_rows(groups=groups,metadata=metadata,audio_root=Path(m.audio_root),aliases=FIXED_ARCHIVE_ALIASES)
 originals={(str(r['sound_id']),str(r['question'])):sample_for(r) for r in rows}
 count=0
 for a in m.current_reserves:
  for line in bound_lines(a):
   actual=Sample.model_validate_json(line);expected=originals[(actual.meta.family_id,actual.questions[0].instruction)]
   left=actual.model_dump(mode='json');right=expected.model_dump(mode='json')
   # Frozen S3 allocation changes only pool/version; it must not change source semantics.
   for key in ('pool','data_version'):left['meta'].pop(key);right['meta'].pop(key)
   assert left==right;count+=1
 assert count==120
 report.update(outcome='original_mapping_verified',exit_code=0,preserved_items=count,source_artifact_sha256={n:getattr(m,n).sha256 for n in ('train','metadata','license_plan')},build_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
except Exception:report['failure']='preserved_source_mismatch'
report.update(elapsed_seconds=round(time.monotonic()-start,3),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (root/'reports/s4-clotho-preserved-source-001.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(report['exit_code'])
