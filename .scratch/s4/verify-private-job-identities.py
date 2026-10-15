"""Real Linux negative probes with numeric identities; emits aggregate evidence."""
import json,os,time
from pathlib import Path
from private_gold_job_d4c76d2 import JobRejected,check_process_ids,denied_probe,drop_identity,reserve_ids,unused_uid
root=Path('/var/tmp/omnijev-gold-jobs-001');report={};uid=62061
assert unused_uid(uid)
r,w=os.pipe();ready_r,ready_w=os.pipe();pid=os.fork()
if pid==0:
 try:
  os.close(w);os.close(ready_r);drop_identity(uid,uid);os.write(ready_w,b'1');os.close(ready_w);os.read(r,1);os._exit(0)
 except BaseException:os._exit(1)
os.close(r);os.close(ready_w)
try:
 assert os.read(ready_r,1)==b'1'
 try:check_process_ids({uid})
 except JobRejected as e:report['live_unregistered_identity_rejected']=e.category=='identity_unavailable'
 else:report['live_unregistered_identity_rejected']=False
finally:
 os.close(ready_r);os.close(w);os.waitpid(pid,0)
try:reserve_ids(root,62051,62052)
except JobRejected as e:report['used_identity_rejected']=e.category=='identity_unavailable'
else:report['used_identity_rejected']=False
runs=[p for p in root.iterdir() if p.is_dir() and p.name!='.identity-leases']
report['retained_runs_root_only']=all((p.stat().st_mode&0o777)==0o700 and p.stat().st_uid==p.stat().st_gid==0 for p in runs)
report['prior_workers_denied_retained_runs']=all(denied_probe(u,u,runs,[]) for u in (62041,62051))
report.update(scope='observed Linux ordinary-identity denial only; not root or network isolation',retained_run_count=len(runs),exit_code=0 if all(v for v in report.values()) else 1)
path=Path('/root/autodl-tmp/s1-text-lora-001/reports/s4-private-identity-negative-probes-001.json')
with path.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(report['exit_code'])
