"""Linux synthetic failure injection; no real gold payloads, no production hooks."""
import contextlib,io,json,os,time,signal
from pathlib import Path
import private_gold_job_d4c76d2 as job
from scripts.gold_inventory import digest
root=Path('/root/autodl-tmp/s1-text-lora-001');private=root/'artifacts/s4-private-failure-probes-002';private.mkdir(mode=0o700)
work=Path('/var/tmp/omnijev-gold-jobs-001');secret='SYNTHETIC_PRIVATE_SENTINEL_20261016';checks={};exports=[];started=time.monotonic()
def sample(name,pool):
 return {'id':name,'prefix':{'state':secret},'questions':[{'qid':'q','type':'noul','instruction':secret,'target':{'kind':'hard','value':'yes','source':'synthetic'}}],'meta':{'modality':['text'],'source_dataset':'synthetic','source_revision':'test','source_sha256':digest(b'synthetic'),'license':'MIT','usage':'research','pool':pool,'family_id':'same-family','data_version':'synthetic'}}
def manifest(name,groups):
 specs=[]
 for index,(pool,payload) in enumerate(groups):
  path=private/f'{name}-{index}.jsonl';path.write_bytes(payload);path.chmod(0o600)
  specs.append({'path':str(path),'sha256':digest(payload),'cohort':'text','exposure':'current','expected_pool':pool})
 path=private/f'{name}.json';path.write_text(json.dumps({'schema_version':'gold-inventory-manifest-v1','samples':specs,'component_ledgers':[]}));path.chmod(0o600);return path
def line(name,pool):return (json.dumps(sample(name,pool))+'\n').encode()
valid=manifest('valid',[('holdout',line('valid','holdout'))]);invalid=manifest('invalid',[('holdout',secret.encode())]);overlap=manifest('overlap',[('dev',line('a','dev')),('holdout',line('b','holdout'))])
for index,(name,path,expected) in enumerate([('valid',valid,'isolated'),('invalid',invalid,'invalid_sample'),('overlap',overlap,'pool_overlap')]):
 report=private/f'{name}-receipt.json';receipt=job.execute(path,digest(path.read_bytes()),work,report,62201+2*index,62202+2*index,30)
 exports.append(report.read_bytes());checks[name]=receipt.outcome=='isolated' if expected=='isolated' else expected in receipt.categories and receipt.outcome=='rejected'
# Actual CLI failure remains category-only, including secret path/argument.
stdout=io.StringIO();stderr=io.StringIO()
with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):code=job.main(['--private-'+secret,secret])
exports.extend([stdout.getvalue().encode(),stderr.getvalue().encode()]);checks['cli_redaction']=code==1
try:job.execute(valid,'0'*64,work,private/'bad-hash-receipt.json',62207,62208,30)
except job.JobRejected as e:checks['hash_rejection']=e.category=='input_rejected'
else:checks['hash_rejection']=False
before=(private/'valid-receipt.json').read_bytes()
try:job.execute(valid,digest(valid.read_bytes()),work,private/'valid-receipt.json',62209,62210,30)
except job.JobRejected as e:checks['overwrite_rejected']=e.category=='report_io' and before==(private/'valid-receipt.json').read_bytes()
else:checks['overwrite_rejected']=False
for name,payload in [('malformed',secret.encode()),('extra',json.dumps({'outcome':'rejected','categories':['invalid_sample'],'manifest_sha256':'0'*64,'private_label':secret}).encode())]:
 path=private/name;path.write_bytes(payload)
 try:job.validate_receipt(path,'0'*64,1)
 except job.JobRejected as e:checks[name+'_receipt_rejected']=e.category=='invalid_receipt'
 else:checks[name+'_receipt_rejected']=False
# Test-only fixed synthetic callback, never exposed by production CLI.
uid=62211;job.reserve_ids(work,uid,62212);run=work/'synthetic-log-failure-002';original=job.audit_manifest
with job.private_run(run,uid):
 output=run/'output';output.mkdir(mode=0o700);os.chown(output,uid,uid);run.chmod(0o750)
 def noisy_failure(_):
  os.write(1,secret.encode());os.write(2,secret.encode());raise RuntimeError(secret)
 job.audit_manifest=noisy_failure
 try:code,timed_out=job.run_child(valid,output,uid,5)
 finally:job.audit_manifest=original
 checks['private_log_captured']=secret.encode() in (output/'private.log').read_bytes()
 checks['exception_rejected']=code==2 and not timed_out and not (output/'receipt.json').exists()
checks['private_log_denied_after_run']=job.denied_probe(uid,uid,[run/'output/private.log'],[])
pid=os.fork()
if pid==0:
 os.setsid();signal.pause();os._exit(0)
code,timed_out=job.wait_child(pid,0.05)
try:os.waitpid(pid,os.WNOHANG)
except ChildProcessError:reaped=True
else:reaped=False
checks['timeout_reaped']=timed_out and code<0 and reaped
checks['exported_surfaces_redacted']=all(secret.encode() not in e for e in exports)
report={'scope':'real Linux synthetic failure/log/permission behavior, not gold or network security','checks':checks,'passed':all(checks.values()),'code_sha256':digest(job.code_bytes()),'runner_sha256':digest(Path(__file__).read_bytes()),'elapsed_seconds':round(time.monotonic()-started,3),'exit_code':int(not all(checks.values()))}
with (root/'reports/s4-private-failure-probes-002.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report));raise SystemExit(report['exit_code'])
