"""Freeze repository-owned source mapping dependency hashes from executed revisions."""
import ast,hashlib,json,subprocess
from pathlib import Path
jobs=[('text','c7cdb21','s4-gold-code-c7cdb21',['scripts/admit_clinc.py','scripts/s0_text_dev.py','scripts/s1_text_pools.py'],{}),('gui','5096382','s4-gold-code-5096382',['scripts/gui_gold_pools.py'],{}),('clotho','1f02a8a','s4-audio-code-1f02a8a',['scripts/clotho_gold_pools.py'],{}),('speech','1f02a8a','s4-audio-code-1f02a8a',['scripts/librispeech_gold_pools.py'],{}),('video','f1ad912','s4-gold-code-f1ad912',['scripts/clevrer_gold_pools.py'],{}),('vqa','c7cdb21','s4-gold-code-c7cdb21',['scripts/vqa2_gold_pools.py'],{}),('crossmodal','c7cdb21','s4-gold-code-c7cdb21',['scripts/crossmodal_gold_pools.py'],{}),('crossmodal_dev','c7cdb21','s4-gold-code-c7cdb21',['scripts/crossmodal_gold_pools.py'],{'scripts/crossmodal_gold_pools.py':('04a5239','/root/autodl-tmp/crossmodal_gold_pools_04a5239.py')})]
records=[]
for name,commit,directory,entries,overrides in jobs:
 available=set(subprocess.check_output(['git','ls-tree','-r','--name-only',commit],text=True).splitlines());pending=list(entries);done={}
 while pending:
  path=pending.pop()
  if path in done or path not in available:continue
  revision,remote=overrides.get(path,(commit,'/root/autodl-tmp/'+directory+'/'+path))
  raw=subprocess.check_output(['git','show',revision+':'+path]);done[path]={'path':remote,'sha256':hashlib.sha256(raw).hexdigest(),'revision':revision}
  tree=ast.parse(raw);modules=[]
  for n in ast.walk(tree):
   if isinstance(n,ast.Import):modules.extend(a.name for a in n.names)
   elif isinstance(n,ast.ImportFrom) and n.module:
    modules.append(n.module);modules.extend(n.module+'.'+a.name for a in n.names)
   elif isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):
    if (isinstance(n.func,ast.Name) and n.func.id=='__import__') or (isinstance(n.func,ast.Attribute) and n.func.attr=='import_module'):modules.append(n.args[0].value)
  for m in modules:
   if m.split('.')[0] not in ('scripts','data','eval','train','serve'):continue
   stem=m.replace('.','/');pending.extend([stem+'.py',stem+'/__init__.py'])
   parts=stem.split('/')
   for i in range(1,len(parts)):pending.append('/'.join(parts[:i])+'/__init__.py')
 records.append({'source':name,'execution_revision':commit,'files':done})
p=Path('.scratch/s4/source-code-closure-001.json');p.write_text(json.dumps({'scope':'repository-owned statically imported source mapping code plus literal dynamic imports; external dependencies remain runtime evidence','jobs':records},indent=2)+'\n')
print(json.dumps({'jobs':len(records),'file_bindings':sum(len(j['files']) for j in records),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
