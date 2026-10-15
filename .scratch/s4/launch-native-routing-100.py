"""Bounded multimodal native server lifecycle, after the HF process terminates."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

root = Path('/root/autodl-tmp/s1-text-lora-001')
source = Path('/root/autodl-tmp/s4-routing-code-d5c3661')
train_python = '/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python'
service = Path('/root/autodl-tmp/envs/serve-vllm026-offline-check')
model = '/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2'
out = root / 'artifacts/s4-native-routing-server-100-001'
out.mkdir(exist_ok=False)
hf = json.loads((root / 'reports/s4-native-routing-hf-100-001.json').read_text())
assert hf['status'] == 'completed' and len(hf['rows']) == 100
argv = [str(service / 'bin/vllm'), 'serve', model, '--host', '127.0.0.1', '--port', '8014', '--served-model-name', 's4-base', '--dtype', 'bfloat16', '--max-model-len', '8192', '--max-num-seqs', '1', '--max-num-batched-tokens', '8192', '--gpu-memory-utilization', '0.70', '--kv-cache-memory-bytes', '4294967296', '--seed', '42', '--enforce-eager', '--generation-config', 'vllm', '--logprobs-mode', 'raw_logprobs', '--max-logprobs', '32', '--enable-lora', '--max-lora-rank', '16', '--max-loras', '2', '--limit-mm-per-prompt', '{"image":1,"audio":1,"video":1}', '--skip-mm-profiling', '--mm-processor-cache-gb', '0', '--attention-backend', 'TRITON_ATTN', '--no-enable-prefix-caching', '--no-async-scheduling', '--no-enable-chunked-prefill', '--no-enable-log-requests', '--lora-modules', f's4-general={root}/artifacts/s4-joint-pilot-train-001/adapter', f's4-video={root}/artifacts/s4-video-pilot-train-001/adapter']
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(source))
env['PATH'] = str(service/'bin') + ':' + env['PATH']
report = {'argv':argv, 'quality_passed':False, 'ready':False}
start=time.monotonic()
with (out/'server.log').open('w') as log:
    proc=subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    report['server_pid']=proc.pid
    try:
        while proc.poll() is None and time.monotonic()-start<240:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8014/v1/models',timeout=2) as response:
                    report['models']=json.load(response)
                report['ready']=True
                break
            except Exception:
                time.sleep(2)
        if report['ready']:
            command=[train_python,'/root/autodl-tmp/native-routing-multimodal-100-001.py','vllm','--per-cohort','25','--prepared',str(root/'reports/s4-native-routing-prepare-100-001.json'),'--output',str(root/'reports/s4-native-routing-vllm-100-001.json'),'--url','http://127.0.0.1:8014']
            report['client_argv']=command
            with (out/'client.log').open('w') as clientlog:
                result=subprocess.run(command,cwd=source,env=env,stdout=clientlog,stderr=subprocess.STDOUT,timeout=300)
            report['client_exit_code']=result.returncode
    except Exception as error:
        report['error']=str(error)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
        report['server_exit_code']=proc.returncode
        report['wall_seconds']=time.monotonic()-start
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
