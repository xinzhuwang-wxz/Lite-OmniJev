"""Bounded native vLLM startup probe; no quality or parity claim."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

root = Path('/root/autodl-tmp/s1-text-lora-001')
out = root / 'artifacts/s4-native-multilora-probe-002'
out.mkdir(exist_ok=False)
envroot = Path('/root/autodl-tmp/envs/serve-vllm026-offline-check')
model = '/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2'
adapters = {
    'general': root / 'artifacts/s4-joint-pilot-train-001/adapter',
    'video': root / 'artifacts/s4-video-pilot-train-001/adapter',
}
expected = {'general': '6bb86b09028aa06f1733f54c7db665a8845126d8ba623be44c8d92bcfd8c96ac', 'video': '14496e283908180303e6d48d0983dbd4c574b4f35f7bb350acf59d9c2a176c9c'}
for name, path in adapters.items():
    assert hashlib.sha256((path / 'adapter_model.safetensors').read_bytes()).hexdigest() == expected[name]
argv = [str(envroot / 'bin/vllm'), 'serve', model, '--host', '127.0.0.1', '--port', '8014', '--served-model-name', 's4-base', '--dtype', 'bfloat16', '--max-model-len', '4096', '--max-num-seqs', '1', '--gpu-memory-utilization', '0.70', '--seed', '42', '--enforce-eager', '--generation-config', 'vllm', '--logprobs-mode', 'raw_logprobs', '--max-logprobs', '32', '--enable-lora', '--max-lora-rank', '16', '--max-loras', '2', '--limit-mm-per-prompt', '{"image":0,"audio":0,"video":0}', '--skip-mm-profiling', '--lora-modules', *[f'{name}={path}' for name,path in adapters.items()]]
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
env['PATH'] = str(envroot / 'bin') + ':' + env['PATH']
report = {'argv': argv, 'adapter_sha256': expected, 'scope': 'native_two_adapter_text_only_startup', 'quality_passed': False, 'parity_passed': False}
start = time.monotonic()
with (out / 'server.log').open('w') as log:
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while time.monotonic() - start < 240 and proc.poll() is None:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8014/v1/models', timeout=2) as response:
                    report['models'] = json.load(response)
                report['ready'] = True
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
                payload = json.dumps({'state': 'The red box contains two apples.', 'question': 'Does the red box contain apples?', 'answers': {'A': 'yes', 'B': 'no'}, 'U': 'Insufficient evidence to decide', 'response': 'Return exactly one answer token, or U if evidence is insufficient.'}, separators=(',', ':'))
                prompt = tok.apply_chat_template([{'role': 'user', 'content': payload}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                prefix_ids = tok.encode(prompt, add_special_tokens=False)
                ids = []
                for label in ['A', 'B', 'U']:
                    combined = tok.encode(prompt + label, add_special_tokens=False)
                    assert combined[:-1] == prefix_ids
                    ids.append(combined[-1])
                assert len(set(ids)) == 3
                report['answer_token_ids'] = ids
                report['requests'] = []
                for adapter in ['general', 'video', 'general', 'video']:
                    body = {'model': adapter, 'prompt': prompt, 'temperature': 1, 'max_tokens': 1, 'logprobs': 1, 'logprob_token_ids': ids, 'return_tokens_as_token_ids': True, 'seed': 42}
                    req = urllib.request.Request('http://127.0.0.1:8014/v1/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    t = time.monotonic()
                    try:
                        with urllib.request.urlopen(req, timeout=40) as response:
                            result = json.load(response)
                        report['requests'].append({'adapter': adapter, 'seconds': time.monotonic()-t, 'response': result})
                    except Exception as error:
                        report['requests'].append({'adapter': adapter, 'error': str(error)})
                        break
                break
            except Exception as error:
                if report.get('ready'):
                    report['probe_error'] = str(error)
                    break
                time.sleep(2)
        report.setdefault('ready', False)
        report['exit_before_cleanup'] = proc.poll()
    finally:
        import signal
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        report['exit_after_cleanup'] = proc.returncode
        report['wall_seconds'] = time.monotonic() - start
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
