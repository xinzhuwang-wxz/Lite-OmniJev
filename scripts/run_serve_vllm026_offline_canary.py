import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path('/root/autodl-tmp/lite-omnijev')
WHEELHOUSE = ROOT / 'artifacts/runtime-builds/python-wheels/serve-vllm026'
REQ = WHEELHOUSE / 'requirements.txt'
ENV = pathlib.Path('/root/autodl-tmp/envs/serve-vllm026-offline-check')
LOG = ROOT / 'logs/serve-vllm026-offline-install.log'
REPORT = ROOT / 'reports/serve-vllm026-offline-import.json'
LOCK = pathlib.Path('/tmp/omnijev-cpu-heavy.lock')


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, **kwargs)


def module_probe(python: pathlib.Path):
    code = r'''
import importlib, json, sys
out = {"python": sys.version, "imports": {}}
for name in ["torch", "transformers", "vllm"]:
    try:
        mod = importlib.import_module(name)
        out["imports"][name] = {"status": "ok", "version": getattr(mod, "__version__", None), "file": getattr(mod, "__file__", None)}
    except Exception as exc:
        out["imports"][name] = {"status": "error", "type": type(exc).__name__, "message": str(exc)}
try:
    import torch
    out["torch_probe"] = {
        "version": torch.__version__,
        "file": torch.__file__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_sum": float(torch.ones(2, dtype=torch.float32).sum().item()),
    }
except Exception as exc:
    out["torch_probe"] = {"status": "error", "type": type(exc).__name__, "message": str(exc)}
print(json.dumps(out, sort_keys=True))
'''
    result = run([str(python), '-c', code])
    payload = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    try:
        payload["parsed"] = json.loads(result.stdout)
    except Exception:
        pass
    return payload


def main() -> int:
    started = time.time()
    files = sorted(p for p in WHEELHOUSE.iterdir() if p.is_file())
    manifest = {
        "captured_at": time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        "schema": "lite-omnijev.serve-vllm026-offline-import.v1",
        "wheelhouse": str(WHEELHOUSE),
        "requirements": str(REQ),
        "requirements_sha256": sha(REQ),
        "file_count": len(files),
        "contains": {
            "vllm-0.26.0": any(p.name.startswith('vllm-0.26.0') for p in files),
            "torch-2.11.0": any(p.name.startswith('torch-2.11.0') for p in files),
            "triton-3.6.0": any(p.name.startswith('triton-3.6.0') for p in files),
        },
        "env": str(ENV),
        "log": str(LOG),
        "not_claimed": ["No GPU execution", "No vLLM server startup", "No model forward"],
    }
    if ENV.exists():
        shutil.rmtree(ENV)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    create = run([sys.executable, '-m', 'venv', str(ENV)])
    manifest['venv_create'] = {"returncode": create.returncode, "stderr": create.stderr[-4000:]}
    if create.returncode != 0:
        REPORT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        return create.returncode
    py = ENV / 'bin/python'
    command = [
        str(py), '-m', 'pip', 'install',
        '--no-index', '--find-links', str(WHEELHOUSE), '--require-hashes',
        '-r', str(REQ),
    ]
    with LOG.open('w', encoding='utf-8') as log:
        install = subprocess.run(command, text=True, stdout=log, stderr=subprocess.STDOUT)
    manifest['install_returncode'] = install.returncode
    manifest['log_sha256'] = sha(LOG)
    if install.returncode == 0:
        probe = module_probe(py)
        manifest.update(probe.get('parsed', {}))
        manifest['probe_returncode'] = probe['returncode']
        if probe['returncode'] != 0:
            manifest['probe_stdout'] = probe['stdout'][-4000:]
            manifest['probe_stderr'] = probe['stderr'][-4000:]
    manifest['elapsed_sec'] = round(time.time() - started, 1)
    REPORT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return install.returncode or manifest.get('probe_returncode', 0)

if __name__ == '__main__':
    raise SystemExit(main())
