import hashlib
import importlib
import importlib.metadata as md
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path('/root/autodl-tmp/lite-omnijev')
BASE = ROOT / 'artifacts/runtime-builds/python-wheels/serve-next-vllm031'
PYARROW = ROOT / 'artifacts/runtime-builds/python-wheels/data-pyarrow'
ENV = pathlib.Path('/root/autodl-tmp/envs/before-gpu-teacher-runtime-offline-check')
LOG = ROOT / 'logs/before-gpu-teacher-runtime-offline-install.log'
REPORT = ROOT / 'reports/before-gpu-teacher-runtime-canary.json'
SRC = ROOT / 'artifacts/before-gpu-teacher-runtime-source/main-08709d9'

def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def run(command, **kwargs):
    return subprocess.run(command, text=True, capture_output=True, **kwargs)

def canary(py: pathlib.Path) -> dict:
    code = r'''
import importlib, importlib.metadata as md, json, os, pathlib, sys, tempfile
source = pathlib.Path(os.environ['TEACHER_RUNTIME_SOURCE'])
sys.path.insert(0, str(source))
mods = ['torch','transformers','safetensors','PIL.Image','numpy','pyarrow','pyarrow.parquet','cv2','torchvision','scripts.s0_teacher_eval','eval.teacher_benchmarks','train.teacher_inference','train.teacher_media']
out = {'imports': {}}
for mod in mods:
    try:
        m = importlib.import_module(mod)
        out['imports'][mod] = {'status': 'ok', 'file': getattr(m, '__file__', None)}
    except Exception as exc:
        out['imports'][mod] = {'status': 'error', 'type': type(exc).__name__, 'message': str(exc)[:500]}
for dist in ['torch','transformers','safetensors','Pillow','numpy','pyarrow','opencv-python-headless','torchvision','vllm']:
    try:
        out.setdefault('versions', {})[dist] = md.version(dist)
    except Exception as exc:
        out.setdefault('versions', {})[dist] = {'error': type(exc).__name__}
import torch
out['torch_probe'] = {'cpu_sum': float(torch.ones(2).sum().item()), 'cuda_available': bool(torch.cuda.is_available()), 'version': torch.__version__}
from PIL import Image
import numpy as np
img = Image.new('RGB', (2, 2), color=(10, 20, 30))
out['pil_numpy_probe'] = {'shape': list(np.asarray(img).shape), 'sum': int(np.asarray(img).sum())}
import torchvision.transforms.functional as F
out['torchvision_probe'] = {'tensor_shape': list(F.pil_to_tensor(img).shape)}
import cv2
out['cv2_probe'] = {'version': cv2.__version__}
import pyarrow as pa
import pyarrow.parquet as pq
with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / 'tiny.parquet'
    table = pa.table({'x': [1, 2, 3]})
    pq.write_table(table, p)
    out['pyarrow_probe'] = {'rows': pq.read_table(p).num_rows, 'columns': pq.read_table(p).num_columns}
print(json.dumps(out, sort_keys=True))
'''
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': '',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        'TEACHER_RUNTIME_SOURCE': str(SRC),
    })
    proc = run([str(py), '-c', code], env=env)
    result = {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    try:
        result['parsed'] = json.loads(proc.stdout)
    except Exception:
        pass
    return result

def main() -> int:
    started = time.time()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'captured_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'schema': 'lite-omnijev.before-gpu-teacher-runtime-canary.v1',
        'selected_runtime': 'serve-next-vllm031 plus data-pyarrow overlay',
        'source_snapshot': str(SRC),
        'env': str(ENV),
        'base_wheelhouse': str(BASE),
        'pyarrow_wheelhouse': str(PYARROW),
        'base_requirements_sha256': sha(BASE / 'requirements.txt'),
        'pyarrow_requirements_sha256': sha(PYARROW / 'requirements.txt'),
        'install_log': str(LOG),
        'install_command': [str(ENV / 'bin/python'), '-m', 'pip', 'install', '--no-index', '--find-links', str(BASE), '--find-links', str(PYARROW), '--require-hashes', '-r', str(BASE / 'requirements.txt'), '-r', str(PYARROW / 'requirements.txt')],
        'not_claimed': ['No GPU execution', 'No model forward', 'No processor AutoProcessor load', 'No vLLM server startup'],
    }
    if ENV.exists():
        shutil.rmtree(ENV)
    create = run([sys.executable, '-m', 'venv', str(ENV)])
    report['venv_create'] = {'returncode': create.returncode, 'stderr': create.stderr[-4000:]}
    if create.returncode != 0:
        REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        return create.returncode
    with LOG.open('w', encoding='utf-8') as log:
        install = subprocess.run(report['install_command'], text=True, stdout=log, stderr=subprocess.STDOUT)
    report['install_returncode'] = install.returncode
    report['install_log_sha256'] = sha(LOG)
    if install.returncode == 0:
        probe = canary(ENV / 'bin/python')
        report['probe_returncode'] = probe['returncode']
        if 'parsed' in probe:
            report['probe'] = probe['parsed']
        else:
            report['probe_stdout_tail'] = probe['stdout'][-4000:]
            report['probe_stderr_tail'] = probe['stderr'][-4000:]
    report['elapsed_sec'] = round(time.time() - started, 1)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report.get('install_returncode', 1) or report.get('probe_returncode', 0)

if __name__ == '__main__':
    raise SystemExit(main())
