"""Stop only this session's recorded validation processes; retain uploader/assets."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

root = Path('/root/autodl-tmp/s6-release-serving-001')
owned = []
for name, expected in [('final-vllm.pid', b'vllm'), ('final-gateway.pid', b'uvicorn')]:
    pid = int((root / name).read_text())
    proc = Path(f'/proc/{pid}/cmdline')
    if not proc.exists():
        continue
    cmd = proc.read_bytes()
    if expected not in cmd:
        raise RuntimeError(f'refusing unexpected process in {name}')
    owned.append(pid)
for pid in owned:
    os.kill(pid, signal.SIGTERM)
for _ in range(20):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True)
    if not result.stdout.strip():
        break
    time.sleep(1)
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, check=True)
report = {'status': 'safe_to_disable_gpu' if not result.stdout.strip() else 'still_in_use', 'stopped_validation_pids': owned, 'gpu_compute_processes': result.stdout.strip(), 'gpu_state': gpu.stdout.strip(), 'uploader_stopped': False, 'weights_deleted': False, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cost': None, 'cost_note': 'billing rate unavailable; no additional rental or purchase authorized'}
(root / 'artifacts/gpu-release-final-001.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
if report['status'] != 'safe_to_disable_gpu':
    raise SystemExit(1)
