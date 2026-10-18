#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/s6-release-serving-001
P=/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python
cd "$R/repo"
ready=0
for i in $(seq 1 90); do
  if curl --fail -s -m 2 http://127.0.0.1:8001/health >/dev/null; then ready=1; break; fi
  sleep 2
done
test "$ready" = 1
"$P" - <<'PY'
import json, time
from pathlib import Path
r=Path('/root/autodl-tmp/s6-release-serving-001')
d=json.loads((r/'artifacts/s6-vllm-server-record-007.json').read_text())
pid=int((r/'final-vllm.pid').read_text())
cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\x00',b' ').decode()
assert 'serving-view-bound-final' in cmd and 'config-bound-final' in cmd
d['launch_command']=cmd
d['observed_pid']=pid
d['observed_time_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
d['artifact_integrity_receipt']='flat-remap-recovery-001.json'
(r/'artifacts/s6-vllm-server-record-final-001.json').write_text(json.dumps(d,indent=2)+'\n')
PY
"$P" -m scripts.s0_parity compare-vllm \
 --manifest tests/fixtures/s0-prepare-public-001/manifest.json \
 --asset-root /root/autodl-tmp/lite-omnijev/artifacts/s0-prepare-public-001 \
 --model-dir /root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2 \
 --asset-manifest /root/autodl-tmp/lite-omnijev/artifacts/manifests/e4b.json \
 --adapter-dir /root/autodl-tmp/s1-text-lora-001/artifacts/s6-d3p-student-600-001/run/adapter \
 --dtype float32 \
 --reference "$R/artifacts/s6-serving-hf-reference-d3p600-001.json" \
 --server-record "$R/artifacts/s6-vllm-server-record-final-001.json" \
 --report "$R/artifacts/s6-serving-compare-final-bound-001.json"
echo FINAL_PARITY_EXIT:0
