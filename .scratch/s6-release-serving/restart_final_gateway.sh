#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/s6-release-serving-001
"$R/envs/serve/bin/python" - <<'PY'
import os, signal
from pathlib import Path
p = Path('/root/autodl-tmp/s6-release-serving-001/final-gateway.pid')
pid = int(p.read_text())
cmd = Path(f'/proc/{pid}/cmdline').read_bytes()
if b'uvicorn' not in cmd or b'serve.gateway.app:app' not in cmd:
    raise RuntimeError('refusing to stop unexpected process')
os.kill(pid, signal.SIGTERM)
PY
sleep 2
B=/root/autodl-tmp/lite-omnijev/artifacts/models/google__gemma-4-E4B-it/ee0ef6023621cff504d758262d4e04895a5af4a2
GATEWAY_PYTHON="$R/envs/serve/bin/python" REPO="$R/repo" LITE_MODEL_DIR="$B" LITE_MEDIA_ROOT=/root/autodl-tmp/lite-omnijev/artifacts/s0-prepare-public-001 \
  nohup bash "$R/repo/serve/deploy/launch-gateway.sh" > "$R/logs/gateway-final-flat.log" 2>&1 < /dev/null &
echo $! > "$R/final-gateway.pid"
