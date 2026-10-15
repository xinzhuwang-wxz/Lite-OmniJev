"""Observe the local derived-model server and run the frozen public100 comparison."""

import hashlib
import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

root = Path("/root/autodl-tmp/s1-text-lora-001")
reference = root / "reports/s3-serving-hf1120-reference-001.json"
ref = json.loads(reference.read_text())
assert ref["status"] == "reference-ready"
expected = str(root / "artifacts/s3-merged-fp32-reference-001")
for _ in range(60):
    try:
        with urllib.request.urlopen("http://127.0.0.1:18001/health", timeout=2) as response:
            if response.status == 200:
                break
    except (OSError, urllib.error.URLError):
        time.sleep(3)
else:
    raise RuntimeError("local_server_not_ready")
with urllib.request.urlopen("http://127.0.0.1:18001/v1/models", timeout=5) as response:
    models = json.load(response)
assert any(m["id"] == "s3-decision" for m in models["data"])
processes = []
for path in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        args = [x.decode() for x in path.read_bytes().split(b"\0") if x]
    except (OSError, UnicodeError):
        continue
    if (
        expected in args
        and any(x.endswith("/bin/vllm") for x in args)
        and not args[0].endswith("timeout")
    ):
        processes.append((int(path.parent.name), args))
assert len(processes) == 1
pid, args = processes[0]
assert args[args.index("--dtype") + 1] == "float32"
assert args[args.index("--logprobs-mode") + 1] == "raw_logprobs"
assert args[args.index("--tokenizer") + 1] == ref["provenance"]["model_dir"]
assert "--enable-lora" not in args
record = {
    "provenance": ref["provenance"],
    "served_model": "s3-decision",
    "dtype": "float32",
    "logprobs_mode": "raw_logprobs",
    "operator_attested": True,
    "raw_logprobs_confirmed": True,
    "launch_command": shlex.join(args),
    "pid": pid,
    "scope": "Observed local derived-weight server, not external attestation or release acceptance",
    "adapter_materialization": {
        name: hashlib.sha256((root / "reports" / name).read_bytes()).hexdigest()
        for name in ["s3-merged-fp32-materialization-001.json", "s3-merged-alias-repair-001.json"]
    },
}
server = root / "reports/s3-serving-merged-server-001.json"
assert not server.exists()
server.write_text(json.dumps(record, indent=2) + "\n")
cmd = [
    "/root/autodl-tmp/envs/before-gpu-teacher-job-final-001/bin/python",
    "-u",
    "-m",
    "scripts.s0_parity",
    "compare-vllm",
    "--manifest",
    "tests/fixtures/s0-prepare-public-001/manifest.json",
    "--asset-root",
    ref["provenance"]["asset_root"],
    "--model-dir",
    ref["provenance"]["model_dir"],
    "--asset-manifest",
    ref["provenance"]["asset_manifest"]["path"],
    "--adapter-dir",
    str(root / "artifacts/s3-stable-probe-train-001/adapter"),
    "--dtype",
    "float32",
    "--image-max-soft-tokens",
    "1120",
    "--reference",
    str(reference),
    "--server-record",
    str(server),
    "--upstream-url",
    "http://127.0.0.1:18001",
    "--report",
    "reports/s3-serving-merged-parity-001.json",
]
result = subprocess.run(cmd, cwd=root, timeout=600, check=False)
(root / "artifacts/s3-serving-reference-001/compare-exit-code.txt").write_text(
    str(result.returncode) + "\n"
)
raise SystemExit(result.returncode)
