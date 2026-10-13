"""Historical run-002 command evidence; use a new report ID before repeating.

Reads a project-external private key file. Contains no credential or signed URL.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.mdc_download_dataset import DownloadError, read_secret, stream_download

report = Path("reports/mdc-commonvoice-download-002.json")
result = {
    "status": "running",
    "transport": "official curl POST flow; API key stdin config only",
    "credential_logged": False,
    "signed_url_logged": False,
    "training_admitted": False,
    "gpu_used": False,
}
report.write_text(json.dumps(result, indent=2) + "\n")
private_path = None
try:
    secret = read_secret(Path("/root/.config/lite-omnijev/mdc.key"))
    with tempfile.NamedTemporaryFile(
        dir="/root/.config/lite-omnijev", prefix=".mdc-api-", delete=False
    ) as handle:
        private_path = Path(handle.name)
    config = (
        'header = "Authorization: Bearer '
        + secret
        + '"\nheader = "Content-Type: application/json"\n'
    )
    secret = ""
    completed = subprocess.run(
        [
            "curl",
            "--silent",
            "--show-error",
            "--config",
            "-",
            "--request",
            "POST",
            "--proto",
            "=https",
            "--max-redirs",
            "0",
            "--retry",
            "0",
            "--connect-timeout",
            "15",
            "--max-time",
            "45",
            "--max-filesize",
            "65536",
            "--output",
            str(private_path),
            "--write-out",
            "%{http_code}",
            "https://mozilladatacollective.com/api/datasets/cmu5nqn1h00vwmi07b4dbk085/download",
        ],
        input=config,
        text=True,
        capture_output=True,
    )
    config = ""
    result["curl_exit_code"] = completed.returncode
    status = completed.stdout.strip()
    result["api_http_status"] = int(status) if status.isdigit() else None
    if completed.returncode or status != "200":
        raise DownloadError("mdc_curl_api_failed")
    with private_path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise DownloadError("api_metadata_too_large")
    payload = json.loads(raw)
    raw = b""
    if not isinstance(payload, dict) or not isinstance(payload.get("downloadUrl"), str):
        raise DownloadError("api_metadata_invalid")
    url = payload["downloadUrl"]
    payload = {}
    private_path.unlink()
    private_path = None
    output = Path("artifacts/data/quarantine/commonvoice-spontaneous-5-en/archive.tar.gz")
    size, sha = stream_download(
        url,
        output,
        expected_size_bytes=544267165,
        expected_sha256="390b5a54c9afe0cc01da039ad206248f85682f247dd2b27d4cc0ab9a68e860b6",
    )
    url = ""
    result.update(
        status="downloaded_hash_verified", bytes=size, sha256=sha, output=str(output), exit_code=0
    )
except DownloadError as exc:
    result.update(status="failed", error_code=str(exc), exit_code=1)
except Exception as exc:
    result.update(status="failed", error_type=type(exc).__name__, exit_code=1)
finally:
    if private_path is not None:
        private_path.unlink(missing_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
sys.exit(result["exit_code"])
