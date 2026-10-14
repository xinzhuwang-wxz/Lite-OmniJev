"""Actual Chromium/official CPU media gateway checks with an explicitly fake backend."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import runpy
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI
from playwright.sync_api import sync_playwright

from scripts.s0_prepare import read_manifest
from serve.gateway.app import Settings, create_app

ROOT = Path(__file__).resolve().parents[1]
VERSION = "cpu-fixture-not-real-model"


def cpu_app() -> FastAPI:
    """Development-only fixture; it is never the production module's app factory."""
    if os.environ.get("S0_BROWSER_UNCONFIGURED") == "1":
        return create_app(Settings())
    processor = importlib.import_module("transformers").AutoProcessor.from_pretrained(
        os.environ["S0_LOCAL_MODEL_DIR"], local_files_only=True, trust_remote_code=False
    )
    fixture = runpy.run_path(str(ROOT / "tests/test_gateway_media.py"))["NativeChatFixture"]
    return create_app(
        Settings(
            upstream_url="http://explicit-cpu-fixture",
            upstream_model="pinned-fixture-model",
            model_version=VERSION,
            media_root=Path(os.environ["S0_CANONICAL_MEDIA_ROOT"]),
            raw_logprobs_confirmed=True,
        ),
        processor=processor,
        transport=httpx.MockTransport(fixture(processor)),
    )


@contextmanager
def server(output: Path, *, unconfigured: bool = False) -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "S0_BROWSER_UNCONFIGURED": "1" if unconfigured else "0",
    }
    with (output / ("unconfigured.log" if unconfigured else "canonical.log")).open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "scripts.browser_s0_cpu:cpu_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-access-log",
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("CPU fixture server exited")
                try:
                    if httpx.get(url + "/health", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("CPU fixture startup timed out")
            yield url
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url", help="Optional existing loopback gateway; never starts a model"
    )
    parser.add_argument("--model-version", help="Required expected declaration for a live gateway")
    args = parser.parse_args()
    if args.base_url:
        parsed = urlsplit(args.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or not args.model_version
        ):
            parser.error(
                "live mode requires an existing loopback gateway and expected model version"
            )
    expected_version = args.model_version if args.base_url else VERSION
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = {
        "scope": "Actual Chromium + actual gateway + official CPU processor; fake native backend",
        "status": "blocked",
        "exit_code": 1,
        "gpu_used": None if args.base_url else False,
        "real_model_forward": "not_attested_by_browser" if args.base_url else False,
        "release_or_stage_acceptance": False,
        "cases": [],
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "cost_usd": 0.0 if not args.base_url else None,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "serve/gateway/app.py",
                ROOT / "serve/vllm_chat.py",
                ROOT / "train/multimodal.py",
                ROOT / "tests/test_gateway_media.py",
            )
        },
    }
    if args.base_url:
        report["scope"] = (
            "Actual Chromium against existing gateway; model/quality provenance separate"
        )
    try:
        cases = read_manifest(args.manifest)
        questions = runpy.run_path(str(ROOT / "tests/test_gateway_media.py"))["QUESTIONS"]
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_context(viewport={"width": 1280, "height": 960})
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda _: errors.append("browser_page_error"))
            try:
                with (
                    nullcontext(args.base_url) if args.base_url else server(args.output_dir)
                ) as url:
                    page.goto(url + "/docs", wait_until="networkidle")
                    page.get_by_text(
                        "Lite-OmniJev local text gateway", exact=False
                    ).first.wait_for()
                    for index in (0, 20, 40, 60, 80, 90):
                        case = cases[index]
                        body = {
                            "state": case.prefix.state,
                            "media": [media.model_dump() for media in case.prefix.media],
                            "video_frames": case.video_frames,
                            "questions": questions,
                        }
                        result = post(page, body)
                        assert (
                            result["status"] == 200 and result["body"]["model"] == expected_version
                        )
                        for name, answer in result["body"]["answers"].items():
                            values = list(answer["probabilities"].values())
                            assert all(type(v) in (int, float) and math.isfinite(v) for v in values)
                            assert abs(sum(values) - 1) < 1e-6 and answer["type"] == name
                            assert 0 <= answer["p_insufficient"] <= 1
                        report["cases"].append(
                            {
                                "category": case.category,
                                "frames": case.video_frames,
                                "status": result["status"],
                                "server_timing": result["timing"],
                            }
                        )
                    negative = post(
                        page,
                        {
                            "state": "synthetic privacy sentinel",
                            "media": [
                                {"type": "image", "uri": "../privacy-sentinel", "sha256": "0" * 64}
                            ],
                            "questions": questions,
                        },
                    )
                    assert negative["status"] == 422
                    assert "privacy" not in json.dumps(negative["body"]).lower()
                    report["bad_media_status"] = negative["status"]
                    page.screenshot(path=str(args.output_dir / "docs.png"), full_page=True)
                if not args.base_url:
                    with server(args.output_dir, unconfigured=True) as url:
                        page.goto(url + "/docs", wait_until="networkidle")
                        result = post(page, {"state": "public", "questions": questions})
                        assert result["status"] == 503 and "answers" not in result["body"]
                        report["unconfigured_status"] = result["status"]
                assert not errors
                report.update(
                    status="passed_live_browser_contract_only"
                    if args.base_url
                    else "passed_cpu_browser_only",
                    exit_code=0,
                    browser_version=browser.version,
                    page_errors=errors,
                )
            finally:
                context.tracing.stop(path=str(args.output_dir / "trace.zip"))
                browser.close()
    except Exception as error:
        report["error_type"] = type(error).__name__
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "exit_code")}))
    return int(report["exit_code"])


def post(page: Any, body: Any) -> dict[str, Any]:
    result: dict[str, Any] = page.evaluate(
        """async (body) => {
      const response = await fetch('/v1/systemone', {
        method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
      return {status:response.status, body:await response.json(),
              timing:response.headers.get('Server-Timing')};
    }""",
        body,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
