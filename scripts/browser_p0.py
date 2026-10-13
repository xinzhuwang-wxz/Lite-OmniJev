"""Real Chromium contract acceptance against an already configured local P0 gateway."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import Page, expect, sync_playwright


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def check_answer(answer: dict[str, Any], kind: str, labels: set[str]) -> None:
    require(answer["type"] == kind, "answer primitive changed")
    probabilities = answer["probabilities"]
    require(set(probabilities) == labels, "answer labels differ or U entered distribution")
    require(all(probability(value) for value in probabilities.values()), "invalid probabilities")
    require(abs(sum(probabilities.values()) - 1) < 1e-6, "non-U distribution not normalized")
    require(probability(answer["p_insufficient"]), "p(U) must be finite and bounded")
    require(probability(answer["confidence"]), "confidence must be finite and bounded")
    require(answer["calibration"] == "unfitted", "P0 must disclose unfitted calibration")
    require(answer["confidence_semantics"] == "normalized_peak_non_U", "confidence semantics")
    if kind == "choice":
        require(answer["choice"] in labels, "unknown choice")
    elif kind == "noul":
        require(answer["noul"] == probabilities["yes"], "noul must equal p(yes)")
    else:
        score = answer["score"]
        require(type(score) in (int, float) and math.isfinite(score), "invalid score")
        require(0 <= score <= len(labels) - 1, "score outside scale")
        require(set(answer["legend"]) == labels, "score legend differs")


def post(page: Page, body: Any, *, malformed: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = page.evaluate(
        """async ({body, malformed}) => {
          const start = performance.now();
          const response = await fetch('/v1/systemone', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: malformed ? body : JSON.stringify(body),
            signal: AbortSignal.timeout(120000)
          });
          return {status: response.status, body: await response.json(),
                  elapsed_ms: performance.now() - start};
        }""",
        {"body": body, "malformed": malformed},
    )
    return result


def run(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report: dict[str, Any] = {
        "time": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "expected_model_version": args.expected_model_version,
        "tool": "Python Playwright actual Chromium (MCP unavailable)",
        "playwright_version": version("playwright"),
        "environment": platform.platform(),
        "code_version": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "P0 text gateway contract; no accuracy, calibration, latency or parity gate claim",
        "model_quality_verified": False,
        "health_executed": False,
        "requests": [],
        "page_errors": [],
        "console_errors": [],
        "failed_requests": [],
        "exit_code": 1,
    }
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            report["browser_version"] = browser.version
            context = browser.new_context(viewport={"width": 1280, "height": 960})
            context.tracing.start(screenshots=True, snapshots=True, sources=True)
            page = context.new_page()
            page.set_default_timeout(60000)
            page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
            page.on(
                "console",
                lambda message: (
                    report["console_errors"].append(message.text)
                    if message.type == "error"
                    else None
                ),
            )
            page.on(
                "requestfailed",
                lambda request: report["failed_requests"].append(
                    {"url": request.url, "failure": request.failure}
                ),
            )
            try:
                page.goto(args.base_url.rstrip("/") + "/docs", wait_until="networkidle")
                expect(
                    page.get_by_text("Lite-OmniJev local text gateway", exact=False).first
                ).to_be_visible()
                health = page.locator("#operations-default-health_health_get")
                health.locator(".opblock-summary").click()
                health.get_by_role("button", name="Try it out").click()
                with page.expect_response("**/health") as response:
                    health.get_by_role("button", name="Execute", exact=True).click()
                report["health_executed"] = True
                report["health_status"] = response.value.status
                report["health"] = payload = response.value.json()
                require(response.value.status == 200, "health HTTP status")
                require(payload["inference_configured"] is True, "model backend not configured")
                require(payload["model_version"] == args.expected_model_version, "health version")
                page.screenshot(path=str(output / "swagger-health.png"), full_page=True)
                questions = {
                    "choice": {
                        "type": "choice",
                        "instructions": "What color is the box?",
                        "criteria": {"red": "Red", "blue": "Blue"},
                    },
                    "noul": {"type": "noul", "instructions": "Is the box red?"},
                    "score": {
                        "type": "score",
                        "instructions": "Rate how many boxes there are.",
                        "criteria": ["None", "One", "Two or more"],
                    },
                    "insufficient": {
                        "type": "noul",
                        "instructions": "Was the box manufactured in France?",
                    },
                }
                body = {
                    "state": "There is one red box. Its origin is unknown.",
                    "questions": questions,
                }
                result = post(page, body)
                report["requests"].append({"name": "four_questions", "request": body, **result})
                require(result["status"] == 200, "configured decision request failed")
                payload = result["body"]
                require(payload["model"] == args.expected_model_version, "decision model version")
                require(set(payload["answers"]) == set(questions), "missing or extra questions")
                for qid, kind, labels in [
                    ("choice", "choice", {"red", "blue"}),
                    ("noul", "noul", {"yes", "no"}),
                    ("score", "score", {"0", "1", "2"}),
                    ("insufficient", "noul", {"yes", "no"}),
                ]:
                    check_answer(payload["answers"][qid], kind, labels)
                invalid = {
                    "state": "A red box.",
                    "questions": {
                        "bad": {
                            "type": "choice",
                            "instructions": "Color?",
                            "criteria": {"red": "Red"},
                        }
                    },
                }
                for name, request, malformed in [
                    ("invalid_criteria", invalid, False),
                    ("malformed_json", "{broken", True),
                ]:
                    result = post(page, request, malformed=malformed)
                    report["requests"].append({"name": name, "request": request, **result})
                    require(result["status"] == 422, f"{name} did not return 422")
                    require("answers" not in result["body"], f"{name} returned fake answers")
                require(not report["page_errors"], "browser JavaScript exception")
                require(not report["failed_requests"], "browser network request failed")
                report["exit_code"] = 0
            finally:
                try:
                    page.screenshot(path=str(output / "final.png"), full_page=True)
                finally:
                    context.tracing.stop(path=str(output / "trace.zip"))
                    browser.close()
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {error}"
        report["exit_code"] = 1
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["requests_sha256"] = hashlib.sha256(
            json.dumps(report["requests"], sort_keys=True).encode()
        ).hexdigest()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"report": str(output / "report.json"), "exit_code": report["exit_code"]}))
    return int(report["exit_code"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Forwarded local gateway URL")
    parser.add_argument("--expected-model-version", required=True)
    parser.add_argument("--output", required=True, help="New directory; refuses existing paths")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (
        url.scheme not in {"http", "https"}
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        parser.error("base URL must be a localhost origin without credentials, path or query")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
