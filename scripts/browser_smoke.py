"""Verify the local API documentation in a real Chromium browser, without model claims."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from playwright.sync_api import expect, sync_playwright


def main() -> int:
    output = Path("reports")
    output.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 960})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8000/docs", wait_until="networkidle")
        expect(
            page.get_by_text("Lite-OmniJev local text gateway", exact=False).first
        ).to_be_visible()
        health = page.locator("#operations-default-health_health_get")
        health.locator(".opblock-summary").click()
        health.get_by_role("button", name="Try it out").click()
        with page.expect_response("**/health") as health_response:
            health.get_by_role("button", name="Execute", exact=True).click()
        assert health_response.value.status == 200
        payload = health_response.value.json()
        assert payload["inference_configured"] is False
        decision = page.evaluate("""async () => {
          const response = await fetch('/v1/systemone', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({state:'The order shipped.',questions:{q:{
              type:'noul',instructions:'Has the order shipped?'}}})
          });
          return {status:response.status, body:await response.json()};
        }""")
        assert decision["status"] == 503 and "answers" not in decision["body"]
        page.screenshot(path=str(output / "browser-smoke.png"), full_page=True)
        report = {
            "time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "tool": "Python Playwright actual Chromium (MCP unavailable)",
            "browser_version": browser.version,
            "url": page.url,
            "health_execute_status": health_response.value.status,
            "health": payload,
            "unconfigured_decision": decision,
            "page_errors": errors,
            "scope": "M0 docs and unavailable path; final Playground/model acceptance pending",
            "exit_code": 0 if not errors else 1,
        }
        (output / "browser-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        browser.close()
        print(json.dumps(report, indent=2))
        return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
