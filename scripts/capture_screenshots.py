"""
Capture dashboard screenshots using Playwright.
Starts Streamlit, waits for it to be ready, scrolls to tab content, shuts down.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BASE_DIR / "assets"
APP_PATH = BASE_DIR / "app" / "dashboard.py"
PORT = 8502


def wait_for_server(url: str, timeout: int = 60) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def scroll_and_shoot(page, path: str, scroll_to: int = 600, wait_ms: int = 1500) -> None:
    """Scroll to tab content area then screenshot."""
    page.evaluate(f"window.scrollTo(0, {scroll_to})")
    page.wait_for_timeout(wait_ms)
    page.screenshot(path=path, full_page=False)
    page.evaluate("window.scrollTo(0, 0)")


def click_tab(page, partial_name: str, wait_ms: int = 3000) -> None:
    """Click the first tab whose label contains partial_name (emoji-safe)."""
    page.locator(f"[role='tab']:has-text('{partial_name}')").first.click()
    page.wait_for_timeout(wait_ms)


def capture() -> None:
    from playwright.sync_api import sync_playwright

    print("Starting Streamlit…")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH),
         "--server.port", str(PORT),
         "--server.headless", "true",
         "--server.runOnSave", "false",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    base_url = f"http://localhost:{PORT}"
    print(f"Waiting for server at {base_url}…")
    if not wait_for_server(base_url):
        proc.terminate()
        raise RuntimeError("Streamlit did not start in time.")

    time.sleep(8)  # let Streamlit fully render
    ASSETS_DIR.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # 1440×900 — matches a typical laptop display
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(base_url, wait_until="networkidle")
        page.wait_for_timeout(4000)

        # ── Home (hero + model cards) ────────────────────────────────────────
        print("Capturing home hero…")
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(800)
        page.screenshot(path=str(ASSETS_DIR / "dashboard_home.png"), full_page=False)

        # ── Climate Overview ─────────────────────────────────────────────────
        print("Capturing Climate Overview…")
        click_tab(page, "Climate Overview")
        scroll_and_shoot(page, str(ASSETS_DIR / "dashboard_overview.png"), scroll_to=520)

        # ── Team Rankings ────────────────────────────────────────────────────
        print("Capturing Team Rankings…")
        click_tab(page, "Team Rankings")
        scroll_and_shoot(page, str(ASSETS_DIR / "dashboard_rankings.png"), scroll_to=480)

        # ── Match Simulator ──────────────────────────────────────────────────
        print("Capturing Match Simulator…")
        click_tab(page, "Match Simulator")
        scroll_and_shoot(page, str(ASSETS_DIR / "dashboard_simulator.png"), scroll_to=440)

        # ── Future Scenarios ─────────────────────────────────────────────────
        print("Capturing Future Scenarios…")
        click_tab(page, "Future Scenarios")
        scroll_and_shoot(page, str(ASSETS_DIR / "dashboard_future.png"), scroll_to=400)

        # ── Extreme Weather ──────────────────────────────────────────────────
        print("Capturing Extreme Weather…")
        click_tab(page, "Extreme Weather")
        scroll_and_shoot(page, str(ASSETS_DIR / "dashboard_extreme.png"), scroll_to=440)

        browser.close()

    proc.terminate()
    print("\nDone. Screenshots saved to", ASSETS_DIR)
    for f in sorted(ASSETS_DIR.glob("dashboard_*.png")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    capture()
