#!/usr/bin/env python3
"""Run a cell in an existing molab notebook and print its output.

Drives the real page with Chromium on the runner: opens the notebook, writes the
code into the last cell, runs it, and scrapes the console output. molab has no
REST API for this, so the page is the interface.

Env:
  MOLAB_COOKIES  base64(JSON cookie list)
  NB_URL         notebook url
  CELL_FILE      path to a file holding the python to run
  WAIT_MS        how long to wait for the cell to finish (default 180000)
"""
import base64
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
SAME_SITE = {"no_restriction": "None", "lax": "Lax", "strict": "Strict",
             "unspecified": "Lax", "none": "None", None: "Lax"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cookies():
    data = json.loads(base64.b64decode(os.environ["MOLAB_COOKIES"]))
    out = []
    for c in data:
        d = {"name": c["name"], "value": c["value"], "domain": c["domain"],
             "path": c.get("path", "/"), "secure": bool(c.get("secure")),
             "httpOnly": bool(c.get("httpOnly")),
             "sameSite": SAME_SITE.get(c.get("sameSite"), "Lax")}
        if c.get("expirationDate"):
            d["expires"] = float(c["expirationDate"])
        out.append(d)
    return out


def main():
    nb = os.environ["NB_URL"].strip()
    code = open(os.environ["CELL_FILE"]).read()
    wait_ms = int(os.environ.get("WAIT_MS", "180000"))
    head = code.strip().splitlines()[0][:30]

    with sync_playwright() as p:
        b = p.chromium.launch(args=[
            "--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled"])
        ctx = b.new_context(viewport={"width": 1400, "height": 950},
                            user_agent=UA, locale="en-US")
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
        ctx.add_cookies(cookies())
        page = ctx.new_page()

        log("opening", nb)
        page.goto(nb, wait_until="domcontentloaded", timeout=150000)
        page.wait_for_timeout(12000)
        log("url:", page.url)

        # the editor lives in an iframe
        target = None
        for i in range(30):
            for fr in page.frames:
                try:
                    if fr.locator(".cm-editor").count():
                        target = fr
                        break
                except Exception:  # noqa: BLE001
                    continue
            if target:
                break
            page.wait_for_timeout(2000)
        if target is None:
            log("no .cm-editor found; frames:", [f.url[:60] for f in page.frames])
            b.close()
            return 2
        log("editor frame:", target.url[:80])

        editors = target.locator(".cm-editor")
        n = editors.count()
        log("editors:", n)
        ed = editors.nth(n - 1)
        ed.scroll_into_view_if_needed()
        ed.click()
        page.wait_for_timeout(400)
        page.keyboard.press("Control+a")
        page.keyboard.press("Backspace")
        page.keyboard.insert_text(code)
        page.wait_for_timeout(1200)
        txt = ed.inner_text()
        if head not in txt:
            log("INSERT MISMATCH, got:", repr(txt[:80]))
            b.close()
            return 3
        log("code inserted")

        page.keyboard.press("Shift+Enter")
        log("ran, waiting up to %ds" % (wait_ms // 1000))

        outs = target.locator("[data-testid=console-output-area]")
        last = ""
        stable = 0
        deadline = time.time() + wait_ms / 1000
        while time.time() < deadline:
            page.wait_for_timeout(3000)
            try:
                cur = outs.last.inner_text() if outs.count() else ""
            except Exception:  # noqa: BLE001
                cur = last
            if cur and cur == last:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
                last = cur
        log("OUTPUT_START")
        print(last[:8000], flush=True)
        log("OUTPUT_END")
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
