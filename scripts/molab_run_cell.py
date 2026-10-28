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

        # APPEND=0: use the notebook's existing empty cell (fresh notebook).
        if False:
          try:
              target.locator(".cm-content").last.click()
              page.keyboard.press("Control+End")
              page.wait_for_timeout(300)
              page.keyboard.press("Shift+Enter")
              page.wait_for_timeout(800)
              page.keyboard.press("Control+End")
              page.wait_for_timeout(300)
              n2 = editors.count()
              log("editors after append:", n2)
              if n2 > n:
                  n = n2
          except Exception as exc:  # noqa: BLE001
              log("append failed:", type(exc).__name__)
        ed = editors.first
        ed.scroll_into_view_if_needed()
        page.wait_for_timeout(400)

        # The new cell's CodeMirror content is what actually takes keystrokes.
        inserted = False
        for attempt in range(4):
            try:
                content = target.locator(".cm-content").first
                content.click()
                page.wait_for_timeout(500)
                page.keyboard.press("Control+a")
                page.keyboard.press("Delete")
                page.wait_for_timeout(300)
                page.keyboard.insert_text(code)
                page.wait_for_timeout(1500)
                txt = content.inner_text()
                if head in txt:
                    inserted = True
                    log("code inserted on attempt", attempt)
                    break
                log("attempt", attempt, "mismatch:", repr(txt[:60]))
            except Exception as exc:  # noqa: BLE001
                log("attempt", attempt, "err", type(exc).__name__)
            page.keyboard.press("Escape")
            page.wait_for_timeout(600)

        if not inserted:
            log("INSERT FAILED")
            b.close()
            return 3

        page.keyboard.press("Shift+Enter")
        log("ran, waiting up to %ds" % (wait_ms // 1000))

        outs = target.locator("[data-testid=console-output-area]")
        log("console areas:", outs.count())
        last = ""
        stable = 0
        deadline = time.time() + wait_ms / 1000
        while time.time() < deadline:
            page.wait_for_timeout(3000)
            try:
                # the newest cell's output is the one that keeps growing;
                # pick the longest visible block rather than the last in DOM order
                best = ""
                for i in range(outs.count()):
                    try:
                        t = outs.nth(i).inner_text() or ""
                    except Exception:  # noqa: BLE001
                        continue
                    if len(t) > len(best):
                        best = t
                cur = best
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
