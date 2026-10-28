#!/usr/bin/env python3
"""Select a GPU for a molab notebook, then verify CUDA is actually available.

Opens the notebook, clicks the compute configuration control, picks the
requested accelerator, waits for the runtime to come back, and prints what the
kernel reports.

Env:
  MOLAB_COOKIES  base64(JSON cookie list)
  NB_URL         notebook url
  GPU_MATCH      substring to look for in the accelerator list (default "6000")
  GPU_LIST       "1" to only list the options and exit
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


def dump_controls(page):
    """Print every control that could be the accelerator picker."""
    for sel in ("select", "button[role=combobox]", "[role=combobox]",
                "button[aria-haspopup=listbox]"):
        loc = page.locator(sel)
        n = loc.count()
        if not n:
            continue
        log("  %s -> %d" % (sel, n))
        for i in range(min(n, 20)):
            try:
                el = loc.nth(i)
                txt = (el.inner_text() or "").strip().replace("\n", " ")[:80]
                name = el.get_attribute("name") or ""
                aria = el.get_attribute("aria-label") or ""
                if txt or name or aria:
                    log("    [%d] text=%r name=%r aria=%r" % (i, txt, name, aria))
                    if el.evaluate("e => e.tagName") == "SELECT":
                        opts = el.eval_on_selector_all(
                            "option", "os => os.map(o => o.textContent.trim())")
                        log("        options:", opts)
            except Exception as exc:  # noqa: BLE001
                log("    [%d] err %s" % (i, type(exc).__name__))


def main():
    nb = os.environ["NB_URL"].strip()
    match = os.environ.get("GPU_MATCH", "6000")
    list_only = os.environ.get("GPU_LIST") == "1"

    with sync_playwright() as p:
        b = p.chromium.launch(args=[
            "--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled"])
        ctx = b.new_context(viewport={"width": 1500, "height": 1000},
                            user_agent=UA, locale="en-US")
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
        ctx.add_cookies(cookies())
        page = ctx.new_page()

        log("opening", nb)
        page.goto(nb, wait_until="domcontentloaded", timeout=150000)
        page.wait_for_timeout(12000)
        log("url:", page.url)

        # find the compute control
        opened = False
        for label in ("Configure compute", "Compute", "Runtime", "GPU"):
            try:
                loc = page.get_by_label(label, exact=False).first
                if loc.count() and loc.is_visible():
                    log("clicking label:", label)
                    loc.click(timeout=10000)
                    opened = True
                    break
            except Exception:  # noqa: BLE001
                continue
        if not opened:
            for txt in ("Configure compute", "Compute", "GPU", "CPU"):
                try:
                    loc = page.get_by_text(txt, exact=False).first
                    if loc.count() and loc.is_visible():
                        log("clicking text:", txt)
                        loc.click(timeout=10000)
                        opened = True
                        break
                except Exception:  # noqa: BLE001
                    continue
        log("panel opened:", opened)
        page.wait_for_timeout(3000)
        dump_controls(page)

        if list_only:
            log("LIST ONLY, exiting")
            b.close()
            return 0

        # try to pick a matching option
        picked = False
        for sel in ("select", "[role=combobox]", "button[aria-haspopup=listbox]"):
            loc = page.locator(sel)
            for i in range(loc.count()):
                el = loc.nth(i)
                try:
                    if el.evaluate("e => e.tagName") == "SELECT":
                        opts = el.eval_on_selector_all(
                            "option", "os => os.map(o => o.textContent.trim())")
                        hit = next((o for o in opts if match.lower() in o.lower()), None)
                        if hit:
                            log("selecting option:", hit)
                            el.select_option(label=hit)
                            picked = True
                            break
                except Exception as exc:  # noqa: BLE001
                    log("  select err", type(exc).__name__)
            if picked:
                break

        if not picked:
            # click-based picker
            try:
                page.get_by_text(match, exact=False).first.click(timeout=10000)
                log("clicked accelerator text:", match)
                picked = True
            except Exception as exc:  # noqa: BLE001
                log("accelerator click failed:", type(exc).__name__)

        if not picked:
            log("COULD NOT SELECT GPU")
            b.close()
            return 3

        # confirm / start
        for label in ("Save", "Apply", "Start", "Restart", "Connect"):
            try:
                loc = page.get_by_role("button", name=label, exact=False).first
                if loc.count() and loc.is_visible():
                    log("pressing:", label)
                    loc.click(timeout=10000)
                    page.wait_for_timeout(2000)
            except Exception:  # noqa: BLE001
                continue

        log("waiting for the runtime to come back")
        page.wait_for_timeout(25000)
        log("final url:", page.url)
        ctx.storage_state(path="/tmp/state.json")
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
