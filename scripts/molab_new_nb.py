#!/usr/bin/env python3
"""Create (or reuse) a molab notebook from a GitHub runner, then report its URL.

Uses the stored browser cookies to drive the real page with Chromium, because
molab has no REST endpoint for notebook creation - it goes through the page's
own runtime. Everything runs on the runner; nothing touches the phone.

Env:
  MOLAB_COOKIES  base64(JSON list of cookie objects)
  MODE           "create" (default) or "list"
"""
import base64
import json
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
SAME_SITE = {"no_restriction": "None", "lax": "Lax", "strict": "Strict",
             "unspecified": "Lax", "none": "None", None: "Lax"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_cookies():
    raw = os.environ["MOLAB_COOKIES"]
    data = json.loads(base64.b64decode(raw))
    out = []
    for c in data:
        d = {
            "name": c["name"],
            "value": c["value"],
            "domain": c["domain"],
            "path": c.get("path", "/"),
            "secure": bool(c.get("secure")),
            "httpOnly": bool(c.get("httpOnly")),
            "sameSite": SAME_SITE.get(c.get("sameSite"), "Lax"),
        }
        exp = c.get("expirationDate")
        if exp:
            d["expires"] = float(exp)
        out.append(d)
    return out


def main():
    cookies = load_cookies()
    log("cookies loaded:", len(cookies))
    mode = os.environ.get("MODE", "create")

    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ])
        ctx = browser.new_context(
            viewport={"width": 1400, "height": 950},
            user_agent=UA, locale="en-US",
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
        ctx.add_cookies(cookies)
        page = ctx.new_page()

        log("opening /notebooks")
        page.goto("https://molab.marimo.io/notebooks",
                  wait_until="domcontentloaded", timeout=120000)
        page.wait_for_timeout(6000)
        log("url:", page.url)
        log("title:", page.title())

        if "sign-in" in page.url or "login" in page.url.lower():
            log("NOT LOGGED IN - cookies rejected")
            body = page.inner_text("body")[:400]
            log("body:", body.replace("\n", " | "))
            ctx.storage_state(path="/tmp/state.json")
            browser.close()
            return 2

        if mode == "list":
            links = page.eval_on_selector_all(
                "a[href*='/notebooks/nb_']",
                "els => els.map(e => e.getAttribute('href'))")
            log("notebooks found:", len(links))
            for l in dict.fromkeys(links):
                log("  ", l)
            browser.close()
            return 0

        # create
        log("looking for a create control")
        clicked = False
        for sel in ("text=Create a notebook", "text=New notebook",
                    "button:has-text('Create')", "text=Create"):
            try:
                loc = page.locator(sel).first
                if loc.count() and loc.is_visible():
                    log("clicking:", sel)
                    loc.click(timeout=15000)
                    clicked = True
                    break
            except Exception as exc:  # noqa: BLE001
                log("  skip", sel, type(exc).__name__)
        if not clicked:
            log("no create control found; dumping buttons")
            for i, t in enumerate(page.eval_on_selector_all(
                    "button", "els => els.map(e => (e.innerText||e.getAttribute('aria-label')||'').trim())")):
                if t:
                    log("   btn", i, t[:60])
            browser.close()
            return 3

        nb_url = None
        for i in range(40):
            page.wait_for_timeout(3000)
            if re.search(r"/notebooks/nb_[A-Za-z0-9]+", page.url):
                nb_url = page.url
                break
            if "/notebooks/" in page.url and page.url.rstrip("/").endswith("/notebooks") is False:
                nb_url = page.url
                break
        log("result url:", page.url)

        # fall back to scraping the listing
        if not nb_url or "nb_" not in (nb_url or ""):
            page.goto("https://molab.marimo.io/notebooks",
                      wait_until="domcontentloaded", timeout=90000)
            page.wait_for_timeout(5000)
            links = page.eval_on_selector_all(
                "a[href*='/notebooks/nb_']",
                "els => els.map(e => e.getAttribute('href'))")
            if links:
                nb_url = "https://molab.marimo.io" + links[0]
                log("from listing:", nb_url)

        log("NOTEBOOK_URL:", nb_url)
        ctx.storage_state(path="/tmp/state.json")
        browser.close()

        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as f:
                f.write("## molab\n\n")
                f.write("- notebook: `%s`\n" % (nb_url or "not found"))
                f.write("- cookies: %d\n" % len(cookies))
        return 0 if nb_url else 4


if __name__ == "__main__":
    sys.exit(main())
