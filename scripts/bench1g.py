#!/usr/bin/env python3
"""Benchmark 1 GiB upload + download against a Nebula instance.

Runs on a GitHub runner: generates 1 GiB locally, uploads it with N parallel
chunk lanes, waits for the storage worker, then downloads it back with parallel
Range requests (the same shape the service worker uses) and a single stream for
comparison. Both directions are verified with SHA-256.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ.get("NEBULA_BASE", "https://drive.rikuzetsi.my.id").rstrip("/")
SIZE = int(os.environ.get("BENCH_MB", "1024")) * 1024 * 1024
UP_LANES = int(os.environ.get("UP_LANES", "6"))
DL_LANES = int(os.environ.get("DL_LANES", "8"))
BLOCK = 8 * 1024 * 1024
PATH = "/tmp/bench.bin"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

ok = fail = 0
results = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chk(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        log("  PASS", name)
    else:
        fail += 1
        log("  FAIL", name, extra)
    return cond


def fnv1a(text):
    h = 0x811C9DC5
    for b in text.encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def solve_pow(seed, bits):
    mask = (1 << bits) - 1
    n = 0
    while (fnv1a(seed + str(n)) & mask) != 0:
        n += 1
    return "%s:%d" % (seed, n)


def call(path, method="GET", body=None, headers=None, raw=False, timeout=300):
    h = dict(headers or {})
    h.setdefault("User-Agent", UA)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["content-type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as f:
            b = f.read()
            return f.status, (b if raw else json.loads(b or b"{}"))
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            return e.code, json.loads(b or b"{}")
        except Exception:
            return e.code, b[:300]


def make_file():
    log("generating %d MiB…" % (SIZE // 1048576))
    if os.path.exists(PATH) and os.path.getsize(PATH) == SIZE:
        return
    rnd = os.urandom(1024 * 1024)
    with open(PATH, "wb") as fh:
        for i in range(SIZE // (1024 * 1024)):
            # vary each MiB so the payload is not a repeated block
            fh.write(bytes([(i * 37) & 0xFF]) + rnd[1:])
    with open(PATH, "r+b") as fh:
        fh.seek(SIZE // 2)
        fh.write(os.urandom(1 << 16))
        fh.seek(SIZE - (1 << 16))
        fh.write(os.urandom(1 << 16))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    make_file()
    src_hash = sha256_file(PATH)
    log("source sha256:", src_hash)

    # ---------- account ----------
    user = "bench" + secrets.token_hex(4)
    pw = "pw" + secrets.token_hex(6)
    st, ch = call("/api/challenge")
    if st != 200:
        log("challenge failed:", ch)
        return 1
    t0 = time.time()
    powtok = solve_pow(ch["seed"], ch["bits"])
    log("pow solved in %.1fs (bits %s)" % (time.time() - t0, ch.get("bits")))
    st, d = call("/api/register", "POST",
                 {"username": user, "password": pw, "nickname": "", "pow": powtok})
    if not chk("register", st == 200, d):
        return 1
    st, d = call("/api/keys", "POST", {"name": "bench", "scope": "full"})
    if not chk("mint api key", st == 200, d):
        return 1
    key = d["key"]
    kid = d["info"]["id"]
    H = {"X-API-Key": key, "User-Agent": UA}
    log("account:", user, "| key:", key[:12] + "…")

    # ---------- upload ----------
    log("[1] upload %d MiB, %d lanes" % (SIZE // 1048576, UP_LANES))
    st, d = call("/api/upload/init", "POST", {"name": "bench.bin", "size": SIZE}, headers=H)
    if not chk("init", st == 200, d):
        return 1
    uid, CH = d["id"], d["chunk_size"]
    nchunks = (SIZE + CH - 1) // CH
    log("  chunk %d KiB · %d chunks" % (CH // 1024, nchunks))

    cursor = [0]
    lock = threading.Lock()
    errs = []
    sent = [0]
    fh = open(PATH, "rb")
    t_up = time.time()

    def lane():
        while True:
            with lock:
                i = cursor[0]
                cursor[0] += 1
            if i >= nchunks:
                return
            off = i * CH
            with lock:
                fh.seek(off)
                part = fh.read(CH)
            for attempt in range(5):
                try:
                    req = urllib.request.Request(
                        BASE + "/api/upload/chunk?id=%s&offset=%d" % (uid, off),
                        data=part,
                        headers={**H, "content-type": "application/octet-stream"},
                        method="POST")
                    urllib.request.urlopen(req, timeout=300).read()
                    with lock:
                        sent[0] += len(part)
                        if sent[0] % (128 * 1024 * 1024) < CH:
                            el = time.time() - t_up
                            log("    %d MiB · %.1f MB/s" % (sent[0] // 1048576,
                                                            (sent[0] / 1048576) / el))
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 4:
                        errs.append(repr(e))
                        return
                    time.sleep(0.7 * (attempt + 1))

    ts = [threading.Thread(target=lane) for _ in range(UP_LANES)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    fh.close()
    up_el = time.time() - t_up
    if not chk("all chunks sent", not errs, errs[:1]):
        return 1
    up_mbps = (SIZE / 1048576) / up_el
    log("  upload %.1fs · %.2f MB/s" % (up_el, up_mbps))

    log("[2] finish")
    for _ in range(90):
        st, d = call("/api/upload/finish", "POST", {"id": uid}, headers=H)
        if st == 200 and d.get("retry"):
            log("    queued, waiting %ss" % d.get("wait"))
            time.sleep(min(int(d.get("wait", 5)), 65))
            continue
        break
    if not chk("finish", st == 200, d):
        return 1
    fid = d["file"]["id"]
    log("  file id:", fid, "status:", d["file"]["status"])

    log("[3] waiting for storage worker…")
    t_store = time.time()
    ready = False
    row = None
    for i in range(180):
        st, fl = call("/api/files", headers=H)
        row = next((f for f in fl.get("files", []) if f["id"] == fid), None)
        if row and row["status"] == "ready":
            ready = True
            break
        if row and row["status"] == "failed":
            break
        if i % 8 == 0:
            log("    status=%s · %.0fs" % ((row or {}).get("status"), time.time() - t_store))
        time.sleep(4)
    store_el = time.time() - t_store
    if not chk("file ready in storage", ready, (row or {}).get("status")):
        return 1
    log("  stored in %.1fs" % store_el)

    # ---------- download: parallel ----------
    log("[4] download %d MiB, %d parallel ranges of %d MiB" %
        (SIZE // 1048576, DL_LANES, BLOCK // 1048576))
    nblocks = (SIZE + BLOCK - 1) // BLOCK
    t_dl = time.time()
    got = [0]
    glock = threading.Lock()

    def fetch(a, b):
        req = urllib.request.Request(
            "%s/api/file/%s/download" % (BASE, fid),
            headers={**H, "Range": "bytes=%d-%d" % (a, b)})
        data = urllib.request.urlopen(req, timeout=300).read()
        with glock:
            got[0] += len(data)
            if got[0] % (256 * 1024 * 1024) < BLOCK:
                el = time.time() - t_dl
                log("    %d MiB · %.2f MB/s" % (got[0] // 1048576, (got[0] / 1048576) / el))
        return data

    out_hash = hashlib.sha256()
    out_size = 0
    with cf.ThreadPoolExecutor(max_workers=DL_LANES) as ex:
        futs = {}
        nxt = 0
        for i in range(nblocks):
            while len(futs) < DL_LANES and nxt < nblocks:
                j = nxt
                nxt += 1
                a = j * BLOCK
                b = min(a + BLOCK, SIZE) - 1
                futs[j] = ex.submit(fetch, a, b)
            buf = futs.pop(i).result()  # in order, so the hash streams
            out_hash.update(buf)
            out_size += len(buf)
    dl_el = time.time() - t_dl
    dl_mbps = (SIZE / 1048576) / dl_el
    chk("downloaded size exact", out_size == SIZE, (out_size, SIZE))
    chk("sha256 matches source", out_hash.hexdigest() == src_hash,
        (out_hash.hexdigest(), src_hash))
    log("  parallel %.1fs · %.2f MB/s" % (dl_el, dl_mbps))

    # ---------- download: single stream ----------
    log("[5] single stream for comparison")
    t1 = time.time()
    tot = 0
    h1 = hashlib.sha256()
    req = urllib.request.Request("%s/api/file/%s/download" % (BASE, fid), headers=H)
    with urllib.request.urlopen(req, timeout=900) as r:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            tot += len(b)
            h1.update(b)
    el1 = time.time() - t1
    single_mbps = (SIZE / 1048576) / el1
    chk("single stream intact", tot == SIZE and h1.hexdigest() == src_hash)
    log("  single %.1fs · %.2f MB/s" % (el1, single_mbps))

    # ---------- dedup ----------
    log("[6] re-upload identical payload (dedup)")
    t2 = time.time()
    st, d = call("/api/upload/init", "POST",
                 {"name": "bench-copy.bin", "size": SIZE}, headers=H)
    uid2 = d["id"]
    cursor2 = [0]

    def lane2():
        while True:
            with lock:
                i = cursor2[0]
                cursor2[0] += 1
            if i >= nchunks:
                return
            off = i * CH
            with lock:
                fh.seek(off)
                part = fh.read(CH)
            try:
                req = urllib.request.Request(
                    BASE + "/api/upload/chunk?id=%s&offset=%d" % (uid2, off),
                    data=part,
                    headers={**H, "content-type": "application/octet-stream"},
                    method="POST")
                urllib.request.urlopen(req, timeout=300).read()
            except Exception:  # noqa: BLE001
                return

    fh = open(PATH, "rb")
    ts = [threading.Thread(target=lane2) for _ in range(UP_LANES)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    fh.close()
    st, d = call("/api/upload/finish", "POST", {"id": uid2}, headers=H)
    chk("dedup finish", st == 200, d)
    dedup = (d.get("file") or {}).get("dedup") is True
    chk("dedup flagged", dedup, d.get("file"))
    chk("dedup returned ready", (d.get("file") or {}).get("status") == "ready",
        (d.get("file") or {}).get("status"))
    dedup_el = time.time() - t2
    st, me = call("/api/me", headers=H)
    chk("quota charged once", me["used"] == SIZE, (me["used"], SIZE))
    log("  dedup round-trip %.1fs · quota %.0f MiB" % (dedup_el, me["used"] / 1048576))

    # ---------- cleanup ----------
    log("[7] cleanup")
    for f in call("/api/files", headers=H)[1].get("files", []):
        call("/api/file/" + f["id"], "DELETE", None, headers=H)
    st, d = call("/api/trash", "DELETE", None, headers=H)
    log("  trashed removed:", d.get("removed"))
    call("/api/keys/" + kid, "DELETE")

    results.update({
        "size_mb": SIZE // 1048576,
        "upload_mbps": round(up_mbps, 2),
        "upload_secs": round(up_el, 1),
        "download_mbps": round(dl_mbps, 2),
        "download_secs": round(dl_el, 1),
        "single_mbps": round(single_mbps, 2),
        "single_secs": round(el1, 1),
        "speedup": round(dl_mbps / single_mbps, 2) if single_mbps else 0,
        "store_secs": round(store_el, 1),
        "dedup_secs": round(dedup_el, 1),
        "dedup_worked": dedup,
        "sha256": src_hash,
        "passed": ok,
        "failed": fail,
    })
    log("")
    log("RESULT: %d passed, %d failed" % (ok, fail))
    log(json.dumps(results, indent=1))

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## Nebula %d MiB benchmark\n\n" % (SIZE // 1048576))
            f.write("| metric | value |\n| --- | --- |\n")
            f.write("| upload (%d lanes) | %.2f MB/s (%.1fs) |\n" %
                    (UP_LANES, up_mbps, up_el))
            f.write("| download (%d ranges) | %.2f MB/s (%.1fs) |\n" %
                    (DL_LANES, dl_mbps, dl_el))
            f.write("| download (1 stream) | %.2f MB/s (%.1fs) |\n" % (single_mbps, el1))
            f.write("| parallel speedup | %.2fx |\n" % results["speedup"])
            f.write("| storage worker | %.1fs |\n" % store_el)
            f.write("| dedup re-upload | %.1fs (worked: %s) |\n" % (dedup_el, dedup))
            f.write("| sha256 verified | `%s` |\n" % src_hash)
            f.write("| assertions | %d passed, %d failed |\n" % (ok, fail))

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
