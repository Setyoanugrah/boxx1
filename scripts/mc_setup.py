#!/usr/bin/env python3
"""Patch server.properties untuk Bedrock dedicated server.

transport=raknet wajib: default 1.26+ adalah 'nethernet' (WebRTC) yang
pakai port acak dan gak bisa di-port-forward / di-tunnel UDP biasa.
"""
import sys, os

WANT = {
    "transport": "raknet",
    "server-name": "Runzz Bedrock",
    "gamemode": "survival",
    "difficulty": "normal",
    "max-players": "10",
    "online-mode": "true",
    "allow-cheats": "true",
    "allow-list": "false",
    "server-port": "19132",
    "server-portv6": "19133",
    "view-distance": "16",
    "tick-distance": "4",
    "player-idle-timeout": "0",
    "enable-lan-visibility": "true",
    "level-name": "Bedrock level",
    "default-player-permission-level": "member",
    "content-log-console-output-enabled": "true",
}

path = sys.argv[1] if len(sys.argv) > 1 else "server.properties"
if not os.path.exists(path):
    sys.exit("server.properties tidak ada: " + path)

out, seen = [], set()
for line in open(path, encoding="utf-8").read().splitlines():
    s = line.strip()
    if s and not s.startswith("#") and "=" in s:
        k = s.split("=", 1)[0].strip()
        if k in WANT:
            out.append("%s=%s" % (k, WANT[k]))
            seen.add(k)
            continue
    out.append(line)

for k, v in WANT.items():
    if k not in seen:
        out.append("%s=%s" % (k, v))

open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("patched %s (transport=raknet)" % path)
for k in ("transport", "server-port", "server-name", "max-players"):
    print("  %s=%s" % (k, WANT[k]))
