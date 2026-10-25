#!/usr/bin/env python3
"""Print URL download server Bedrock Linux terbaru dari API Mojang.

Response-nya nested: {"result": {"links": [...]}}, bukan {"links": [...]}.
"""
import json
import sys
import urllib.request

API = "https://net-secondary.web.minecraft-services.net/api/v1.0/download/links"


def main():
    req = urllib.request.Request(API, headers={"User-Agent": "Mozilla/5.0"})
    data = json.load(urllib.request.urlopen(req, timeout=30))
    links = data.get("result", data).get("links", [])
    for item in links:
        if item.get("downloadType") == "serverBedrockLinux":
            print(item["downloadUrl"])
            return 0
    print("tidak ada serverBedrockLinux di response", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
