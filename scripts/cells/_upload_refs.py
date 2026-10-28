import base64, json, os, urllib.parse, urllib.request
KEY = os.environ.get("IMGBB_KEY","")
out = {}
for i in range(1, 8):
    p = "/root/boxx1/scripts/refs/ref%02d.jpg" % i
    if not os.path.exists(p):
        continue
    data = base64.b64encode(open(p,"rb").read()).decode()
    b = urllib.parse.urlencode({"image": data, "name": "ref%02d" % i}).encode()
    req = urllib.request.Request("https://api.imgbb.com/1/upload?key="+KEY, data=b, method="POST")
    with urllib.request.urlopen(req, timeout=240) as f:
        out["ref%02d" % i] = json.loads(f.read())["data"]["url"]
    print("ref%02d -> %s" % (i, out["ref%02d"%i][:60]), flush=True)
print("REFS " + json.dumps(out))
