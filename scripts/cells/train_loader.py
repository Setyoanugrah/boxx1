import urllib.request, os, time
os.environ["IMGBB_KEY"] = "f83b4337c725d0d7b354352b083fd641"
U = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/cells/_train_body.py?v=" + str(int(time.time()))
src = urllib.request.urlopen(U, timeout=60).read().decode()
assert "torch.cat([e1, e2], dim=-1)" in src, "STALE BODY - concat fix missing"
print("BODY OK, %d chars" % len(src), flush=True)
exec(src)
