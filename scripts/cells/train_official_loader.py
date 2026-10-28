import urllib.request, os, time
os.environ["IMGBB_KEY"] = "f83b4337c725d0d7b354352b083fd641"
U = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/cells/_train_official.py?v=" + str(int(time.time()))
exec(urllib.request.urlopen(U, timeout=60).read().decode())
