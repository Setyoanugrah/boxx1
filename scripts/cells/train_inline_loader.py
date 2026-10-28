import urllib.request, time
U = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/cells/_train_inline.py?v=" + str(int(time.time()))
exec(urllib.request.urlopen(U, timeout=60).read().decode())
