import urllib.request, os
os.environ["IMGBB_KEY"] = "f83b4337c725d0d7b354352b083fd641"
exec(urllib.request.urlopen("https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/cells/_upload_refs.py", timeout=60).read().decode())
