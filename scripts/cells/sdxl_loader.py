import urllib.request, os
U = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/cells/_sdxl_body.py"
src = urllib.request.urlopen(U, timeout=60).read().decode()
src = src.replace('os.environ.get("GH_TOKEN", "")', repr(os.environ.get("GH_TOKEN", "")))
exec(src)
