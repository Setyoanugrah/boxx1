import urllib.request, os
os.environ["IMGBB_KEY"] = "f83b4337c725d0d7b354352b083fd641"
U = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/cells/_lora_body.py"
exec(urllib.request.urlopen(U, timeout=60).read().decode())
