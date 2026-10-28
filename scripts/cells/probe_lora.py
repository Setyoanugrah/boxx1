import os, glob, json
paths = glob.glob("/tmp/lora/**/*", recursive=True)
files = [p for p in paths if os.path.isfile(p)]
print("PROBE_FILES " + json.dumps(files[:30]))
print("PROBE_SIZES " + json.dumps({p: os.path.getsize(p) for p in files if os.path.getsize(p) > 100000}))
