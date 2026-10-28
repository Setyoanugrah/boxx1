import os, time
LOG = "/tmp/train.log"; DONE = "/tmp/train.done"
time.sleep(2)
txt = open(LOG).read() if os.path.exists(LOG) else "(no log yet)"
lines = [l for l in txt.splitlines() if any(k in l for k in
         ("step ","LORA","trained","SAMPLE","MODEL","ALLDONE","FATAL","encoded","images","base loaded","worker"))]
print("TRAINLOG_BEGIN")
print("\n".join(lines[-40:]))
print("TRAINLOG_END")
print("DONE" if os.path.exists(DONE) else "RUNNING")
