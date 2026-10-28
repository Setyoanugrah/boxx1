import os
LOG = "/tmp/train.log"; DONE = "/tmp/train.done"
txt = open(LOG).read() if os.path.exists(LOG) else "(log belum ada)"
print("=== TRAIN LOG ===")
print(txt[-3000:])
print("=== STATUS:", "SELESAI" if os.path.exists(DONE) else "JALAN", "===")
