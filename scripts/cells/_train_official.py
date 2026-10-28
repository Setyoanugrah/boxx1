import base64, json, os, subprocess, sys, time, urllib.request

# Use diffusers' own DreamBooth LoRA SDXL example instead of a hand-rolled loop.
# The example handles the two text encoders, pooled embeddings, time_ids and the
# noise offset correctly; a manual loop kept failing on those details.

t0 = time.time()
def note(m): print(str(m)[:300], flush=True)

LOG = "/tmp/train.log"
for f in (LOG, "/tmp/train.done"):
    try: os.remove(f)
    except FileNotFoundError: pass

WORKER = r"""
import json, os, subprocess, sys, time, base64, urllib.request

def log(m):
    line = time.strftime("%H:%M:%S") + " " + str(m)[:300]
    print(line, flush=True)
    with open("/tmp/train.log", "a") as f:
        f.write(line + "\n")

try:
    log("start")
    subprocess.run([sys.executable,"-m","pip","install","-q","diffusers[training]",
                    "accelerate","transformers","peft","safetensors","Pillow",
                    "huggingface_hub","bitsandbytes","torchvision"], capture_output=True)
    log("pip done")

    # fetch the official example script
    SCRIPT = "/tmp/train_dreambooth_lora_sdxl.py"
    url = ("https://raw.githubusercontent.com/huggingface/diffusers/main/"
           "examples/dreambooth/train_dreambooth_lora_sdxl.py")
    try:
        src = urllib.request.urlopen(url, timeout=120).read()
        open(SCRIPT, "wb").write(src)
        log("script %d bytes" % len(src))
    except Exception as e:
        log("script fetch failed %s" % type(e).__name__)
        raise

    # dataset: <root>/<n>_rikustyle/  with image + matching .txt caption
    BASE = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/data/part%02d.json"
    ROOT = "/tmp/db"
    DST = os.path.join(ROOT, "5_rikustyle")
    os.makedirs(DST, exist_ok=True)
    pack = []
    for i in range(8):
        try:
            pack.extend(json.loads(urllib.request.urlopen(BASE % i, timeout=120).read()))
        except Exception:
            break
    log("rows %d" % len(pack))
    for row in pack:
        open(os.path.join(DST, row["id"] + ".png"), "wb").write(base64.b64decode(row["b64"]))
        open(os.path.join(DST, row["id"] + ".txt"), "w").write(row["caption"])
    log("dataset %d files" % len(os.listdir(DST)))

    OUT = "/tmp/lora_out"
    os.makedirs(OUT, exist_ok=True)
    cmd = [sys.executable, SCRIPT,
        "--pretrained_model_name_or_path", "cagliostrolab/animagine-xl-3.1",
        "--instance_data_dir", ROOT,
        "--output_dir", OUT,
        "--instance_prompt", "rikustyle, 1girl, solo, anime illustration",
        "--resolution", "1024",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "1e-4",
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", "50",
        "--max_train_steps", "900",
        "--checkpointing_steps", "450",
        "--mixed_precision", "bf16",
        "--rank", "16",
        "--gradient_checkpointing",
        "--use_8bit_adam",
        "--caption_column", "text",
        "--report_to", "none",
        "--seed", "42",
    ]
    log("launch: " + " ".join(cmd[-12:]))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=7000)
    log("exit %d" % p.returncode)
    tail = (p.stdout or "")[-2500:] + (p.stderr or "")[-2500:]
    for line in tail.splitlines()[-25:]:
        log("  | " + line[:220])

    if p.returncode == 0:
        for f in sorted(os.listdir(OUT)):
            fp = os.path.join(OUT, f)
            if os.path.isfile(fp):
                log("OUT %s %d KB" % (f, os.path.getsize(fp)//1024))
        log("ALLDONE")
    else:
        log("TRAIN_FAILED")
except Exception as e:
    import traceback
    log("FATAL " + repr(e)[:200])
    log(traceback.format_exc()[-1200:])
finally:
    open("/tmp/train.done","w").write("1")
"""

WPATH = "/tmp/train_worker_%d.py" % int(time.time())
open(WPATH,"w").write(WORKER)
p = subprocess.Popen([sys.executable,"-u",WPATH],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
note("worker pid %d launched" % p.pid)
note("returns immediately")
