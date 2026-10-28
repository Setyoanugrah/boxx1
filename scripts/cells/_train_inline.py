import base64, glob, json, os, shutil, site, subprocess, sys, time, urllib.request

# Train a LoRA for "Trixtom Imagen 1" with the official diffusers DreamBooth SDXL
# example. Runs inline in this cell on purpose: molab suspends a notebook once the
# browser disconnects, which killed every detached worker we tried.

t0 = time.time()
def log(m): print(str(m)[:280], flush=True)

log("cuda check")
import torch
log("cuda=%s %s" % (torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))

subprocess.run([sys.executable,"-m","pip","install","-q","diffusers","accelerate",
                "transformers","peft","safetensors","bitsandbytes","Pillow"], capture_output=True)
log("pip done")

# --- wandb: this image ships a broken proto module and every diffusers example
# --- does `import wandb` at module level. Remove the package tree outright.
roots = list(site.getsitepackages()) + [site.getusersitepackages()]
hits = []
for r in roots:
    hits += glob.glob(os.path.join(r, "wandb*"))
log("wandb paths %s" % [os.path.basename(h) for h in hits][:5])
for h in hits:
    shutil.rmtree(h, ignore_errors=True) if os.path.isdir(h) else os.remove(h)
chk = subprocess.run([sys.executable,"-c","import wandb"], capture_output=True, text=True)
log("wandb import rc=%d" % chk.returncode)

SCRIPT = "/tmp/db_lora_sdxl.py"
url = ("https://raw.githubusercontent.com/huggingface/diffusers/main/"
       "examples/dreambooth/train_dreambooth_lora_sdxl.py")
open(SCRIPT,"wb").write(urllib.request.urlopen(url, timeout=120).read())
log("script %d bytes" % os.path.getsize(SCRIPT))

BASE = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/data/part%02d.json"
DST = "/tmp/db"
shutil.rmtree(DST, ignore_errors=True); os.makedirs(DST, exist_ok=True)
rows = []
for i in range(8):
    try: rows.extend(json.loads(urllib.request.urlopen(BASE % i, timeout=120).read()))
    except Exception: break
for r in rows:
    open(os.path.join(DST, r["id"] + ".png"),"wb").write(base64.b64decode(r["b64"]))
    open(os.path.join(DST, r["id"] + ".txt"),"w").write(r["caption"])
log("dataset %d images" % len(rows))

OUT = "/tmp/lora_out"
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT, exist_ok=True)

cmd = [sys.executable, SCRIPT,
  "--pretrained_model_name_or_path", "cagliostrolab/animagine-xl-3.1",
  "--instance_data_dir", DST,
  "--output_dir", OUT,
  "--instance_prompt", "rikustyle, 1girl, solo, anime illustration, masterpiece",
  "--resolution", "1024",
  "--train_batch_size", "1",
  "--gradient_accumulation_steps", "4",
  "--learning_rate", "1e-4",
  "--lr_scheduler", "cosine",
  "--lr_warmup_steps", "50",
  "--max_train_steps", "900",
  "--checkpointing_steps", "900",
  "--mixed_precision", "bf16",
  "--rank", "16",
  "--gradient_checkpointing",
  "--use_8bit_adam",
  "--report_to", "none",
  "--seed", "42"]

log("launching trainer")
p = subprocess.run(cmd, capture_output=True, text=True)
log("trainer exit %d" % p.returncode)
tail = ((p.stdout or "") + "\n" + (p.stderr or "")).strip().splitlines()
for line in tail[-18:]:
    log("  | " + line[:220])

saf = glob.glob(os.path.join(OUT, "*.safetensors"))
if not saf:
    log("NO_LORA")
    raise SystemExit(1)
log("LORA %s %d KB" % (os.path.basename(saf[0]), os.path.getsize(saf[0])//1024))

from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipe.load_lora_weights(OUT)
log("lora loaded for sampling")

KEY = os.environ.get("IMGBB_KEY", "")
import urllib.parse
def imgbb(path, name):
    d = base64.b64encode(open(path,"rb").read()).decode()
    b = urllib.parse.urlencode({"image": d, "name": name}).encode()
    req = urllib.request.Request("https://api.imgbb.com/1/upload?key=" + KEY, data=b, method="POST")
    with urllib.request.urlopen(req, timeout=240) as f:
        return json.loads(f.read())["data"]["url"]

TESTS = [
  (5,  "rikustyle, 1girl, solo, long red hair, red eyes, black leather jacket, neon city at night, upper body, masterpiece, high score, absurdres"),
  (9,  "rikustyle, 1girl, solo, long black hair, purple eyes, white crop top, bedroom, sitting on bed, soft light, masterpiece, high score, absurdres"),
  (13, "rikustyle, 1girl, solo, silver hair, grey eyes, blue one-piece swimsuit, arms behind head, plain background, masterpiece, high score, absurdres"),
]
for seed, prompt in TESTS:
    img = pipe(prompt=prompt,
               negative_prompt="lowres, bad anatomy, bad hands, worst quality, low quality, watermark, text",
               height=1024, width=832, num_inference_steps=28, guidance_scale=6.0,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    fp = "/tmp/s%d.png" % seed
    img.save(fp)
    try: log("SAMPLE %s" % imgbb(fp, "trixtom%d" % seed))
    except Exception as e: log("SAMPLE FAIL %s" % type(e).__name__)

log("MODEL Trixtom Imagen 1 READY %.0fs" % (time.time()-t0))
