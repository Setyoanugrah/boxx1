import base64, json, os, subprocess, sys, time, urllib.request

t0 = time.time()
def note(m): print(str(m)[:300], flush=True)

MODEL_NAME = "Trixtom Imagen 1"
TRIGGER = "rikustyle"

note("installing")
subprocess.run([sys.executable,"-m","pip","install","-q","diffusers","accelerate",
                "safetensors","transformers","peft","Pillow","huggingface_hub"], capture_output=True)
note("pip %.0fs" % (time.time()-t0))

import torch
note("cuda=%s %s" % (torch.cuda.is_available(), torch.cuda.get_device_name(0)))

# pull the 24 captioned images from the repo, in parts
BASE_URL = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/data/part%02d.json"
WORK = "/tmp/lora"
IMGS = os.path.join(WORK, "img")
os.makedirs(IMGS, exist_ok=True)

pack = []
for i in range(8):
    try:
        part = json.loads(urllib.request.urlopen(BASE_URL % i, timeout=120).read())
        pack.extend(part)
        note("part%02d: %d rows" % (i, len(part)))
    except Exception:
        break
note("total rows: %d" % len(pack))

kept = 0
for row in pack:
    p = os.path.join(IMGS, row["id"] + ".png")
    try:
        open(p, "wb").write(base64.b64decode(row["b64"]))
        open(os.path.join(IMGS, row["id"] + ".txt"), "w").write(row["caption"])
        kept += 1
    except Exception as exc:
        note("skip %s %s" % (row["id"], type(exc).__name__))
note("images ready: %d" % kept)
if kept < 10:
    note("TOO FEW"); raise SystemExit(1)

from diffusers import StableDiffusionXLPipeline
BASE = "cagliostrolab/animagine-xl-3.1"
note("loading base %s" % BASE)
pipe = StableDiffusionXLPipeline.from_pretrained(BASE, torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")
note("base loaded %.0fs" % (time.time()-t0))

vae, te1, te2 = pipe.vae, pipe.text_encoder, pipe.text_encoder_2
tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
sched, unet = pipe.scheduler, pipe.unet
vae.requires_grad_(False); te1.requires_grad_(False); te2.requires_grad_(False); unet.requires_grad_(False)

from peft import LoraConfig
unet.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                            target_modules=["to_k","to_q","to_v","to_out.0"]))
trainable = [p for p in unet.parameters() if p.requires_grad]
note("trainable %.1fM" % (sum(p.numel() for p in trainable)/1e6))

from PIL import Image
import torch.nn.functional as F

RES = 1024
def prep(path, caption):
    im = Image.open(path).convert("RGB")
    w,h = im.size; s = min(w,h)
    im = im.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)).resize((RES,RES), Image.LANCZOS)
    px = torch.tensor(list(im.getdata()), dtype=torch.float32).reshape(RES,RES,3)/127.5-1.0
    px = px.permute(2,0,1).unsqueeze(0)
    t1 = tok1(caption, padding="max_length", max_length=tok1.model_max_length, truncation=True, return_tensors="pt")
    t2 = tok2(caption, padding="max_length", max_length=tok2.model_max_length, truncation=True, return_tensors="pt")
    return px, t1.input_ids, t2.input_ids

note("pre-encoding")
samples = []
with torch.no_grad():
    for i in range(1, kept+1):
        pid = "ref%02d" % i
        p = os.path.join(IMGS, pid + ".png")
        if not os.path.exists(p): continue
        cap = open(os.path.join(IMGS, pid + ".txt")).read()
        px, ids1, ids2 = prep(p, cap)
        px = px.to("cuda", dtype=torch.float16)
        lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
        e1 = te1(ids1.to("cuda"))[0]
        o2 = te2(ids2.to("cuda"), output_hidden_states=True)
        e2 = o2[0]; pooled = o2.hidden_states[-2][:, -1]
        samples.append({"lat": lat.cpu(), "e1": e1.cpu(), "e2": e2.cpu(), "pool": pooled.cpu()})
note("encoded %d in %.0fs" % (len(samples), time.time()-t0))

STEPS, LR = 1100, 1e-4
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
from diffusers.optimization import get_scheduler
lr_s = get_scheduler("cosine", opt, num_warmup_steps=60, num_training_steps=STEPS)
scaler = torch.amp.GradScaler("cuda")

unet.train()
t_tr = time.time()
for step in range(STEPS):
    s = samples[step % len(samples)]
    lat = s["lat"].to("cuda", dtype=torch.float16)
    e1 = s["e1"].to("cuda", dtype=torch.float16)
    e2 = s["e2"].to("cuda", dtype=torch.float16)
    pool = s["pool"].to("cuda", dtype=torch.float16)
    noise = torch.randn_like(lat)
    ts = torch.randint(0, sched.config.num_train_timesteps, (1,), device="cuda").long()
    noisy = sched.add_noise(lat, noise, ts)
    add = {"text_embeds": pool, "time_ids": torch.tensor([[RES,RES,0,0,RES,RES]], device="cuda", dtype=torch.float16)}
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = unet(noisy, ts, encoder_hidden_states=e1, added_cond_kwargs=add).sample
        loss = F.mse_loss(pred.float(), noise.float())
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); lr_s.step()
    if step % 100 == 0:
        note("step %d/%d loss %.4f %.0fs" % (step, STEPS, loss.item(), time.time()-t_tr))
note("trained %.0fs" % (time.time()-t_tr))

from diffusers.utils import convert_state_dict_to_diffusers
from peft import get_peft_model_state_dict
sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
StableDiffusionXLPipeline.save_lora_weights(WORK, unet_lora_layers=sd)
lora_path = os.path.join(WORK, "pytorch_lora_weights.safetensors")
note("LORA %s %d KB" % (lora_path, os.path.getsize(lora_path)//1024))

# test render with the trigger
pipe.unload_lora_weights()
pipe.load_lora_weights(WORK, adapter_name=TRIGGER)
urls = []
KEY = os.environ.get("IMGBB_KEY","")
import urllib.parse
def imgbb(path, name):
    if not KEY: return ""
    d = base64.b64encode(open(path,"rb").read()).decode()
    b = urllib.parse.urlencode({"image": d, "name": name}).encode()
    r = urllib.request.Request("https://api.imgbb.com/1/upload?key="+KEY, data=b, method="POST")
    with urllib.request.urlopen(r, timeout=240) as f:
        return json.loads(f.read())["data"]["url"]

TESTS = [
 ("t1", 5,  f"{TRIGGER}, 1girl, solo, long red hair, red eyes, black leather jacket, neon city at night, upper body, masterpiece, high score, absurdres, safe"),
 ("t2", 9,  f"{TRIGGER}, 1girl, solo, long black hair, purple eyes, white crop top, bedroom, sitting on bed, soft light, masterpiece, high score, absurdres, safe"),
 ("t3", 13, f"{TRIGGER}, 1girl, solo, silver hair, grey eyes, blue one-piece swimsuit, arms behind head, plain background, masterpiece, high score, absurdres, safe"),
]
for name, seed, prompt in TESTS:
    img = pipe(prompt=prompt,
               negative_prompt="lowres, bad anatomy, bad hands, worst quality, low quality, watermark, text",
               height=1024, width=832, num_inference_steps=28, guidance_scale=6.0,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    p = "/tmp/%s.png" % name
    img.save(p)
    try:
        u = imgbb(p, name); urls.append([name, u]); note("%s -> %s" % (name, u[:80]))
    except Exception as exc:
        note("%s upload fail %s" % (name, type(exc).__name__))

note("MODEL_NAME " + MODEL_NAME)
note("SAMPLES " + json.dumps(urls))
note("done %.0fs" % (time.time()-t0))
