import base64, json, os, subprocess, sys, time, urllib.request

# Launch training as a detached background process inside the notebook runtime,
# then return immediately. The cell finishes in seconds, so the Playwright
# driver can disconnect without killing the work. Progress goes to a log file
# that later cells can read.

t0 = time.time()
def note(m): print(str(m)[:300], flush=True)

LOG = "/tmp/train.log"
DONE = "/tmp/train.done"
for f in (LOG, DONE):
    try: os.remove(f)
    except FileNotFoundError: pass

WORKER = r"""
import base64, json, os, subprocess, sys, time, urllib.request
import torch, torch.nn.functional as F
from PIL import Image

def log(m):
    line = time.strftime("%H:%M:%S") + " " + str(m)[:300]
    print(line, flush=True)
    with open("/tmp/train.log", "a") as f:
        f.write(line + "\n")

try:
    log("worker start")
    subprocess.run([sys.executable,"-m","pip","install","-q","diffusers","accelerate",
                    "safetensors","transformers","peft","Pillow","huggingface_hub"], capture_output=True)
    log("pip done cuda=%s" % torch.cuda.is_available())

    BASE_URL = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/data/part%02d.json"
    WORK = "/tmp/lora"; IMGS = os.path.join(WORK, "img")
    os.makedirs(IMGS, exist_ok=True)
    pack = []
    for i in range(8):
        try:
            pack.extend(json.loads(urllib.request.urlopen(BASE_URL % i, timeout=120).read()))
        except Exception:
            break
    log("rows %d" % len(pack))
    kept = 0
    for row in pack:
        p = os.path.join(IMGS, row["id"] + ".png")
        open(p, "wb").write(base64.b64decode(row["b64"]))
        open(os.path.join(IMGS, row["id"] + ".txt"), "w").write(row["caption"])
        kept += 1
    log("images %d" % kept)

    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1", torch_dtype=torch.float16, use_safetensors=True)
    pipe.to("cuda")
    log("base loaded")
    vae, te1, te2 = pipe.vae, pipe.text_encoder, pipe.text_encoder_2
    tok1, tok2 = pipe.tokenizer, pipe.tokenizer_2
    sched, unet = pipe.scheduler, pipe.unet
    for m in (vae, te1, te2, unet): m.requires_grad_(False)

    from peft import LoraConfig
    unet.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian",
                                target_modules=["to_k","to_q","to_v","to_out.0"]))
    tr = [p for p in unet.parameters() if p.requires_grad]
    log("trainable %.1fM" % (sum(p.numel() for p in tr)/1e6))

    RES = 1024
    samples = []
    with torch.no_grad():
        for pid in sorted({os.path.splitext(f)[0] for f in os.listdir(IMGS) if f.endswith(".png")}):
            im = Image.open(os.path.join(IMGS, pid + ".png")).convert("RGB")
            w,h = im.size; s=min(w,h)
            im = im.crop(((w-s)//2,(h-s)//2,(w+s)//2,(h+s)//2)).resize((RES,RES), Image.LANCZOS)
            px = torch.tensor(list(im.getdata()), dtype=torch.float32).reshape(RES,RES,3)/127.5-1.0
            px = px.permute(2,0,1).unsqueeze(0).to("cuda", dtype=torch.float16)
            cap = open(os.path.join(IMGS, pid + ".txt")).read()
            t1 = tok1(cap, padding="max_length", max_length=tok1.model_max_length, truncation=True, return_tensors="pt")
            t2 = tok2(cap, padding="max_length", max_length=tok2.model_max_length, truncation=True, return_tensors="pt")
            lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            e1 = te1(t1.input_ids.to("cuda"))[0]
            o2 = te2(t2.input_ids.to("cuda"), output_hidden_states=True)
            samples.append({"lat": lat.cpu(), "e1": e1.cpu(), "e2": o2[0].cpu(),
                            "pool": o2.hidden_states[-2][:, -1].cpu()})
    log("encoded %d" % len(samples))

    STEPS, LR = 1100, 1e-4
    opt = torch.optim.AdamW(tr, lr=LR, weight_decay=1e-4)
    from diffusers.optimization import get_scheduler
    lr_s = get_scheduler("cosine", opt, num_warmup_steps=60, num_training_steps=STEPS)
    sc = torch.amp.GradScaler("cuda")
    unet.train()
    t_tr = time.time()
    for step in range(STEPS):
        s = samples[step % len(samples)]
        lat = s["lat"].to("cuda", dtype=torch.float16)
        enc = torch.cat([s["e1"], s["e2"]], dim=-1).to("cuda", dtype=torch.float16)
        pool = s["pool"].to("cuda", dtype=torch.float16)
        noise = torch.randn_like(lat)
        ts = torch.randint(0, sched.config.num_train_timesteps, (1,), device="cuda").long()
        noisy = sched.add_noise(lat, noise, ts)
        add = {"text_embeds": pool, "time_ids": torch.tensor([[RES,RES,0,0,RES,RES]], device="cuda", dtype=torch.float16)}
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = unet(noisy, ts, encoder_hidden_states=enc, added_cond_kwargs=add).sample
            loss = F.mse_loss(pred.float(), noise.float())
        opt.zero_grad(set_to_none=True)
        sc.scale(loss).backward(); sc.step(opt); sc.update(); lr_s.step()
        if step % 100 == 0:
            log("step %d/%d loss %.4f %.0fs" % (step, STEPS, loss.item(), time.time()-t_tr))
    log("trained %.0fs" % (time.time()-t_tr))

    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import get_peft_model_state_dict
    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(WORK, unet_lora_layers=sd)
    lp = os.path.join(WORK, "pytorch_lora_weights.safetensors")
    log("LORA %d KB" % (os.path.getsize(lp)//1024))

    pipe.unload_lora_weights(); pipe.load_lora_weights(WORK)
    KEY = os.environ.get("IMGBB_KEY","")
    import urllib.parse
    def imgbb(p, name):
        d = base64.b64encode(open(p,"rb").read()).decode()
        b = urllib.parse.urlencode({"image": d, "name": name}).encode()
        r = urllib.request.Request("https://api.imgbb.com/1/upload?key="+KEY, data=b, method="POST")
        with urllib.request.urlopen(r, timeout=240) as f:
            return json.loads(f.read())["data"]["url"]
    TESTS = [
      ("t1", 5,  "rikustyle, 1girl, solo, long red hair, red eyes, black leather jacket, neon city at night, upper body, masterpiece, high score, absurdres, safe"),
      ("t2", 9,  "rikustyle, 1girl, solo, long black hair, purple eyes, white crop top, bedroom, sitting on bed, soft light, masterpiece, high score, absurdres, safe"),
      ("t3", 13, "rikustyle, 1girl, solo, silver hair, grey eyes, blue one-piece swimsuit, arms behind head, plain background, masterpiece, high score, absurdres, safe"),
    ]
    for name, seed, prompt in TESTS:
        img = pipe(prompt=prompt,
                   negative_prompt="lowres, bad anatomy, bad hands, worst quality, low quality, watermark, text",
                   height=1024, width=832, num_inference_steps=28, guidance_scale=6.0,
                   generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        p = "/tmp/%s.png" % name
        img.save(p)
        try:
            log("SAMPLE %s %s" % (name, imgbb(p, name)))
        except Exception as e:
            log("SAMPLE %s FAIL %s" % (name, type(e).__name__))
    log("MODEL Trixtom Imagen 1")
    log("ALLDONE")
except Exception as e:
    import traceback
    log("FATAL " + repr(e)[:200])
    log(traceback.format_exc()[-1500:])
finally:
    open("/tmp/train.done", "w").write("1")
"""

open("/tmp/train_worker.py", "w").write(WORKER)
env = dict(os.environ)
env["IMGBB_KEY"] = os.environ.get("IMGBB_KEY", "")
p = subprocess.Popen([sys.executable, "-u", "/tmp/train_worker.py"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True, env=env)
note("worker launched pid %d" % p.pid)
note("this cell returns immediately; poll /tmp/train.log")
note("done %.1fs" % (time.time()-t0))
