import json, os, subprocess, sys, time, urllib.request

t0 = time.time()
def note(m): print(str(m)[:300], flush=True)

# Train a LoRA on Animagine XL 3.1 from the seed set:
#   7 hand-picked reference images + 36 generated variations, all captioned.
# Everything runs on the molab GPU; nothing touches the phone.

DATASET_URL = "https://raw.githubusercontent.com/Setyoanugrah/boxx1/main/scripts/dataset_pack.json"

note("installing")
subprocess.run([sys.executable,"-m","pip","install","-q","diffusers","accelerate",
                "safetensors","transformers","peft","datasets","Pillow",
                "huggingface_hub","torchvision","bitsandbytes"], capture_output=True)
note("pip %.0fs" % (time.time()-t0))

import torch
note("cuda=%s %s" % (torch.cuda.is_available(), torch.cuda.get_device_name(0)))
note("vram %.0f GB" % (torch.cuda.get_device_properties(0).total_memory/1073741824))

pack = json.loads(urllib.request.urlopen(DATASET_URL, timeout=120).read())
note("dataset rows: %d" % len(pack))

WORK = "/tmp/lora"
IMGS = os.path.join(WORK, "img")
os.makedirs(IMGS, exist_ok=True)

# download every image and write its caption next to it
kept = []
for i, row in enumerate(pack):
    url = row.get("image") or ""
    if not url.startswith("http"):
        continue
    p = os.path.join(IMGS, "%03d.png" % i)
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
        open(p, "wb").write(data)
        open(os.path.join(IMGS, "%03d.txt" % i), "w").write(row["caption"])
        kept.append(p)
    except Exception as exc:
        note("skip %d %s" % (i, type(exc).__name__))
note("images ready: %d" % len(kept))

if len(kept) < 10:
    note("TOO FEW IMAGES, aborting")
    raise SystemExit(1)

from diffusers import StableDiffusionXLPipeline
from diffusers.optimization import get_scheduler

BASE = "cagliostrolab/animagine-xl-3.1"
note("loading base")
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE, torch_dtype=torch.float16, use_safetensors=True, variant=None)
pipe.to("cuda")
note("base loaded %.0fs" % (time.time()-t0))

from diffusers import AutoencoderKL
vae = pipe.vae
text_encoder = pipe.text_encoder
text_encoder_2 = pipe.text_encoder_2
tokenizer = pipe.tokenizer
tokenizer_2 = pipe.tokenizer_2
noise_scheduler = pipe.scheduler
unet = pipe.unet

# freeze everything except the LoRA adapters
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
unet.requires_grad_(False)

from peft import LoraConfig
lora = LoraConfig(
    r=16, lora_alpha=16,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)
unet.add_adapter(lora)
trainable = [p for p in unet.parameters() if p.requires_grad]
note("trainable params: %.1fM" % (sum(p.numel() for p in trainable)/1e6))

from PIL import Image
import torch.nn.functional as F

RES = 1024
def prep(path, caption):
    im = Image.open(path).convert("RGB")
    # center crop to square, then resize
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((RES, RES), Image.LANCZOS)
    px = torch.tensor(list(im.getdata()), dtype=torch.float32).reshape(RES, RES, 3)
    px = px / 127.5 - 1.0
    px = px.permute(2, 0, 1).unsqueeze(0)
    tok = tokenizer(caption, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
    tok2 = tokenizer_2(caption, padding="max_length", max_length=tokenizer_2.model_max_length,
                       truncation=True, return_tensors="pt")
    return px, tok.input_ids, tok2.input_ids

# pre-encode latents and text embeddings (SDXL has two text encoders)
note("pre-encoding %d images" % len(kept))
samples = []
with torch.no_grad():
    for i, p in enumerate(kept):
        cap = open(p.replace(".png", ".txt")).read()
        px, ids, ids2 = prep(p, cap)
        px = px.to("cuda", dtype=torch.float16)
        lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
        emb = text_encoder(ids.to("cuda"))[0]
        emb2 = text_encoder_2(ids2.to("cuda"))[0]
        pooled = text_encoder_2(ids2.to("cuda"), output_hidden_states=True).hidden_states[-2][:, -1]
        samples.append({"lat": lat.cpu(), "emb": emb.cpu(), "emb2": emb2.cpu(), "pooled": pooled.cpu()})
        if i % 10 == 0:
            note("  encoded %d/%d" % (i, len(kept)))
note("encode done %.0fs" % (time.time()-t0))

STEPS = 900
LR = 1e-4
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
sched = get_scheduler("cosine", opt, num_warmup_steps=50, num_training_steps=STEPS)
scaler = torch.amp.GradScaler("cuda")

unet.train()
t_train = time.time()
for step in range(STEPS):
    s = samples[step % len(samples)]
    lat = s["lat"].to("cuda", dtype=torch.float16)
    emb = s["emb"].to("cuda", dtype=torch.float16)
    emb2 = s["emb2"].to("cuda", dtype=torch.float16)
    pooled = s["pooled"].to("cuda", dtype=torch.float16)

    noise = torch.randn_like(lat)
    ts = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda").long()
    noisy = noise_scheduler.add_noise(lat, noise, ts)
    add_kwargs = {"text_embeds": pooled, "time_ids": torch.tensor(
        [[RES, RES, 0, 0, RES, RES]], device="cuda", dtype=torch.float16)}

    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = unet(noisy, ts, encoder_hidden_states=emb, added_cond_kwargs=add_kwargs).sample
        loss = F.mse_loss(pred.float(), noise.float())

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    sched.step()

    if step % 50 == 0:
        el = time.time() - t_train
        note("step %d/%d loss %.4f  %.1fs" % (step, STEPS, loss.item(), el))

note("training done %.0fs total" % (time.time()-t0))

from diffusers.utils import convert_state_dict_to_diffusers
from peft import get_peft_model_state_dict
sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
StableDiffusionXLPipeline.save_lora_weights(WORK, unet_lora_layers=sd)
out = os.path.join(WORK, "pytorch_lora_weights.safetensors")
note("LORA_SAVED %s %d bytes" % (out, os.path.getsize(out)))

# sample render with the LoRA
pipe.unload_lora_weights()
pipe.load_lora_weights(WORK)
for name, seed in (("sample1", 5), ("sample2", 9)):
    img = pipe(prompt=("rikustyle, 1girl, solo, long red hair, red eyes, black leather jacket, "
                       "neon city at night, upper body, masterpiece, high score, absurdres, safe"),
               negative_prompt="lowres, bad anatomy, worst quality, low quality, watermark",
               height=1024, width=832, num_inference_steps=26, guidance_scale=6.0,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    p = "/tmp/%s.png" % name
    img.save(p)
    note("sample %s %d bytes" % (name, os.path.getsize(p)))

note("done %.0fs" % (time.time()-t0))
