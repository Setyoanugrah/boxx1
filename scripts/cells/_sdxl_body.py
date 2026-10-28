import base64
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

t0 = time.time()


def note(m):
    print(str(m)[:300], flush=True)


KEY = os.environ.get("IMGBB_KEY", "")
note("imgbb key: %s" % bool(KEY))

note("installing")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers", "accelerate", "safetensors", "transformers",
                "sentencepiece", "protobuf", "huggingface_hub", "Pillow"],
               capture_output=True)
note("pip done %.0fs" % (time.time() - t0))

import torch
note("cuda=%s %s" % (torch.cuda.is_available(), torch.cuda.get_device_name(0)))

from diffusers import StableDiffusionXLPipeline

# Animagine XL 3.1: an anime-tuned SDXL that is not gated on Hugging Face.
# SDXL base renders "anime" as a western cartoon, so it cannot match the
# reference style; this checkpoint is trained on anime illustration.
MODEL = "cagliostrolab/animagine-xl-3.1"

t1 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
note("model loaded %.0fs" % (time.time() - t1))


def imgbb(path, name):
    if not KEY:
        return ""
    data = base64.b64encode(open(path, "rb").read()).decode()
    body = urllib.parse.urlencode({"image": data, "name": name}).encode()
    req = urllib.request.Request(
        "https://api.imgbb.com/1/upload?key=" + KEY, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=240) as f:
        d = json.loads(f.read())
    return d["data"]["url"]


Q = "masterpiece, high score, great score, absurdres, newest, safe"

JOBS = [
    ("beach", 11,
     "1girl, solo, short green hair, green eyes, yellow tank top, yellow shorts, "
     "wet skin, glossy skin, smile, looking at viewer, medium breasts, "
     "tropical beach, sunset, ocean, palm trees, rocks, golden hour, "
     "lens flare, vivid colors, " + Q,
     "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, "
     "jpeg artifacts, watermark, signature, blurry, extra digit"),
    ("neon", 27,
     "1girl, solo, long red hair, red eyes, black leather jacket, open jacket, "
     "cleavage, looking at viewer, smirk, city at night, neon lights, "
     "neon signs, cyberpunk, japanese text, bar, rain, reflections, "
     "cinematic lighting, " + Q,
     "lowres, bad anatomy, bad hands, text, error, worst quality, low quality, "
     "jpeg artifacts, watermark, signature, blurry, extra digit"),
]

urls = []
for name, seed, prompt, neg in JOBS:
    t = time.time()
    img = pipe(prompt=prompt, negative_prompt=neg, height=1024, width=1024,
               num_inference_steps=28, guidance_scale=6.0,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    path = "/tmp/%s.png" % name
    img.save(path)
    note("rendered %s %.1fs %d bytes" % (name, time.time() - t, os.path.getsize(path)))
    try:
        u = imgbb(path, name)
        note("imgbb %s -> %s" % (name, u[:110]))
        if u:
            urls.append((name, u))
    except Exception as exc:
        note("imgbb %s failed: %s %s" % (name, type(exc).__name__, str(exc)[:120]))

note("IMAGES " + json.dumps(urls))
note("done %.0fs" % (time.time() - t0))
