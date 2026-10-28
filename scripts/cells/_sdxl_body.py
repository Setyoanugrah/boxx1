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
note("imgbb key present: %s" % bool(KEY))

note("installing")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers", "accelerate", "safetensors", "transformers",
                "sentencepiece", "protobuf", "huggingface_hub", "Pillow"],
               capture_output=True)
note("pip done %.0fs" % (time.time() - t0))

import torch
note("cuda=%s %s" % (torch.cuda.is_available(), torch.cuda.get_device_name(0)))

from diffusers import StableDiffusionXLPipeline

t1 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
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
    with urllib.request.urlopen(req, timeout=180) as f:
        d = json.loads(f.read())
    return d["data"]["url"]


JOBS = [
    ("photo", 7,
     "close-up portrait photograph of a woman with freckles in her early thirties, "
     "soft window light from the left, 85mm lens, shallow depth of field, natural "
     "skin texture, photorealistic, editorial photography, highly detailed",
     "cartoon, anime, painting, illustration, lowres, blurry, deformed, watermark"),
    ("art", 42,
     "oil painting of a lone lighthouse on a cliff at dusk, thick impasto brush "
     "strokes, moody teal and amber palette, dramatic storm clouds, painterly",
     "photo, photorealistic, lowres, blurry, watermark, text"),
]

urls = []
for name, seed, prompt, neg in JOBS:
    t = time.time()
    img = pipe(prompt=prompt, negative_prompt=neg, height=1024, width=1024,
               num_inference_steps=30, guidance_scale=7.0,
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
