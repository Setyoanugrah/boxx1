import json
import os
import subprocess
import sys
import time

# Render with FLUX.1-schnell (Apache-2.0, ungated) and push the PNGs to a file
# host so they can be pulled back outside the notebook.
#
# schnell is used instead of dev on purpose: dev is licence-gated on Hugging
# Face and needs an accepted agreement plus a token, which this notebook does
# not have. schnell needs 4 steps instead of 28, so the first result lands in
# seconds rather than minutes.

t0 = time.time()


def note(m):
    print(str(m)[:300], flush=True)


note("installing")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers", "accelerate", "safetensors", "transformers",
                "sentencepiece", "protobuf", "huggingface_hub", "Pillow"],
               capture_output=True)
note("pip done %.0fs" % (time.time() - t0))

import torch
note("cuda=%s %s" % (torch.cuda.is_available(), torch.cuda.get_device_name(0)))

from diffusers import FluxPipeline

t1 = time.time()
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
note("model loaded %.0fs" % (time.time() - t1))

OUT = "/tmp/flux_out"
os.makedirs(OUT, exist_ok=True)

JOBS = [
    ("photo", 7,
     "close-up portrait photograph of a woman with freckles in her early thirties, "
     "soft window light from the left, 85mm lens, shallow depth of field, natural "
     "skin texture with visible pores, subtle film grain, photorealistic, "
     "highly detailed, editorial photography"),
    ("art", 42,
     "oil painting of a lone lighthouse on a cliff at dusk, thick impasto brush "
     "strokes, moody teal and amber palette, dramatic storm clouds, painterly "
     "texture, romantic landscape"),
]

urls = []
for name, seed, prompt in JOBS:
    t = time.time()
    img = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    path = os.path.join(OUT, name + ".png")
    img.save(path)
    note("rendered %s in %.1fs -> %d bytes" % (name, time.time() - t, os.path.getsize(path)))

    # upload to catbox
    r = subprocess.run(
        ["curl", "-s", "-F", "reqtype=fileupload",
         "-F", "fileToUpload=@" + path, "https://catbox.moe/user/api.php"],
        capture_output=True, text=True, timeout=180)
    url = (r.stdout or "").strip()
    note("upload %s -> %s" % (name, url[:120]))
    if url.startswith("http"):
        urls.append((name, url))

note("IMAGES " + json.dumps(urls))
note("done %.0fs" % (time.time() - t0))
