import base64
import json
import os
import subprocess
import sys
import time

# Render with SDXL (ungated, so no HF token needed) and commit the PNGs straight
# into a GitHub repo through the contents API. Every file host tried so far
# rejected anonymous uploads from this network, and the notebook has no way to
# expose files directly, so the repo is the reliable path out.

t0 = time.time()


def note(m):
    print(str(m)[:300], flush=True)


GH_TOKEN = os.environ.get("GH_TOKEN", "")
GH_REPO = os.environ.get("GH_REPO", "Setyoanugrah/boxx1")
GH_DIR = os.environ.get("GH_DIR", "renders")

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
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
note("model loaded %.0fs" % (time.time() - t1))

OUT = "/tmp/sdxl_out"
os.makedirs(OUT, exist_ok=True)

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


def gh_put(path, data_bytes, message):
    """Create or update a file in the repo via the contents API."""
    import urllib.error
    import urllib.request

    url = "https://api.github.com/repos/%s/contents/%s" % (GH_REPO, path)
    headers = {"Authorization": "Bearer " + GH_TOKEN,
               "Accept": "application/vnd.github+json",
               "User-Agent": "molab-sdxl"}
    sha = None
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as f:
            sha = json.loads(f.read()).get("sha")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    body = {"message": message,
            "content": base64.b64encode(data_bytes).decode()}
    if sha:
        body["sha"] = sha
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={**headers, "Content-Type": "application/json"},
                                 method="PUT")
    with urllib.request.urlopen(req, timeout=120) as f:
        return json.loads(f.read())


urls = []
for name, seed, prompt, neg in JOBS:
    t = time.time()
    img = pipe(
        prompt=prompt,
        negative_prompt=neg,
        height=1024,
        width=1024,
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    path = os.path.join(OUT, name + ".png")
    img.save(path)
    note("rendered %s in %.1fs -> %d bytes" % (name, time.time() - t, os.path.getsize(path)))

    if not GH_TOKEN:
        note("no GH_TOKEN, skipping upload")
        continue
    stamp = time.strftime("%Y%m%d-%H%M%S")
    remote = "%s/%s-%s.png" % (GH_DIR, name, stamp)
    try:
        res = gh_put(remote, open(path, "rb").read(), "render: %s" % name)
        html = (res.get("content") or {}).get("html_url", "")
        raw = ("https://raw.githubusercontent.com/%s/main/%s" % (GH_REPO, remote))
        note("committed %s" % html[:110])
        urls.append((name, raw))
    except Exception as exc:  # noqa: BLE001
        note("commit %s failed: %s %s" % (name, type(exc).__name__, str(exc)[:90]))

note("IMAGES " + json.dumps(urls))
note("done %.0fs" % (time.time() - t0))
