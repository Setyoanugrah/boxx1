import json
import os
import subprocess
import sys
import time

# Render with a genuinely ungated model.
#
# FLUX.1-dev and FLUX.1-schnell are both gated on Hugging Face: they need a
# logged-in account that has accepted the licence, so from a bare notebook the
# download returns 401. SDXL base is not gated, so it works with no token at
# all - which is what this cell proves end to end.

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

    # upload: try a few hosts, keep whichever returns a URL
    url = ""
    for host, cmd in (
        ("0x0.st", ["curl", "-s", "-F", "file=@" + path, "https://0x0.st"]),
        ("litterbox", ["curl", "-s", "-F", "reqtype=fileupload",
                       "-F", "time=1h", "-F", "fileToUpload=@" + path,
                       "https://litterbox.catbox.moe/resources/internals/api.php"]),
        ("tmpfiles", ["curl", "-s", "-F", "file=@" + path, "https://tmpfiles.org/api/v1/upload"]),
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            out = (r.stdout or "").strip()
            if out.startswith("http"):
                url = out.split()[0]
                note("upload %s via %s -> %s" % (name, host, url[:110]))
                break
            note("  %s said: %s" % (host, out[:90]))
        except Exception as exc:  # noqa: BLE001
            note("  %s failed: %s" % (host, type(exc).__name__))
    if url:
        urls.append((name, url))

note("IMAGES " + json.dumps(urls))
note("done %.0fs" % (time.time() - t0))
