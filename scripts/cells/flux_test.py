import json
import os
import subprocess
import sys
import time

# Install/refresh the image stack, then render a couple of test images with
# Flux so quality can be judged before committing to a LoRA training run.

LOG = []


def note(m):
    LOG.append(str(m)[:400])
    print(str(m)[:400], flush=True)


note("start")
t0 = time.time()

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers", "accelerate", "safetensors", "transformers",
                "sentencepiece", "protobuf", "huggingface_hub"],
               capture_output=True)
note("pip done in %.0fs" % (time.time() - t0))

import torch
note("torch %s cuda=%s" % (torch.__version__, torch.cuda.is_available()))
note("gpu %s" % torch.cuda.get_device_name(0))
note("vram %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1073741824))

from diffusers import FluxPipeline

t1 = time.time()
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
note("flux loaded in %.0fs" % (time.time() - t1))

OUT = "/tmp/flux_out"
os.makedirs(OUT, exist_ok=True)

JOBS = [
    ("photo", 0,
     "close-up portrait photograph of a woman with freckles, soft window light, "
     "85mm lens, shallow depth of field, natural skin texture, film grain, "
     "photorealistic, highly detailed"),
    ("art", 1,
     "oil painting of a lone lighthouse on a cliff at dusk, thick impasto strokes, "
     "moody teal and amber palette, dramatic sky, painterly texture"),
]

for name, seed, prompt in JOBS:
    t = time.time()
    img = pipe(
        prompt=prompt,
        height=1024, width=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    path = os.path.join(OUT, name + ".png")
    img.save(path)
    note("rendered %s in %.1fs -> %s (%d bytes)"
         % (name, time.time() - t, path, os.path.getsize(path)))

note("done total %.0fs" % (time.time() - t0))
note("FILES " + json.dumps(sorted(os.listdir(OUT))))
