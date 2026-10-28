import base64, json, os, subprocess, sys, time
import urllib.parse, urllib.request

t0 = time.time()
def note(m): print(str(m)[:300], flush=True)

KEY = os.environ.get("IMGBB_KEY", "")
note("imgbb key: %s" % bool(KEY))
note("installing")
subprocess.run([sys.executable,"-m","pip","install","-q","diffusers","accelerate",
                "safetensors","transformers","sentencepiece","protobuf",
                "huggingface_hub","Pillow"], capture_output=True)
note("pip done %.0fs" % (time.time()-t0))

import torch
note("cuda=%s %s" % (torch.cuda.is_available(), torch.cuda.get_device_name(0)))
from diffusers import StableDiffusionXLPipeline

t1 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.1", torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
note("model loaded %.0fs" % (time.time()-t1))

def imgbb(path, name):
    if not KEY: return ""
    data = base64.b64encode(open(path,"rb").read()).decode()
    body = urllib.parse.urlencode({"image": data, "name": name}).encode()
    req = urllib.request.Request("https://api.imgbb.com/1/upload?key="+KEY,
                                 data=body, method="POST")
    with urllib.request.urlopen(req, timeout=240) as f:
        return json.loads(f.read())["data"]["url"]

Q = "masterpiece, high score, great score, absurdres, newest, safe"
NEG = ("lowres, bad anatomy, bad hands, text, error, worst quality, low quality, "
       "jpeg artifacts, watermark, signature, blurry, extra digit, missing fingers")

# Dataset spec: every row carries structured labels, so the same file can train
# either a caption model or a tag-conditioned one.
ROWS = [
    # (name, hair, eyes, outfit, background, seed)
    ("d01", "short green hair",  "green eyes",  "yellow tank top, yellow shorts",      "tropical beach, sunset, ocean, palm trees", 101),
    ("d02", "long red hair",     "red eyes",    "black leather jacket, open jacket",   "city at night, neon signs, cyberpunk, rain", 102),
    ("d03", "long black hair",   "purple eyes", "purple ribbed turtleneck",            "modern room, bookshelf, potted plant",      103),
    ("d04", "short blonde hair", "blue eyes",   "white blouse, pleated skirt",         "classroom, window light, desks",            104),
    ("d05", "long silver hair",  "gold eyes",   "black dress, choker",                 "moonlit rooftop, city skyline, night",      105),
    ("d06", "twin tails, brown hair", "amber eyes", "red hoodie, denim shorts",        "forest path, sunlight through trees",       106),
    ("d07", "long pink hair",    "violet eyes", "oversized white sweater",             "cafe interior, warm lamps, rain window",    107),
    ("d08", "short black hair",  "grey eyes",   "grey suit, tie",                      "office, glass windows, daylight",           108),
    ("d09", "long blue hair",    "cyan eyes",   "blue swimsuit",                       "poolside, summer, bright sun",              109),
    ("d10", "wavy auburn hair",  "hazel eyes",  "cream knit cardigan, long skirt",     "autumn park, fallen leaves, golden light",  110),
    ("d11", "long purple hair",  "pink eyes",   "black corset, lace gloves",           "gothic cathedral, stained glass",           111),
    ("d12", "short teal hair",   "orange eyes", "orange bomber jacket",                "train station platform, dusk",              112),
]

manifest = []
for name, hair, eyes, outfit, bg, seed in ROWS:
    tags = f"1girl, solo, {hair}, {eyes}, {outfit}, {bg}, looking at viewer, upper body, detailed background"
    prompt = f"{tags}, {Q}"
    t = time.time()
    img = pipe(prompt=prompt, negative_prompt=NEG, height=1024, width=1024,
               num_inference_steps=26, guidance_scale=6.0,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    path = "/tmp/%s.png" % name
    img.save(path)
    url = ""
    try:
        url = imgbb(path, name)
    except Exception as exc:
        note("imgbb %s failed %s" % (name, type(exc).__name__))
    note("%s %.1fs %s" % (name, time.time()-t, url[:80]))
    manifest.append({
        "id": name,
        "image": url,
        "seed": seed,
        "labels": {"hair": hair, "eyes": eyes, "outfit": outfit, "background": bg,
                   "style": "anime illustration, glossy rendering, detailed background"},
        "caption": tags,
        "prompt": prompt,
        "negative": NEG,
        "size": [1024, 1024],
    })

note("MANIFEST " + json.dumps(manifest))
note("count %d done %.0fs" % (len(manifest), time.time()-t0))
