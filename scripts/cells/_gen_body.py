import base64, json, os, random, subprocess, sys, time
import urllib.parse, urllib.request

t0 = time.time()
def note(m): print(str(m)[:300], flush=True)

KEY = os.environ.get("IMGBB_KEY", "")
note("imgbb key: %s" % bool(KEY))
subprocess.run([sys.executable,"-m","pip","install","-q","diffusers","accelerate",
                "safetensors","transformers","sentencepiece","protobuf",
                "huggingface_hub","Pillow"], capture_output=True)
note("pip %.0fs" % (time.time()-t0))

import torch
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.1", torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
note("model %.0fs" % (time.time()-t0))

def imgbb(path, name):
    data = base64.b64encode(open(path,"rb").read()).decode()
    body = urllib.parse.urlencode({"image": data, "name": name}).encode()
    req = urllib.request.Request("https://api.imgbb.com/1/upload?key="+KEY,
                                 data=body, method="POST")
    with urllib.request.urlopen(req, timeout=240) as f:
        return json.loads(f.read())["data"]["url"]

# Style DNA taken from the 7 reference images: one anime girl, glossy rendering,
# detailed background, warm/cool cinematic light, upper body framing.
HAIR = ["short green hair","long red hair","long black hair","short blue hair",
        "long pink hair","dark purple long hair","silver white long hair",
        "brown twin tails","blonde short hair","teal short hair",
        "auburn wavy hair","long blue hair"]
EYES = ["green eyes","red eyes","purple eyes","blue eyes","pink eyes",
        "violet eyes","amber eyes","cyan eyes"]
OUTFIT = ["yellow tank top, yellow shorts","black leather jacket, open jacket",
          "purple ribbed turtleneck","white crop top, navy miniskirt",
          "glossy black slip dress","cyan futuristic bodysuit, glowing lines",
          "deep purple glossy dress, black lace","red hoodie, denim shorts",
          "white blouse, pleated skirt","black corset, lace gloves"]
BG = ["tropical beach, sunset, ocean, palm trees","city at night, neon signs, cyberpunk",
      "modern bedroom, window, warm lamp","sci-fi control room, holographic displays",
      "upside down cityscape, rain, reflections","autumn park, fallen leaves, golden light",
      "cafe interior, rain on window, warm lamps","moonlit rooftop, city skyline"]
Q = "masterpiece, high score, great score, absurdres, newest, safe"
NEG = ("lowres, bad anatomy, bad hands, text, error, worst quality, low quality, "
       "jpeg artifacts, watermark, signature, blurry, extra digit, missing fingers, "
       "extra limbs, deformed")

N = 36
random.seed(20261028)
out = []
for i in range(N):
    hair = random.choice(HAIR); eyes = random.choice(EYES)
    outfit = random.choice(OUTFIT); bg = random.choice(BG)
    tags = (f"1girl, solo, {hair}, {eyes}, {outfit}, upper body, looking at viewer, "
            f"{bg}, glossy skin, detailed background, cinematic lighting, {Q}")
    seed = random.randint(1, 2**31-1)
    t = time.time()
    img = pipe(prompt=tags, negative_prompt=NEG, height=832, width=1216,
               num_inference_steps=24, guidance_scale=6.0,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    p = "/tmp/gen%02d.png" % i
    img.save(p)
    url = ""
    try:
        url = imgbb(p, "gen%02d" % i)
    except Exception as exc:
        note("imgbb %d fail %s" % (i, type(exc).__name__))
    note("gen%02d %.1fs %s" % (i, time.time()-t, url[-30:]))
    out.append({"id":"gen%02d"%i, "image":url, "seed":seed,
                "labels":{"hair":hair,"eyes":eyes,"outfit":outfit,"background":bg,
                          "style":"anime illustration, glossy rendering, detailed background"},
                "caption":tags, "trigger":"rikustyle"})

note("MANIFEST " + json.dumps(out))
note("count %d done %.0fs" % (len(out), time.time()-t0))
