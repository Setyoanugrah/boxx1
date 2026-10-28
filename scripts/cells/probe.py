import json
import os
import shutil
import subprocess

out = {}
out["python"] = __import__("sys").version.split()[0]
try:
    out["gpu_nvidia_smi"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e:
    out["gpu_nvidia_smi"] = "ERR " + repr(e)[:120]

try:
    out["cuda_visible"] = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    out["cpu_count"] = os.cpu_count()
    mi = open("/proc/meminfo").read()
    import re
    out["mem_gb"] = round(int(re.search(r"MemTotal:\s+(\d+)", mi).group(1)) / 1048576, 1)
    du = shutil.disk_usage("/")
    out["disk_free_gb"] = round(du.free / 1073741824, 1)
    out["disk_total_gb"] = round(du.total / 1073741824, 1)
except Exception as e:
    out["sys_err"] = repr(e)[:120]

try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
        out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1073741824, 1)
        out["bf16"] = torch.cuda.is_bf16_supported()
except Exception as e:
    out["torch_err"] = repr(e)[:160]

try:
    import diffusers
    out["diffusers"] = diffusers.__version__
except Exception:
    out["diffusers"] = "not installed"

print("PROBE " + json.dumps(out, indent=1))
