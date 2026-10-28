import torch, os, subprocess
print("CUDA", torch.cuda.is_available())
if torch.cuda.is_available():
    print("DEV", torch.cuda.get_device_name(0))
    print("VRAM_GB", round(torch.cuda.get_device_properties(0).total_memory/1e9,1))
print("CPU", os.cpu_count())
print("RAM_GB", round(os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES")/1e9,1))
