#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame

python3 << 'PYEOF'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
    print(f"Device 1: {torch.cuda.get_device_name(1)}")
    t = torch.randn(100, 100).cuda()
    print(f"Tensor test: {t.mean().item():.4f}")
    print(f"GPU mem: {torch.cuda.memory_allocated()/1024/1024:.1f} MB")
else:
    print("ERROR: CUDA not available!")
PYEOF
