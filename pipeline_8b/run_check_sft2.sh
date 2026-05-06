#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
python -c "
from trl import SFTConfig
import inspect
sig = inspect.signature(SFTConfig.__init__)
params = [p for p in sig.parameters.keys()]
for p in params:
    if any(k in p.lower() for k in ['seq','max','length','dataset','pack']):
        print(p)
" 2>&1 | tee /mnt/d/VORTEX_FLAME/pipeline_8b/sft_params.txt
