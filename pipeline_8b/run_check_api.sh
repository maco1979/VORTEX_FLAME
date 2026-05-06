#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
python /mnt/d/VORTEX_FLAME/pipeline_8b/check_sft_api.py
