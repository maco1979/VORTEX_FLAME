#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/pipeline_8b
python quick_test_7b_s3.py 2>&1
