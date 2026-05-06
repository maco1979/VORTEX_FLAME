#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/cezanne_retrain
python diagnostic_single.py S3_v2 2>&1 | tee /mnt/d/VORTEX_FLAME/cezanne_retrain/logs/diag_S3_v2.log
