#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/cezanne_retrain
python train_s2.py 2>&1 | tee /mnt/d/VORTEX_FLAME/cezanne_retrain/logs/s2_foreground.log
