#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/pipeline_8b
echo "=== S3 CS Training (fix retry) ===" > /mnt/d/VORTEX_FLAME/pipeline_8b/s3_train_log.txt
echo "Start: $(date)" >> /mnt/d/VORTEX_FLAME/pipeline_8b/s3_train_log.txt
python train_7b_s3_cs.py >> /mnt/d/VORTEX_FLAME/pipeline_8b/s3_train_log.txt 2>&1
echo "End: $(date)" >> /mnt/d/VORTEX_FLAME/pipeline_8b/s3_train_log.txt
echo "Exit code: $?" >> /mnt/d/VORTEX_FLAME/pipeline_8b/s3_train_log.txt
