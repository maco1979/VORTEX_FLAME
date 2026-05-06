#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/pipeline_8b
nohup python quick_test_7b_s3.py > /mnt/d/VORTEX_FLAME/pipeline_8b/quick_test_log.txt 2>&1 &
echo "PID: $!" > /mnt/d/VORTEX_FLAME/pipeline_8b/quick_test_pid.txt
echo "Started: $(date)" >> /mnt/d/VORTEX_FLAME/pipeline_8b/quick_test_pid.txt
echo "Quick test PID: $!"
