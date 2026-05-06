#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/pipeline_8b
echo "=== Arena 7B vs 8B ===" > /mnt/d/VORTEX_FLAME/pipeline_8b/arena_log.txt
echo "Start: $(date)" >> /mnt/d/VORTEX_FLAME/pipeline_8b/arena_log.txt
python arena_7b_vs_8b.py >> /mnt/d/VORTEX_FLAME/pipeline_8b/arena_log.txt 2>&1
echo "End: $(date)" >> /mnt/d/VORTEX_FLAME/pipeline_8b/arena_log.txt
echo "Exit code: $?" >> /mnt/d/VORTEX_FLAME/pipeline_8b/arena_log.txt
