#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/pipeline_8b
nohup python arena_7b_vs_8b.py > /mnt/d/VORTEX_FLAME/pipeline_8b/arena_log.txt 2>&1 &
ARENA_PID=$!
echo "Arena PID: $ARENA_PID" > /mnt/d/VORTEX_FLAME/pipeline_8b/arena_pid.txt
echo "Started: $(date)" >> /mnt/d/VORTEX_FLAME/pipeline_8b/arena_pid.txt
echo "Arena started in background with PID $ARENA_PID"
