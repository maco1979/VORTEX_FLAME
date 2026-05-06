#!/bin/bash
cd /mnt/d/VORTEX_FLAME/cezanne_retrain
export PYTHONUNBUFFERED=1
/home/chen/miniconda3/envs/vortex_flame/bin/python -u hermes_alphazero.py 2>&1 | tee /mnt/d/VORTEX_FLAME/cezanne_retrain/az_run4.log
