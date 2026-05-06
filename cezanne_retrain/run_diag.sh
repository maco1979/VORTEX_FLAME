#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/cezanne_retrain
STAGE="${1:-Base}"
echo "Testing: $STAGE"
python diagnostic_single.py "$STAGE" 2>&1 | tee /mnt/d/VORTEX_FLAME/cezanne_retrain/logs/diag_${STAGE}.log
