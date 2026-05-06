#!/bin/bash
# Run adversarial self-play Phase 1 (7B answers)
source /home/chen/miniconda3/etc/profile.d/conda.sh
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/cezanne_retrain
python -u adversarial_phase1_7b.py 2>&1 | tee /mnt/d/VORTEX_FLAME/hermes_logs/cezanne/p1.log
echo "Phase 1 exit code: $?"
