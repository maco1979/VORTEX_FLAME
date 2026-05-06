#!/bin/bash
# Adversarial Self-Play: Complete Pipeline
# Phase 1: 7B answers → Phase 2: 8B answers → Generate training data
# Usage: bash run_adversarial.sh [1|2|all]

source /home/chen/miniconda3/etc/profile.d/conda.sh
conda activate vortex_flame
cd /mnt/d/VORTEX_FLAME/cezanne_retrain

LOG="/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"

case "${1:-all}" in
  1)
    echo "=== Phase 1: 7B answers ==="
    python -u adversarial_phase1_7b.py 2>&1 | tee "$LOG/p1.log"
    ;;
  2)
    echo "=== Phase 2: 8B answers + training data ==="
    python -u adversarial_phase2_8b.py 2>&1 | tee "$LOG/p2.log"
    ;;
  all)
    echo "=== Phase 1: 7B answers ==="
    python -u adversarial_phase1_7b.py 2>&1 | tee "$LOG/p1.log"
    echo ""
    echo "=== Phase 2: 8B answers + training data ==="
    python -u adversarial_phase2_8b.py 2>&1 | tee "$LOG/p2.log"
    echo ""
    echo "=== DONE ==="
    ;;
  *)
    echo "Usage: bash run_adversarial.sh [1|2|all]"
    ;;
esac
