#!/bin/bash
# Cezanne 7B Full Retrain — 3 Stages (CS-Focused)
# S1: CS Math 8K → S2: CS Logic 16K → S3: CS Depth 24K
# Each stage: independent LoRA on Base, r=16, accumulated data
# Safety: Loss Gate + Regression Test Gate + Auto-rollback

eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame

SCRIPT_DIR="/mnt/d/VORTEX_FLAME/cezanne_retrain"
LOG_DIR="/mnt/d/VORTEX_FLAME/cezanne_retrain/logs"
mkdir -p "$LOG_DIR"

STAGE="${1:-all}"

run_stage() {
    local name=$1
    local script=$2
    echo "=========================================="
    echo "  Starting: $name at $(date)"
    echo "=========================================="
    python "$SCRIPT_DIR/$script" 2>&1 | tee "$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[FATAL] $name failed"
        exit 1
    fi
    echo "$name done at $(date)"
    sleep 10
}

case "$STAGE" in
    1) run_stage "S1_CS_Math_8K" "train_s1_8k.py" ;;
    2) run_stage "S2_CS_Logic_8K" "train_s2_8k.py" ;;
    3) run_stage "S3_CS_Depth_8K" "train_s3_8k.py" ;;
    all)
        run_stage "S1_CS_Math_8K" "train_s1_8k.py"
        run_stage "S2_CS_Logic_8K" "train_s2_8k.py"
        run_stage "S3_CS_Depth_8K" "train_s3_8k.py"
        ;;
    *) echo "Usage: bash run_full.sh [1|2|3|all]" ;;
esac

echo "=========================================="
echo "  All done!"
echo "=========================================="
