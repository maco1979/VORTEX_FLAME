#!/bin/bash
# Cezanne 7B Retrain — All Stages
# Usage: bash run_all.sh [stage]
#   stage=1  → only S1
#   stage=2  → only S2 (requires S1 done)
#   stage=3  → only S3 (requires S2 done)
#   stage=all → S1 → S2 → S3 sequentially

eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame

STAGE="${1:-all}"
SCRIPT_DIR="/mnt/d/VORTEX_FLAME/cezanne_retrain"
LOG_DIR="/mnt/d/VORTEX_FLAME/cezanne_retrain/logs"
mkdir -p "$LOG_DIR"

run_stage() {
    local name=$1
    local script=$2
    echo "=========================================="
    echo "  Starting: $name"
    echo "  Time: $(date)"
    echo "=========================================="
    python "$SCRIPT_DIR/$script" 2>&1 | tee "$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "[FATAL] $name failed with exit code $exit_code"
        exit 1
    fi
    echo "$name completed at $(date)"
    echo ""
    sleep 10
}

case "$STAGE" in
    1)
        run_stage "S1_Math" "train_s1.py"
        ;;
    2)
        run_stage "S2_Logic_Debug" "train_s2.py"
        ;;
    3)
        run_stage "S3_CS_Depth" "train_s3.py"
        ;;
    all)
        run_stage "S1_Math" "train_s1.py"
        run_stage "S2_Logic_Debug" "train_s2.py"
        run_stage "S3_CS_Depth" "train_s3.py"
        ;;
    *)
        echo "Usage: bash run_all.sh [1|2|3|all]"
        exit 1
        ;;
esac

echo "=========================================="
echo "  All requested stages complete!"
echo "=========================================="
