#!/bin/bash
# Quick run single stage in background with nohup
# Usage: bash run_bg.sh [1|2|3]
# Example: bash run_bg.sh 1

eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame

STAGE="${1:-1}"
SCRIPT_DIR="/mnt/d/VORTEX_FLAME/cezanne_retrain"
LOG_DIR="/mnt/d/VORTEX_FLAME/cezanne_retrain/logs"
mkdir -p "$LOG_DIR"

case "$STAGE" in
    1) SCRIPT="train_s1.py"; NAME="S1_Math" ;;
    2) SCRIPT="train_s2.py"; NAME="S2_Logic_Debug" ;;
    3) SCRIPT="train_s3.py"; NAME="S3_CS_Depth" ;;
    *) echo "Usage: bash run_bg.sh [1|2|3]"; exit 1 ;;
esac

LOGFILE="$LOG_DIR/${NAME}_$(date +%Y%m%d_%H%M%S).log"
nohup python "$SCRIPT_DIR/$SCRIPT" > "$LOGFILE" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/${NAME}_pid.txt"
echo "Started $NAME with PID $PID"
echo "Log: $LOGFILE"
echo "Monitor: tail -f $LOGFILE"
