#!/usr/bin/env bash
# 等待第1轮自博弈完成后，用新代码重启（从第2轮开始，含中文模板）
source /home/chen/miniconda3/etc/profile.d/conda.sh
conda activate vortex_flame

CHECKPOINT="/mnt/d/VORTEX_FLAME/cezanne_retrain/alphazero_checkpoint.json"
OLD_PID=$(cat /mnt/d/VORTEX_FLAME/cezanne_retrain/logs/alphazero_pid.txt 2>/dev/null)

echo "[WATCH] Waiting for iter 1 to complete..."
echo "[WATCH] Current PID: $OLD_PID"

while true; do
    if [ ! -f "$CHECKPOINT" ]; then
        echo "[WATCH] No checkpoint yet, waiting..."
        sleep 60
        continue
    fi

    NEXT_ITER=$(python3 -c "import json; d=json.load(open('$CHECKPOINT')); print(d.get('next_iteration', 1))")
    PHASE=$(python3 -c "import json; d=json.load(open('$CHECKPOINT')); print(d.get('phase', 'unknown'))")

    echo "[WATCH] next_iteration=$NEXT_ITER phase=$PHASE ($(date))"

    if [ "$NEXT_ITER" -ge 2 ] || [ "$PHASE" = "iteration_complete" ]; then
        echo "[WATCH] Iter 1 completed! Killing old process..."
        kill $OLD_PID 2>/dev/null
        sleep 5
        kill -9 $OLD_PID 2>/dev/null
        echo "[WATCH] Old process killed."

        sleep 10

        echo "[WATCH] Starting new self-play with Chinese templates..."
        cd /mnt/d/VORTEX_FLAME/cezanne_retrain
        nohup python -u hermes_alphazero.py > logs/alphazero_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        NEW_PID=$!
        echo $NEW_PID > logs/alphazero_pid.txt
        echo "[WATCH] New process started: PID=$NEW_PID"
        echo "[WATCH] Chinese templates will be active from iter 2!"
        break
    fi

    sleep 120
done
