#!/bin/bash
eval "$(/home/chen/miniconda3/bin/conda shell.bash hook)"
conda activate vortex_flame

cd /mnt/d/VORTEX_FLAME/cezanne_retrain
LOG_DIR="/mnt/d/VORTEX_FLAME/cezanne_retrain/logs"
mkdir -p "$LOG_DIR"

OLD_PIDS=$(pgrep -f hermes_alphazero.py 2>/dev/null)
if [ -n "$OLD_PIDS" ]; then
    echo "Killing old hermes_alphazero processes: $OLD_PIDS"
    kill -9 $OLD_PIDS 2>/dev/null
    sleep 3
fi

GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v "^$" | head -1)
if [ -n "$GPU_PROCS" ]; then
    echo "WARNING: GPU still has processes: $GPU_PROCS"
    echo "Waiting 5s for GPU to clear..."
    sleep 5
fi

LOGFILE="$LOG_DIR/alphazero_$(date +%Y%m%d_%H%M%S).log"
nohup python -u hermes_alphazero.py >> "$LOGFILE" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/alphazero_pid.txt"
echo "Started AlphaZero with PID $PID"
echo "Log: $LOGFILE"

CKPT="/mnt/d/VORTEX_FLAME/cezanne_retrain/alphazero_checkpoint.json"
if [ -f "$CKPT" ]; then
    echo "Checkpoint found - will resume from saved state"
    cat "$CKPT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Resume from iter {d.get(\"next_iteration\",1)}, phase={d.get(\"phase\",\"?\")}, best_score={d.get(\"best_score\",0):.0%}')" 2>/dev/null
else
    echo "No checkpoint - starting fresh"
fi
