#!/bin/bash
source /home/chen/miniconda3/etc/profile.d/conda.sh
conda activate vortex_flame

cd /mnt/d/VORTEX_FLAME/cezanne_retrain

OLD_PIDS=$(pgrep -f hermes_alphazero.py 2>/dev/null)
if [ -n "$OLD_PIDS" ]; then
    echo "Killing old processes: $OLD_PIDS"
    kill -9 $OLD_PIDS 2>/dev/null
    sleep 3
fi

LOG_DIR="/mnt/d/VORTEX_FLAME/cezanne_retrain/logs"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/alphazero_$(date +%Y%m%d_%H%M%S).log"

echo "Starting AlphaZero self-play..."
echo "Log: $LOGFILE"

nohup python -u hermes_alphazero.py >> "$LOGFILE" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/alphazero_pid.txt"
echo "PID: $PID"

sleep 5
if kill -0 $PID 2>/dev/null; then
    echo "Process is alive after 5s"
else
    echo "ERROR: Process died within 5s!"
    tail -20 "$LOGFILE"
fi

echo "Keeping WSL2 alive... (press Ctrl+C to stop)"
exec sleep infinity
