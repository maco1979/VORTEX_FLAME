<# 
.SYNOPSIS
  VORTEX FLAME AlphaZero Guardian - keeps WSL2 alive + self-play running
.DESCRIPTION
  This script MUST run in a dedicated PowerShell window that stays open.
  It keeps WSL2 alive by maintaining a persistent session, and monitors/restarts
  the self-play process if it dies.
.USAGE
  Open a new PowerShell window and run:
  powershell -ExecutionPolicy Bypass -File D:\VORTEX_FLAME\cezanne_retrain\guardian_alphazero.ps1
#>

$Host.UI.RawUI.WindowTitle = "VORTEX FLAME Guardian"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  VORTEX FLAME AlphaZero Guardian" -ForegroundColor Cyan
Write-Host "  Keeps WSL2 alive + monitors self-play" -ForegroundColor Cyan  
Write-Host "  DO NOT CLOSE THIS WINDOW!" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

wsl -d Ubuntu -- bash -c "echo 'WSL2 connection established' && uname -a"
Write-Host ""

$checkInterval = 120

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    $procCheck = wsl -d Ubuntu -- bash -c "ps aux | grep hermes_alphazero | grep -v grep | wc -l" 2>$null
    $isRunning = $procCheck -match "^[1-9]"
    
    if ($isRunning) {
        $pidInfo = wsl -d Ubuntu -- bash -c "ps aux | grep hermes_alphazero | grep -v grep | awk '{print `$2}'" 2>$null
        Write-Host "[$ts] Self-play RUNNING (PID: $pidInfo)" -ForegroundColor Green
    } else {
        Write-Host "[$ts] Self-play NOT running! Restarting..." -ForegroundColor Red
        
        wsl -d Ubuntu -- bash -c "pkill -9 -f hermes_alphazero 2>/dev/null; sleep 2"
        Start-Sleep -Seconds 3
        
        $gpuCheck = wsl -d Ubuntu -- bash -c "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l" 2>$null
        if ($gpuCheck -match "^[1-9]") {
            Write-Host "[$ts] WARNING: GPU still has processes, waiting..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
        }
        
        Write-Host "[$ts] Launching self-play..." -ForegroundColor Yellow
        wsl -d Ubuntu -- bash -c "bash /mnt/d/VORTEX_FLAME/cezanne_retrain/run_alphazero_bg.sh"
        Write-Host "[$ts] Waiting 90s for model loading..." -ForegroundColor Yellow
        Start-Sleep -Seconds 90
    }
    
    $latestLog = wsl -d Ubuntu -- bash -c "ls -t /mnt/d/VORTEX_FLAME/cezanne_retrain/logs/alphazero_*.log 2>/dev/null | head -1" 2>$null
    if ($latestLog) {
        $logTail = wsl -d Ubuntu -- bash -c "tail -2 $latestLog 2>/dev/null" 2>$null
        Write-Host "[$ts] Log: $logTail" -ForegroundColor DarkGray
    }
    
    $ckpt = wsl -d Ubuntu -- bash -c "python3 -c 'import json; d=json.load(open(\"/mnt/d/VORTEX_FLAME/cezanne_retrain/alphazero_checkpoint.json\")); print(f\"iter={d.get(chr(110)+chr(101)+chr(120)+chr(116)+chr(95)+chr(105)+chr(116)+chr(101)+chr(114)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110),1)} best={d.get(chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(115)+chr(99)+chr(111)+chr(114)+chr(101),0):.0%}\")' 2>/dev/null"
    if ($ckpt) {
        Write-Host "[$ts] Checkpoint: $ckpt" -ForegroundColor DarkGray
    }
    
    Write-Host "[$ts] Next check in ${checkInterval}s (WSL2 stays alive)..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $checkInterval
}
