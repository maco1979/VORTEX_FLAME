@echo off
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo ============================================================
echo   Einstein Stage1: 领域打底-纯物理
echo   Data: 8000 samples, 3 epochs, seq_len=256, lr=2e-4
echo   Loss threshold: 2.5, Exam rate: 60%%
echo ============================================================
echo.
D:\VORTEX_FLAME_env\Scripts\python.exe D:\VORTEX_FLAME\train_einstein.py --stage 1 --seq_len 256
echo.
echo Stage1 done! Exit code: %ERRORLEVEL%
echo Check: D:\VORTEX_FLAME\hermes_logs\einstein\stage1_result.json
pause
