@echo off
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo ============================================================
echo   Einstein Stage2: 知识扩展-数学化学
echo   Data: 8000 samples, 3 epochs, seq_len=256, lr=2.5e-4
echo   Loss threshold: 2.0, Exam rate: 70%%
echo   Resume from Stage1 LoRA
echo ============================================================
echo.
D:\VORTEX_FLAME_env\Scripts\python.exe D:\VORTEX_FLAME\train_einstein.py --stage 2 --resume --seq_len 256
echo.
echo Stage2 done! Exit code: %ERRORLEVEL%
echo Check: D:\VORTEX_FLAME\hermes_logs\einstein\stage2_result.json
pause
