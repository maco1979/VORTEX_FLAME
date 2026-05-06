@echo off
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo ============================================================
echo   Einstein Stage3: 融合训练-数理融合
echo   Data: 8000 samples, 2 epochs, seq_len=256, lr=2e-4
echo   Loss threshold: 1.6, Exam rate: 80%%
echo   Resume from Stage2 LoRA
echo ============================================================
echo.
D:\VORTEX_FLAME_env\Scripts\python.exe D:\VORTEX_FLAME\train_einstein.py --stage 3 --resume --seq_len 256
echo.
echo Stage3 done! Exit code: %ERRORLEVEL%
echo Check: D:\VORTEX_FLAME\hermes_logs\einstein\stage3_result.json
pause
