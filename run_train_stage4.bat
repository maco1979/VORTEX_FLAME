@echo off
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo ============================================================
echo   Einstein Stage4: 行业落地-半导体新能源
echo   Data: 4000 samples, 2 epochs, seq_len=128, lr=1.5e-4
echo   Gradient Checkpointing: ON (省显存)
echo   Loss threshold: 1.4, Exam rate: 80%%
echo   Resume from Stage3 LoRA
echo ============================================================
echo.
D:\VORTEX_FLAME_env\Scripts\python.exe D:\VORTEX_FLAME\train_einstein.py --stage 4 --resume
echo.
echo Stage4 done! Exit code: %ERRORLEVEL%
echo Check: D:\VORTEX_FLAME\hermes_logs\einstein\stage4_result.json
pause
