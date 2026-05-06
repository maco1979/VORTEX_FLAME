@echo off
chcp 65001 >nul
set CUDA_VISIBLE_DEVICES=0
set PYTHONUNBUFFERED=1
echo ============================================================
echo   全阶段回测 Stage1-4 (40题, 3060 GPU)
echo ============================================================
D:\VORTEX_FLAME_env\Scripts\python.exe D:\VORTEX_FLAME\run_regression_test.py
echo.
echo 回测完成！
pause
