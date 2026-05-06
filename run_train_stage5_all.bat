@echo off
chcp 65001 >nul
echo ============================================================
echo   Stage5: 全部14灵魂 全阶段回测+行业工具训练 (3060 GPU)
echo   铁则: 旧知识不丢 + 新知识学会 = 才算合格
echo   依次训练: einstein -> davinci -> ... -> herodotus
echo ============================================================
echo.
pause
set CUDA_VISIBLE_DEVICES=0
python train_stage5.py --all --resume --seq_len 256 --epochs 2 --lr 1e-4 --max_samples 8000
echo.
echo 全部训练完成！
pause
