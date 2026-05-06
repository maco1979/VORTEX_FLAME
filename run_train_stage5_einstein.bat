@echo off
chcp 65001 >nul
echo ============================================================
echo   Stage5: Einstein 全阶段回测+行业工具训练 (3060 GPU)
echo   铁则: 旧知识不丢 + 新知识学会 = 才算合格
echo   Tools: Ansys, COMSOL, 电力仿真, 工业AI
echo   SeqLen: 256 | LoRA: r=16, alpha=32
echo ============================================================
echo.
echo 确保Stage4训练已完成！
echo.
pause
set CUDA_VISIBLE_DEVICES=0
python train_stage5.py --soul einstein --resume --seq_len 256 --epochs 2 --lr 1e-4 --max_samples 8000
echo.
echo 训练完成！检查全阶段回测结果！
pause
