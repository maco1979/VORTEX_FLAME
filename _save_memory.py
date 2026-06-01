#!/usr/bin/env python3
"""
保存当前 VORTEX FLAME JEPA 底座训练计划到 LongMemory。
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
from long_memory import LongMemory

memory = LongMemory()
THREAD = "JEPA_BASELINE_TRAINING_2026_0529"

memory.start_thread("system", THREAD)

plan = {
    "phase": "JEPA基座预训练 — Phase 1 (合成时序数据)",
    "architecture": "ObjectSlotEncoder → ObjectLevelMasker → CausalPredictor",
    "loss": "SIGRegWithPredictionLoss (MSE + var+cov, 单超参λ=0.1)",
    "ema": "Target Encoder EMA 0.996, requires_grad=False",
    "optimizer": "Adam, lr=1e-4, wd=1e-5, warmup=200steps, CosineAnnealingWarmRestarts",
    "per_variant_params": "~6M, ~30min/100epochs V100-16GB",
    "variants": [
        {"name": "CAJEPA",      "souls": "beethoven",               "input_dim": 512, "slots": 5, "status": "TRAINING",     "epoch": "~57/100", "terminal": 5},
        {"name": "CPHYSJEPA",   "souls": "einstein, galileo",       "input_dim": 512, "slots": 7, "status": "PENDING_P0",   "script": "train_physjepa.py"},
        {"name": "CCODEJEPA",   "souls": "cezanne",                 "input_dim": 384, "slots": 6, "status": "PENDING_P0",   "script": "train_codejepa.py"},
        {"name": "CBIOJEPA",    "souls": "darwin, yuanlongping",    "input_dim": 512, "slots": 7, "status": "PENDING_P0",   "script": "train_biojepa.py"},
        {"name": "CLAWJEPA",    "souls": "guizhu, montesquieu",     "input_dim": 512, "slots": 7, "status": "PENDING_P1",   "script": "train_lawjepa.py"},
        {"name": "CGEOJEPA",    "souls": "humboldt, herodotus",     "input_dim": 512, "slots": 7, "status": "PENDING_P1",   "script": "train_geojepa.py"},
        {"name": "CARTJEPA",    "souls": "monet, vangogh",          "input_dim": 768, "slots": 8, "status": "PENDING_P1",   "script": "train_artjepa.py"},
        {"name": "CDESIGNJEPA", "souls": "davinci",                 "input_dim": 512, "slots": 6, "status": "PENDING_P1",   "script": "train_designjepa.py"},
        {"name": "CFINJEPA",    "souls": "strategy",                "input_dim": 256, "slots": 6, "status": "PENDING_P2",   "script": "train_finjepa.py"},
        {"name": "CVJEPA",      "souls": "davinci, herodotus",      "input_dim": 768, "slots": 7, "status": "PENDING_P2",   "script": "train_cvjepa_cpu.py"},
    ],
    "post_training_strategy": {
        "align_freeze": "训练完成后对齐时冻结 Context Encoder，只训 Projection头 / 动作头 / 语言头",
        "finetune": "下游深度微调：冻底层，松顶层，lr=1e-5",
        "target_encoder": "只做EMA，不反传，全程冻结梯度",
    },
    "launch_command": "python train_all_jepa.py --epochs 100 --batch 8 --lr 1e-4",
    "verified": "2026 LeJEPA官方流程验证通过：target冻结✓, EMA 0.996✓, SIGReg损失✓, warmup✓, guard✓",
    "豆包流程一致性": "100%对齐 — 预训练 Target Encoder冻结 + EMA, 对齐时冻结 Context Encoder",
}

memory.append("system", THREAD, json.dumps(plan, ensure_ascii=False, indent=2),
              role="system", metadata={"type": "training_plan", "version": "2026-05-31"})

print("✅ JEPA训练计划已保存到 LongMemory")
print(f"   DB: {memory.db_path}")
print(f"   Thread: {THREAD}")
print(f"   {len(plan['variants'])} 个变体已登记")
