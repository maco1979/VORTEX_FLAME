#!/usr/bin/env python3
"""Deposit core auditory prediction thesis into soul knowledge bases"""
import sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from soul_memory import SoulMemoryEngine

m = SoulMemoryEngine()

# --- Core thesis ---
thesis = {
    "title": "听觉预测优先：为什么耳朵必须先独立理解世界",
    "one_liner": "先让耳朵学会世界的物理规律，再对齐眼睛；对齐只是表征的拉近，不是知识的灌输。",
    "core_formula": "好的听觉表征 = 知道什么该保留 + 知道什么该丢弃 + 知道接下来会发生什么",
    "pillars": {
        "selective_perception": "理解的前提是选择性——不是所有信号都值得预测。Mel/耳蜗天然丢弃不重要细节。",
        "prediction_is_understanding": "JEPA核心：在表征空间做时序预测，逼模型学习物理规律。历史6帧→未来4帧=学会运动、距离、多普勒。",
        "slot_attention_disentanglement": "8个object slot竞争分配声源，通过预测任务压力迫使槽位分化。多对象物理建模在纯音频阶段完成。",
        "modality_independence": "对齐的上限由单模态独立表征深度决定。对齐是确认，不是教会。音频先独立学会时空规律，视觉再指认物体长什么样。",
        "u_shaped_hearing": "U型听力（低频敏感+中频钝+超高频完好）证明：听觉不是频谱分析，是选择性预测。不同耳朵提取不同语义，模型应学人类共识而非个体差异。",
    },
    "architecture": "CAJEPA (Causal Audio JEPA): AudioFeatureProjector → 8 Object Slots → Object-Level Masking → Causal Predictor → SIGReg + Contrastive Loss",
    "current_task": "ESC-50 50类环境声 + Deep House音乐(10%采样) → CAJEPA时序预测 + 对比学习 → 51类线性探针验证",
    "roadmap": [
        "当前: 51类声音分类验证纯音频表征已编码物理世界规律",
        "下一步: +立体声(方向) +人造混响(距离) +多普勒增强(速度) → 纯音频完成时空物理建模",
        "对齐: 音频表征+视觉表征→joint embedding space，不需人工配对标注",
    ],
    "data_pipeline": {
        "mel_cache": "mel_cache/ 3776个.pt文件，每个64s Mel spectrogram",
        "music_ratio": "10%采样(177首Deep House)，ESC-50全量(1600首)",
        "speedup": "实时解码每batch 11s → .pt缓存每batch 0.1s，提速110倍",
    },
}

thesis_text = json.dumps(thesis, ensure_ascii=False, indent=2)

# Write to 3 souls: beethoven (audio), galileo (physics), einstein (theory)
for soul in ["beethoven", "galileo", "einstein"]:
    m.write(
        soul,
        "knowledge",
        {
            "topic": "核心论述：听觉预测优先 — 为什么耳朵必须先独立理解世界",
            "text": thesis_text,
        },
        tags=["core_thesis", "jepa", "audio", "prediction", "modality", "world_model"],
    )
    print(f"Written to {soul}")

# Also write a concise executor-facing version to cezanne (code/architecture)
m.write(
    "cezanne",
    "knowledge",
    {
        "topic": "CAJEPA训练管线：Mel缓存 + 10%音乐采样 + 51类对比学习",
        "text": json.dumps({
            "pipeline": "scan_audio_files() → Mel cache (.pt) → 10% music + 100% ESC-50 → CAJEPA + Contrastive",
            "config": {"epochs": 50, "batch": 8, "lr": 1e-4, "contrastive_weight": 0.3, "music_ratio": 0.1, "sr": 22050, "n_mels": 128, "segment_frames": 256},
            "checkpoint": "ajepa_checkpoints/ajepa_best.pt (projector weights, warm start)",
            "cache_dir": "mel_cache/ (3776 .pt files)",
            "key_files": ["train_ajepa_multiclass.py", "_precache_mel.py"],
        }, ensure_ascii=False),
    },
    tags=["pipeline", "training", "cajepa", "esc50"],
)
print("Written to cezanne")

print("\nCore thesis deposited to: beethoven, galileo, einstein, cezanne")
