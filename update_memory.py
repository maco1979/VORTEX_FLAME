#!/usr/bin/env python3
import os

path = r"C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = """  transformers升级: 4.46.3 -> 5.7.0 (支持mistral3)
  当前: 等Stage3训练启动 + 8B基准测试跑分中

===============================================================================
  END OF MEMORY SUPPLEMENT v2.0
==============================================================================="""

new = """  transformers升级: 4.46.3 -> 5.7.0 (支持mistral3)
  升级对7B训练无影响: AutoModelForCausalLM/peft/trl/unsloth全部兼容

  8B基准测试结果(术前原始模型):
  - LowLevel: 100% (10/10), LogicCode: 100% (10/10)
  - Debug: 100% (5/5), Security: 100% (5/5), Agent: 100% (5/5)
  - 总计: 35/35 全满分, 8B底座原始能力极强

  8B脑科手术(2026-05-02):
  - 砍视觉编码器: 403M参数(Pixtral 24层)
  - 砍多模态投影器: 25M参数
  - 砍上下文: 262144 -> 1024 (KV缓存36.5GB -> 0.14GB, 最狠一刀!)
  - 砍6层实验: 失败! 推理输出乱码, 顶层不可砍
  - 术后模型: D:\\models\\Ministral-8B-Reasoning-Text-34L-ctx1024
  - 术后参数: 8.49B (34层全保留)
  - 术后VRAM: 6.9GB (4bit) vs 术前10.9GB
  - 术后推理: 正常(2+3=5, 进程解释, 素数函数)
  - E盘备份: E:\\models\\Ministral-8B-Reasoning-Text-34L-ctx1024 (8.5GB)

  关键发现: KV缓存才是显存杀手,不是模型权重!
  - ctx=262144: KV=36.5GB (不可能跑)
  - ctx=1024: KV=0.14GB (轻松跑)
  - 长记忆靠外挂FAISS,模型本体只负责"思考"

  8B Cezanne训练(进行中):
  - 脚本: pipeline_8b/train_8b_cezanne_s1.py + s2.py
  - 底座: D:\\models\\Ministral-8B-Reasoning-Text-34L-ctx1024
  - 数据: 与7B完全相同(cezanne_stage1_math_8k + cezanne_stage2_logic_8k)
  - LoRA目录: soul_lora_v2/cezanne_8b/
  - 日志目录: hermes_logs/cezanne_8b/
  - peft兼容: 需用lang_target_modules只对language_model加LoRA
  - 不用prepare_model_for_kbit_training(与Mistral3ForConditionalGeneration不兼容)
  - trainable params: 44.5M / 8.96B = 0.5%

  D盘数据同步: E:\\D盘数据\\VORTEX_FLAME (robocopy /MIR)

===============================================================================
  END OF MEMORY SUPPLEMENT v2.0
==============================================================================="""

content = content.replace(old, new)
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Memory updated OK")
