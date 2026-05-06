---
name: "fkj-audio-fine-tuning"
description: "FKJ音频微调统一pipeline：数据清洗→去噪→去冗余→QLoRA微调→AWQ量化→压缩减脂。替换所有旧训练脚本的唯一标准流程。Invoke when user wants to fine-tune FKJ with audio data, or needs the complete ML pipeline from audio to quantized model."
---

# FKJ 音频微调统一Pipeline

音频数据 → 清洗 → 去噪 → 去冗余 → QLoRA微调 → AWQ量化 → 压缩减脂

## 一键运行

```bash
C:\Users\42235\AppData\Local\Programs\Python\Python310\python.exe "d:\贾维斯\FKJ万词计划\FKJ_UNIFIED_PIPELINE.py"
```

## Pipeline 流程

```
Step1: Audio Cleaning    音频质量评估 + 过滤
Step2: Denoise           频谱减法去噪
Step3: Deduplication      MD5指纹 + 频谱相关性去重
Step4: Save Audio        保存清洗后的音频到 processed_audio/
Step5: QLoRA Fine-tune    CUDA微调 (需要transformers+peft)
Step6: AWQ Quantize      4-bit量化压缩
```

## 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| MAX_SAMPLES | 10*48000 | 每文件最多10秒(48kHz) |
| MAX_SIZE_MB | 50 | 跳过大于50MB的文件 |
| SNR_THRESHOLD | -40dB | 信噪比阈值 |
| DUP_CORR_THRESHOLD | 0.85 | 频谱相关系数>0.85为重复 |

## 数据源

- 源目录: `D:\人声训练包\原始音频\原创音乐`
- 39个WAV文件 → 清洗后21个唯一音频
- 支持格式: WAV (soundfile库自动识别float32/int16)

## 音频处理详情

### Step1 清洗
```python
# SNR计算: 中值分割法
seg_median = np.median(audio.reshape(-1, 4096), axis=1).repeat(4096)
noise = audio - seg_median
snr_db = 10 * log10(signal_power / noise_power)

# 过滤条件:
# - snr_db > -40dB (有音乐信号)
# - centroid > 20Hz (有频率内容)
# - duration > 5s (有足够长度)
```

### Step2 去噪
```python
# 频谱减法 (Spectral Subtraction)
mag = np.abs(fft.rfft(audio))
phase = np.angle(fft.rfft(audio))
est_noise = np.percentile(mag, 15)  # 噪声估计
mag_denoised = np.maximum(mag - est_noise * 0.6, 0)
audio_denoised = fft.irfft(mag_denoised * exp(1j * phase), n)
```

### Step3 去冗余
```python
# 1. MD5指纹 (前65536样本)
fp = md5(audio[:65536].tobytes())

# 2. 频谱相关性
s1 = abs(fft.rfft(audio1[:m]))
s2 = abs(fft.rfft(audio2[:m]))
corr = corrcoef(s1, s2)[0, 1]
if corr > 0.85: 标记为重复
```

## 模型微调详情

### Step5 QLoRA配置
```python
LoraConfig(
    r=64,                    # LoRA秩
    lora_alpha=128,          # 缩放因子
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
# 量化: 4bit NF4 + double_quant
# 训练: fp16, batch=1, accumulation=16, lr=2e-4
```

### Step6 AWQ量化
```python
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}
```

## 输出文件

| 文件 | 位置 | 说明 |
|------|------|------|
| fkf微调报告.json | FKJ万词计划/ | 处理结果统计 |
| processed_audio/ | FKJ万词计划/ | 清洗后的21个音频 |
| fkj_lora/ | FKJ万词计划/ | LoRA适配器权重 |
| fkj_quantized/ | FKJ万词计划/ | AWQ量化模型 |

## 报告格式

```json
{
  "timestamp": "2026-04-16 03:59:34",
  "pipeline_version": "unified_v1.0",
  "steps": ["audio_clean", "denoise", "dedup", "qlora_finetune", "awq_quantize", "compress"],
  "stats": {
    "total": 39,
    "kept": 27,
    "removed": 14,
    "features": 21
  },
  "removed": ["文件名(原因)", ...],
  "kept": ["保留的文件名", ...],
  "lora_status": "trained|skipped_missing_libs|skipped",
  "pipeline_status": "audio_only|full"
}
```

## 依赖库

```bash
pip install transformers peft accelerate bitsandbytes soundfile numpy torch
```

## 被替代的旧脚本 (已废弃)

- ❌ FKJ韩风微调流程.py
- ❌ FKJ韩风LoRA训练.py
- ❌ Qwen_LoRA_韩风训练.py
- ❌ 严格清洗数据.py
- ❌ fkf微调_pipeline.py (旧版)
- ❌ 清洗AI居民数据.py
- ❌ FKJ知识蒸馏.py
- ❌ FKJ韩风模块_深度评估.py

## 状态

- ✅ Step1-4 (音频处理): 已验证成功
- ⏳ Step5 (QLoRA微调): 需要 transformers/peft 库
- ⏳ Step6 (AWQ量化): 需要 autoawq 库