---
name: "audio-analysis-learning"
description: "Analyze audio files and extract features for AI soul learning. Invoke when user wants to analyze music files, extract audio features, or let AI souls learn from created songs."
---

# 音频分析与学习技能

## 概述

分析音频文件，提取音乐特征，让 AI 灵魂学习已创作歌曲的风格特点。

## 核心功能

| 功能 | 说明 |
|------|------|
| **BPM分析** | 检测歌曲速度 |
| **调性检测** | 识别音乐调性 (C, D, E...) |
| **能量分析** | 分析歌曲动态范围 |
| **频谱分析** | 提取音色特征 |
| **MFCC特征** | 用于机器学习的音频特征 |

## 使用方法

```python
from audio_analyzer import analyze_and_learn

# 分析歌曲
features = analyze_and_learn("歌曲路径.mp3", "数字春天")

# 输出:
# 时长: 180.5秒
# BPM: 85.2
# 调性: C
# 能量: 0.0234
```

## 分析结果示例

```json
{
  "file": "数字春天.mp3",
  "duration": 180.5,
  "bpm": 85.2,
  "key": "C",
  "energy_mean": 0.0234,
  "spectral_centroid_mean": 2500.5,
  "mfcc_mean": [...]
}
```

## FKJ 学习流程

1. 分析音频提取特征
2. 生成学习提示词
3. FKJ 灵魂学习风格特点
4. 应用于未来创作

## 输出文件

| 文件 | 说明 |
|------|------|
| `{歌曲名}_features.json` | 音频特征数据 |
| `{歌曲名}_learning_prompt.txt` | FKJ学习提示词 |

## 支持格式

- MP3
- WAV
- FLAC
- OGG
- M4A

## 文件位置

- 分析器: `d:\贾维斯\audio_analyzer.py`
- 输出目录: `d:\贾维斯\outputs\audio_analysis\`

## 依赖

- librosa (音频分析)
- pydub (音频处理)
- numpy (数值计算)