---
name: "suno-automation"
description: "Suno API automation module for music generation. Invoke when user wants to generate music from lyrics using Suno API."
---

# Suno Automation - 音乐自动生成模块

## 模块位置
`D:\贾维斯\VORTEX_FLAME\suno_integration.py`

## 功能

1. **生成音乐** - 从歌词或描述生成歌曲
2. **批量生成** - 一次生成多首歌曲
3. **黑陷阱预设** - 使用预设参数快速生成
4. **自动保存** - 保存歌词和生成记录

## 使用方法

### 1. 基本使用

```python
from suno_integration import SunoAutomation, fkj_generate_song

# 方式1: 使用快捷函数
fkj_generate_song("清晨", lyrics_text)

# 方式2: 使用完整类
suno = SunoAutomation(api_key="your_api_key")
suno.generate_from_lyrics("清晨", lyrics_text, style="ambient chillhop")
```

### 2. 黑陷阱音乐预设

```python
from suno_integration import BLACKHOLE_TRAP_PRESETS

# 使用预设风格
style = BLACKHOLE_TRAP_PRESETS["清晨"]
suno.generate_from_lyrics("清晨", lyrics_text, style=style)
```

### 3. 批量生成

```python
songs = [
    {"title": "清晨", "lyrics": "歌词...", "style": "ambient chillhop"},
    {"title": "雨夜", "lyrics": "歌词...", "style": "chillhop"},
]

suno.generate_batch(songs)
```

### 4. 检查余额

```python
suno = SunoAutomation(api_key="your_api_key")
balance = suno.check_balance()
print(balance)
```

## 黑陷阱音乐参数

| 预设 | BPM | 风格关键词 |
|------|-----|-----------|
| 清晨 | 72 | ambient chillhop, rubato, piano, synth pads |
| 雨夜 | 65 | rain sounds, warm piano, melancholic |
| 情绪 | 70 | cinematic, strings, minimal drums |
| 赛博 | 80 | cyberpunk, neon, bass, glitch |

## API Key配置

在代码中替换 `api_key` 为你的Suno API Key:
- Suno API: https://api.sunoapi.org
- 或使用其他兼容API

## 输出目录
所有生成的文件保存在: `D:/贾维斯/Suno输出/`
