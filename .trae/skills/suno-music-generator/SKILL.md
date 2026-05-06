---
name: "suno-music-generator"
description: "Suno音乐生成器 - 一键生成并下载音乐。Invoke when user wants to create music with Suno API and download automatically."
---

# Suno 音乐生成器技能

## 概述

一键完成：歌词创作 → Suno生成 → 自动下载

## 使用方法

```python
from suno_music_generator import SunoMusicGenerator

# 创建生成器
generator = SunoMusicGenerator(api_key="your_api_key")

# 一键生成并下载
result = generator.create_song(
    title="黄昏的渡口",
    lyrics="""
[诗句]
黄昏染红了渡口
一只孤舟静静等候
...
""",
    style="Lo-Fi Chillhop, BPM 85, F major, warm piano",
    output_dir=r"E:\人声训练包\原始音频"
)

# 返回下载路径
print(result["audio_path"])
```

## 完整流程

| 步骤 | 功能 | 状态 |
|------|------|------|
| 1 | 检查积分 | 自动 |
| 2 | 提交生成 | 自动 |
| 3 | 轮询状态 | 自动 |
| 4 | 下载音频 | 自动 |
| 5 | 下载封面 | 自动 |

## 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| title | str | 歌曲名称 |
| lyrics | str | 歌词内容 |
| style | str | 音乐风格描述 |
| output_dir | str | 下载目录 |
| model | str | 模型版本 (V3_5, V4, V5, V5_5) |

## 输出文件

```
{output_dir}/
├── {歌曲名}_版本1.mp3
├── {歌曲名}_版本1_cover.jpg
├── {歌曲名}_版本2.mp3
└── {歌曲名}_版本2_cover.jpg
```

## 示例

```python
# FKJ 创作新歌
result = generator.create_song(
    title="星河古道",
    lyrics="""
[诗句]
星河落古道
月光洒长亭
...
""",
    style="Chinese classical, ambient, BPM 80, emotional ballad",
    output_dir=r"E:\人声训练包\原始音频"
)
```

## 快速使用

```python
from suno_music_generator import quick_create

# 一行代码生成并下载
result = quick_create(
    title="歌名",
    lyrics="歌词内容",
    style="风格描述"
)
```

## 文件位置

- 生成器: [suno_music_generator.py](file:///d:/贾维斯/suno_music_generator.py)
- API文档: https://docs.sunoapi.org/cn
