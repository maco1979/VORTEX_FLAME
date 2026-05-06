---
name: "sunoapi-sdk"
description: "SunoAPI SDK for music generation, lyrics creation, vocal separation, and audio processing. Invoke when user wants to use Suno API for music creation or needs SunoAPI integration."
---

# SunoAPI SDK 技能

## 概述

完整封装 SunoAPI 29+ 端点的 SDK，支持音乐生成、歌词创作、人声分离、音频处理等功能。

## 核心功能

| 功能 | 方法 | 说明 |
|------|------|------|
| 积分查询 | `get_credits()` | 查询账户积分 |
| 音乐生成 | `generate()` | 生成音乐 |
| 歌词生成 | `generate_lyrics()` | AI生成歌词 |
| 人声分离 | `separate_vocals()` | 分离人声和伴奏 |
| WAV转换 | `convert_to_wav()` | 转换为WAV格式 |
| 音乐视频 | `create_music_video()` | 生成音乐视频 |
| MIDI生成 | `generate_midi()` | 从音频生成MIDI |

## 使用示例

```python
from sunoapi_sdk_v2 import SunoSDK

sdk = SunoSDK(api_key="your_api_key")

# 查询积分
credits = sdk.get_credits()

# 生成音乐
result = sdk.generate(
    prompt="lofi chillhop electronic warm",
    instrumental=False,
    custom_mode=True,
    lyrics="[Verse]\n歌词内容...",
    model="V4_5ALL"
)

# 人声分离
vocals = sdk.separate_vocals(task_id)
```

## 端点路径

| 功能 | 端点 |
|------|------|
| 积分 | `/generate/credit` |
| 歌词生成 | `/lyrics` |
| 歌词详情 | `/lyrics/record-info` |
| 音乐生成 | `/generate` |
| 音乐详情 | `/generate/record-info` |

## 注意事项

1. 需要有效的 API Key
2. 所有请求需要 `callBackUrl` 参数
3. 音乐生成消耗积分

## 文件位置

- SDK文件: `d:\贾维斯\sunoapi_sdk_v2.py`
- 文档: https://docs.sunoapi.org/cn