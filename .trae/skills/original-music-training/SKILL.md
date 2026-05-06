---
name: "original-music-training"
description: "使用用户原创音乐训练AI风格模型。Invoke when user wants to train AI on their original music, learn music style, or extract features from songs."
---

# Original Music Training

Train AI to learn your unique music style using your original songs.

## Quick Usage

```bash
# Train with your original music
python d:/贾维斯/train_original_music.py

# Compose using learned style
python d:/贾维斯/compose_quick.py "主题" "情绪"
```

## Files

| File | Purpose |
|------|---------|
| `d:/贾维斯/train_original_music.py` | Training entry point |
| `d:/贾维斯/config/music_train_config.py` | Training configuration |
| `d:/贾维斯/core/collaboration/music_dataset_loader.py` | Dataset loader |
| `d:/贾维斯/models/style_config.json` | Learned style profile |

## Configuration

Edit `d:/贾维斯/config/music_train_config.py`:

```python
TRAIN_CONFIG = {
    "music_root": "d:/贾维斯/原创音乐/",  # Your original music folder
    "supported_formats": [".mp3", ".wav", ".flac"],
    "model_save_path": "d:/贾维斯/models/fkj_soul_music_v1.pth",
}
```

## Learned Style Features

After training, extracts:
- Average BPM
- Main musical key
- Energy level (high/medium/low)
- Tempo style (fast/medium/slow)
- Complexity level

## Current Status

Your music library: 62 songs
Average BPM: 123.7
Main Key: F
Energy: low