---
name: "self-learning"
description: "让AI灵魂能够自我评估、发现问题并生成新技能。Invoke when souls need self-improvement, daily review, or skill auto-generation."
---

# 自进化学习系统

## 功能说明

让AI灵魂能够自我评估、发现问题并生成新技能。

## 每日歌词微调系统 ⭐NEW

FKJ灵魂每天都要进行歌词创作复盘和微调：

### 每日流程

```
每天创作 → 记录日志 → 自动复盘 → 微调优化 → 技能进化
```

### 每日复盘检查

| 检查项 | 标准 | 问题 |
|--------|------|------|
| 词汇重复 | 同一词不超2次 | 答案/孤独/悲伤 |
| 韵律统一 | 副歌韵脚一致 | -ang/-ian/-u/-i |
| 意象独特 | 避免常用词 | 星光/火焰 |
| 句式变化 | 长短句结合 | 堆砌句式 |
| 情感递进 | 情绪逐层升级 | 平铺直叙 |

### 日志目录

```
d:/贾维斯/FKJ万词计划/每日复盘/
├── 2026-04-14_creations.json
├── 2026-04-15_creations.json
└── ...
```

### 调用方法

```python
from fkj_daily_review import DailyLyricsReview

review = DailyLyricsReview()

# 记录新创作
review.record_creation(lyrics="...", theme="种子", score=8.5)

# 每日复盘
review.daily_review()
```

## 灵魂贡献者

爱因斯坦, 贝多芬, 梵高, FKJ ⭐

## 核心实现

```python
class DailyLyricsReview:
    def __init__(self):
        self.soul_name = "FKJ"
        self.log_dir = "d:/贾维斯/FKJ万词计划/每日复盘"

    def record_creation(self, lyrics: str, theme: str, score: float = None):
        """记录每日创作"""

    def review_and_micro_tune(self, lyrics: str) -> Dict:
        """复盘并微调"""

    def daily_review(self):
        """每日复盘主流程"""
```

## 自我进化流程

1. **每日创作** → FKJ创作新歌词
2. **记录日志** → 保存到每日复盘目录
3. **自动复盘** → 检查词汇/韵律/意象/句式
4. **微调优化** → 自动替换问题词汇
5. **技能进化** → 更新SKILL.md

## 调用时机

- FKJ创作新歌词后 → record_creation()
- 每天固定时间 → daily_review()
- 每周总结 → get_weekly_stats()

## 核心原则

- 每天都要有新的表达方式
- 避免重复词汇超过2次
- 每次创作都要有创新点
- 复盘是进化的起点