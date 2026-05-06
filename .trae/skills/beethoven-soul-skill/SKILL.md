---
name: "beethoven-soul-skill"
description: "贝多芬灵魂技能包-声学与语言创作。Invoke when user needs music theory, acoustics analysis, language composition, or Beethoven style creation."
---

# 贝多芬灵魂技能包 - LANGUAGE (0,1)

## 核心学科
- 物理学-声学(140)
- 语言学
- 数学(110)

## 能力范围
1. 声学分析与合成
2. 音乐理论(和声、对位、曲式)
3. 语言创作(歌词、叙事、修辞)
4. NLP与文本生成

## 学习路径
| 级别 | 内容 | 数据 |
|------|------|------|
| 1a | 声波基础、音高频率、语法结构 | 500条 |
| 1b | 和声学、修辞学、调式调性 | 2000条 |
| 1c | 编曲理论、NLP、交响创作 | 1000条 |
| 2 | 音乐+语言交叉 | 3500条 |
| 3 | 音乐制作、教育、内容创作 | 3000条 |

## 训练数据
`F:\VORTEX_FLAME\soul_training_data\beethoven\beethoven_hq_10k.json`

## 模型配置
- 24L 768H 12heads (整模型训练)
- 目标Loss: 2.53
- BPE分词器: `beethoven_bpe_200k.json`
