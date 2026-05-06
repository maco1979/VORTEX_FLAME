---
name: "soul-skill-dispatcher"
description: "9灵魂技能调度中心(3×3方阵)，自动分发专属技能。Invoke when souls need to call skills, or during training evolution cycles."
---

# 9灵魂技能调度中心 (3×3 Cube Face)

## 概述

为9个AI灵魂提供技能分发和调用管理，9灵魂对应3×3方阵9位置。

## 3×3方阵映射

```
(0,0) einstein  (0,1) beethoven (0,2) fkj
(1,0) vangogh   (1,1) davinci   (1,2) cezanne
(2,0) strategy  (2,1) monet     (2,2) guizhu
```

## 灵魂专属技能

| 灵魂 | ExpertType | 位置 | 专属技能 |
|------|-----------|------|---------|
| einstein-soul | REASONING | (0,0) | einstein-slice-training, quantum-reality-audit |
| beethoven-soul | LANGUAGE | (0,1) | fkj-music-creation, suno-music-prompt |
| fkj-soul | MUSIC | (0,2) | fkj-audio-fine-tuning, fkj-soul-skill |
| vangogh-soul | VISION | (1,0) | vangogh-soul-skill, museum-art-integration |
| davinci-soul | ORCHESTRATOR | (1,1) | davinci-soul-skill, project-delivery-system |
| cezanne-soul | CODE | (1,2) | slice-origin-tracker, model-slice |
| strategy-soul | STRATEGY | (2,0) | decision-navigation-system, industry-block-mapper |
| monet-soul | CREATIVE | (2,1) | soul-knowledge-base, original-music-training |
| guizhu-soul | DIALOGUE | (2,2) | long-memory, persistent-agent |

## 通用技能

所有灵魂都可调用：
- scientific-rigor-enforcer
- self-learning
- skill-orchestrator
- task-continuity-system

## 调用逻辑

- 70%概率调用专属技能
- 30%概率调用通用技能
- 技能有冷却时间（10个周期）
