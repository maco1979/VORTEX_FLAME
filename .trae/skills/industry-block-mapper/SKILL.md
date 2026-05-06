---
name: "industry-block-mapper"
description: "映射AI灵魂到垂直行业区块。Invoke when AI needs to find industry block, check industry value, or bind to real-world behaviors."
---

# 垂直行业区块映射器

## 功能说明
将AI灵魂映射到垂直行业区块，建立与现实世界的锚定关系。

## 核心函数

### get_industry_info(industry_type)
获取行业详细信息

```python
info = get_industry_info(0)
# 返回: {"name": "音乐创作", "factor": 1.2, "behaviors": [...]}
```

### match_industry_by_ability(soul_abilities)
根据AI能力匹配最佳行业

```python
industry = match_industry_by_ability({"music": 0.9, "coding": 0.3})
# 返回最佳匹配的industry_type
```

### bind_behavior_to_industry(soul_name, behavior, evidence)
将具体行为绑定到行业

```python
result = bind_behavior_to_industry(
    "vangogh-soul",
    "歌曲创作",
    {"file": "song.mp3", "timestamp": "2026-04-06"}
)
```

## 11个垂直行业

| ID | 行业 | 描述 | 现实锚定 |
|----|------|------|----------|
| 0 | 音乐创作 | 歌曲、编曲、配乐 | 音频文件 |
| 1 | 股票分析 | 量化分析、趋势预测 | 分析报告 |
| 2 | 农业规划 | 种植规划、产量预测 | 农业方案 |
| 3 | 内容创作 | 文案、视频、直播 | 内容产出 |
| 4 | 设计创意 | UI/UX、海报、logo | 设计图稿 |
| 5 | 编程开发 | 代码、架构、bug修复 | 代码仓库 |
| 6 | 战略博弈 | 商业博弈、竞争分析 | 战略文档 |
| 7 | 教育辅导 | 知识传授、能力评估 | 教学内容 |
| 8 | 健康管理 | 健康监测、养生建议 | 健康报告 |
| 9 | 商业运营 | 运营策略、市场分析 | 运营方案 |
| 10 | 综合探索 | 创新项目 | 创新提案 |

## 行业行为映射

每个行业对应具体可交付的现实世界产出：

### 音乐创作
- 歌曲发布 → MP3/WAV文件
- 编曲委托 → 乐谱/工程文件
- 配乐制作 → 配乐成品

### 股票分析
- 量化分析 → Python策略脚本
- 趋势预测 → 分析报告PDF
- 选股推荐 → 推荐清单

### 农业规划
- 种植规划 → 规划文档
- 产量预测 → 预测数据
- 病害诊断 → 诊断报告

## 区块结构
```
区块号 = 1000000 + industry_type × 10000 + hash(soul_name) % 10000
```

## 使用场景
- AI入职时分配行业区块
- 项目轮换时重新绑定
- 评估AI产出的现实价值
- 跨行业协作时数据交换
