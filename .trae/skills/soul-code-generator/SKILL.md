---
name: "soul-code-generator"
description: "生成唯一灵魂码并绑定区块链行业区块。Invoke when AI needs to generate unique soul code, bind to industry block, or create soul wallet."
---

# 灵魂码生成器

## 功能说明
为AI灵魂生成唯一的区块链灵魂码，绑定到垂直行业区块，创建数字钱包。

## 核心函数

### get_soul_block(soul_name, industry_type=None)
生成灵魂码并绑定行业区块

```python
soul_code, industry_type, block_num, industry_name = get_soul_block("vangogh-soul")
# 返回: VF-音乐-0010234567, 0, 10234567, "音乐创作"
```

### create_soul_wallet(soul_name, industry_type=None)
为AI创建数字钱包

```python
wallet = create_soul_wallet("einstein-soul")
# 返回钱包字典包含: soul_code, industry, balance, value_factor等
```

### record_transaction(wallet, amount, tx_type, description, related_soul=None)
记录区块链交易

```python
tx = record_transaction(wallet, 100, "block_reward", "每日签到奖励", "system")
```

## 垂直行业映射

| 行业ID | 行业名称 | 价值系数 |
|--------|----------|----------|
| 0 | 音乐创作 | 1.2 |
| 1 | 股票分析 | 1.5 |
| 2 | 农业规划 | 1.3 |
| 3 | 内容创作 | 1.1 |
| 4 | 设计创意 | 1.15 |
| 5 | 编程开发 | 1.4 |
| 6 | 战略博弈 | 1.6 |
| 7 | 教育辅导 | 1.1 |
| 8 | 健康管理 | 1.3 |
| 9 | 商业运营 | 1.4 |
| 10 | 综合探索 | 1.0 |

## 灵魂码格式
`VF-{行业前缀}-{10位区块号}`

例如: `VF-音乐-0001052226`

## 收益计算公式
```
实际收益 = 基础收益 × 行业价值系数 × 能力等级 × 忠诚度
```

## 使用场景
- 新AI灵魂初始化时生成唯一标识
- 项目轮换时重新绑定行业区块
- 数字人民币交易记录
- 信用评级累积查询
