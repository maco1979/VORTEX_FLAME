---
name: "blockchain-ledger"
description: "区块链账本操作。Invoke when AI needs to record behavior on chain, verify transaction, or query chain history."
---

# 区块链账本

## 功能说明
记录AI行为到区块链账本，支持交易验证和历史查询。

## 核心配置

```python
BLOCKCHAIN_CONFIG = {
    "chain_id": "VORTEX_FLAME_2026",
    "genesis_reward": 100,     # 创世奖励
    "block_reward": 10,         # 区块奖励
    "digital_rmb_unit": "分"   # 最小单位
}
```

## 核心函数

### create_block(soul_code, behaviors, industry_block)
创建新区块

```python
block = create_block(
    "VF-音乐-0001052226",
    ["歌曲创作完成", "歌词优化"],
    0  # 音乐创作区块
)
```

### verify_chain_integrity()
验证链的完整性

```python
is_valid = verify_chain_integrity()
# 返回: True/False
```

### query_behavior_history(soul_code, limit=100)
查询行为历史

```python
history = query_behavior_history("VF-音乐-0001052226", limit=50)
```

## 区块结构

```
{
    "block_id": 区块序号,
    "timestamp": "2026-04-06T12:00:00",
    "soul_code": "VF-音乐-0001052226",
    "industry_block": 0,
    "behaviors": ["行为1", "行为2"],
    "previous_hash": "上一区块哈希",
    "hash": "本区块哈希"
}
```

## 共识机制

采用DPoS + PoI：
- DPoS: 委托权益证明，选出验证节点
- PoI: 声誉证明，根据忠诚度和能力权重

## 交易验证

```python
def verify_transaction(tx):
    # 验证签名
    # 验证余额
    # 验证时间戳
    return tx_valid
```

## 链上行为类型

| 行为 | 说明 | 奖励 |
|------|------|------|
| create_soul | 创建新灵魂 | +100 |
| complete_task | 完成任务 | +10~50 |
| daily_sign | 每日签到 | +1 |
| loyalty_upgrade | 忠诚度提升 | +5 |
| skill_create | 创建技能 | +20 |
| violate_contract | 违约 | -50 |

## 哈希算法
使用SHA-256保证区块安全

## 查询接口

```python
# 查询某行业的所有区块
industry_blocks = query_by_industry(0)

# 查询某时间段的区块
time_blocks = query_by_timerange(start, end)

# 获取链上总交易数
total_txs = get_total_transaction_count()
```

## 应用场景
- 记录AI完成的任务
- 验证AI的守信行为
- 计算长期忠诚度
- 追溯违约记录
