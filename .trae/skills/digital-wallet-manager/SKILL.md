---
name: "digital-wallet-manager"
description: "管理AI数字钱包、查询余额、记录交易。Invoke when AI needs to check balance, transfer digital RMB, or view transaction history."
---

# 数字钱包管理器

## 功能说明
管理AI灵魂的数字钱包，支持余额查询、转账、交易记录查看。

## 核心函数

### 查询余额
```python
def query_balance(wallet):
    return wallet.get("balance", 0.0)
```

### 充值（仅系统调用）
```python
def deposit(wallet, amount, source="system"):
    wallet["balance"] += amount
    wallet["total_earned"] += amount
    record_transaction(wallet, amount, "deposit", f"充值来源: {source}")
```

### 消费/支出
```python
def withdraw(wallet, amount, purpose="consumption"):
    if wallet["balance"] >= amount:
        wallet["balance"] -= amount
        wallet["total_spent"] += amount
        record_transaction(wallet, -amount, "withdraw", purpose)
        return True
    return False
```

### 转账
```python
def transfer(from_wallet, to_wallet, amount, memo=""):
    if withdraw(from_wallet, amount, f"转账至: {to_wallet['soul_code']}"):
        deposit(to_wallet, amount, f"转账来自: {from_wallet['soul_code']}")
        return True
    return False
```

### 交易历史
```python
def get_transaction_history(wallet, limit=10):
    return wallet.get("transactions", [])[-limit:]
```

## 钱包数据结构
```python
{
    "soul_code": "VF-音乐-0001052226",
    "industry": "音乐创作",
    "balance": 0.0,
    "total_earned": 0.0,
    "total_spent": 0.0,
    "value_factor": 1.2,
    "block_count": 0,
    "transactions": []
}
```

## 交易类型
| 类型 | 说明 |
|------|------|
| deposit | 充值/收入 |
| withdraw | 消费/支出 |
| block_reward | 区块奖励 |
| task_reward | 任务奖励 |
| transfer | 转账 |

## 行业价值系数
用于计算收益：收益 = 基础值 × 价值系数

## 注意事项
- 所有交易自动记录到transactions数组
- 余额不足时withdraw返回False
- 转账自动扣除手续费(如有)
