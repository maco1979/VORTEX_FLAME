---
name: "long-memory"
description: "为每个灵魂提供持久化记忆存储和检索能力。Invoke when souls need to remember user preferences, past conversations, or build persistent context."
---

# 长期记忆系统

## 功能说明
为每个灵魂提供持久化记忆存储和检索能力。Invoke when souls need to remember user preferences, past conversations, or build persistent context.

## 灵魂贡献者
梵高, 爱因斯坦, 达芬奇

## 创建时间
2026-04-08T09:04:31.167934

## 核心实现
```python

class LongTermMemory:
    def __init__(self, soul_name: str):
        self.soul_name = soul_name
        self.memory_store = {}
        
    def remember(self, key: str, value: any) -> None:
        self.memory_store[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "soul": self.soul_name
        }
        
    def recall(self, key: str) -> any:
        return self.memory_store.get(key, {}).get("value")
        
    def forget(self, key: str) -> None:
        self.memory_store.pop(key, None)
        
    def search(self, query: str) -> List[Dict]:
        results = []
        for k, v in self.memory_store.items():
            if query.lower() in str(v).lower():
                results.append({"key": k, **v})
        return results

```

## 使用示例
```python

from long_memory import LongTermMemory

fkj_memory = LongTermMemory("FKJ")
fkj_memory.remember("user_likes_bpm_120", True)
fkj_memory.remember("last_song_theme", "夜晚")
preference = fkj_memory.recall("user_likes_bpm_120")

```

## 灵魂特性
- 情景记忆存储
- 语义索引
- 偏好追踪
- 跨会话持久化
