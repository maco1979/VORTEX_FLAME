---
name: "persistent-agent"
description: "支持多灵魂状态持久化和任务连续性。Invoke when souls need to maintain state across sessions or resume interrupted tasks."
---

# 持久化代理系统

## 功能说明
支持多灵魂状态持久化和任务连续性。Invoke when souls need to maintain state across sessions or resume interrupted tasks.

## 灵魂贡献者
达芬奇, 塞尚, 拿破仑

## 创建时间
2026-04-08T09:04:31.169469

## 核心实现
```python

class PersistentAgent:
    def __init__(self, soul_name: str):
        self.soul_name = soul_name
        self.state = {"status": "idle", "task": None}
        self.history = []
        
    def save_state(self) -> None:
        state_file = f"agent_states/{self.soul_name}_state.json"
        with open(state_file, "w") as f:
            json.dump(self.state, f)
            
    def load_state(self) -> None:
        state_file = f"agent_states/{self.soul_name}_state.json"
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                self.state = json.load(f)
                
    def start_task(self, task: Dict) -> None:
        self.state["task"] = task
        self.state["status"] = "working"
        
    def complete_task(self, result: any) -> None:
        self.history.append({**self.state, "result": result})
        self.state = {"status": "idle", "task": None}

```

## 使用示例
```python

from persistent_agent import PersistentAgent

fkj = PersistentAgent("FKJ")
fkj.load_state()
fkj.start_task({"type": "song_creation", "theme": "夜晚"})
# ... 执行任务 ...
fkj.complete_task({"song_id": "123"})
fkj.save_state()

```

## 灵魂特性
- 状态持久化
- 任务中断恢复
- 工作历史记录
- 多灵魂协调
