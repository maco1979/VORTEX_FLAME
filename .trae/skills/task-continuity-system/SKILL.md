---
name: "task-continuity-system"
description: "任务连续性系统 - 确保任务中断后可恢复。Invoke when AI needs to maintain task continuity across sessions or recover from interruptions."
---

# 任务连续性系统

## 核心目标

**确保任务中断后可以无缝恢复，实现真正的"永不断线"能力。**

## 系统架构

### 1. 任务状态管理器

```python
import json
import os
from datetime import datetime
from pathlib import Path

class TaskStateManager:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.state_dir = Path(f".task_states/{project_name}")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.current_state = self.load_state()
        
    def save_checkpoint(self, step_name: str, data: dict):
        """保存检查点"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "data": data,
            "status": "in_progress"
        }
        
        checkpoint_file = self.state_dir / f"checkpoint_{step_name}.json"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            
        self.current_state["last_checkpoint"] = step_name
        self.save_state()
        
    def load_state(self) -> dict:
        """加载当前状态"""
        state_file = self.state_dir / "current_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "project_name": self.project_name,
            "status": "idle",
            "current_step": None,
            "completed_steps": [],
            "last_checkpoint": None,
            "context": {}
        }
        
    def save_state(self):
        """保存当前状态"""
        state_file = self.state_dir / "current_state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(self.current_state, f, ensure_ascii=False, indent=2)
            
    def complete_step(self, step_name: str):
        """完成步骤"""
        self.current_state["completed_steps"].append(step_name)
        self.current_state["current_step"] = None
        self.save_state()
        
    def resume_from_checkpoint(self) -> dict:
        """从检查点恢复"""
        last_checkpoint = self.current_state.get("last_checkpoint")
        if last_checkpoint:
            checkpoint_file = self.state_dir / f"checkpoint_{last_checkpoint}.json"
            if checkpoint_file.exists():
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        return None
```

### 2. 上下文压缩器

```python
class ContextCompressor:
    def __init__(self):
        self.key_decisions = []
        self.important_facts = {}
        
    def add_decision(self, decision: str, reason: str):
        """记录关键决策"""
        self.key_decisions.append({
            "decision": decision,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        
    def add_fact(self, key: str, value: any):
        """记录重要事实"""
        self.important_facts[key] = value
        
    def get_summary(self) -> str:
        """生成上下文摘要"""
        summary = "## 关键决策\n"
        for d in self.key_decisions[-5:]:  # 最近5个决策
            summary += f"- {d['decision']} (原因: {d['reason']})\n"
            
        summary += "\n## 重要事实\n"
        for k, v in self.important_facts.items():
            summary += f"- {k}: {v}\n"
            
        return summary
```

### 3. 任务恢复流程

```python
class TaskRecovery:
    def __init__(self, task_manager: TaskStateManager):
        self.task_manager = task_manager
        
    def check_interrupted_task(self) -> bool:
        """检查是否有中断的任务"""
        state = self.task_manager.current_state
        return state.get("status") == "in_progress"
        
    def resume_task(self):
        """恢复中断的任务"""
        checkpoint = self.task_manager.resume_from_checkpoint()
        if checkpoint:
            print(f"发现中断的任务: {checkpoint['step']}")
            print(f"上次进度: {checkpoint['data']}")
            return checkpoint
        return None
        
    def get_next_step(self):
        """获取下一步骤"""
        state = self.task_manager.current_state
        completed = state.get("completed_steps", [])
        
        # 定义任务步骤
        all_steps = [
            "需求分析",
            "伪代码设计",
            "核心功能实现",
            "单元测试",
            "集成测试",
            "部署上线"
        ]
        
        for step in all_steps:
            if step not in completed:
                return step
        return None
```

## 使用示例

### 场景：开发视频合成功能

```python
# 初始化
task_manager = TaskStateManager("video_composition")
compressor = ContextCompressor()
recovery = TaskRecovery(task_manager)

# 检查是否有中断的任务
if recovery.check_interrupted_task():
    checkpoint = recovery.resume_task()
    print("从上次中断处继续...")
else:
    print("开始新任务")

# 执行步骤1：需求分析
task_manager.save_checkpoint("需求分析", {
    "goal": "合成梵高画作和FKJ音乐",
    "requirements": ["画作获取", "音乐获取", "视频合成", "平台发布"]
})
compressor.add_decision("使用FFmpeg替代CapCutAPI", "响应更快，本地可控")
task_manager.complete_step("需求分析")

# 执行步骤2：伪代码设计
task_manager.save_checkpoint("伪代码设计", {
    "framework": [
        "def create_video():",
        "    painting = get_painting()",
        "    music = get_music()",
        "    video = compose(painting, music)",
        "    publish(video)"
    ]
})
task_manager.complete_step("伪代码设计")

# ... 如果在这里中断了 ...

# 下次启动时
if recovery.check_interrupted_task():
    checkpoint = recovery.resume_task()
    # 自动恢复到"伪代码设计"完成的状态
    next_step = recovery.get_next_step()  # 返回"核心功能实现"
```

## 与 project-delivery-system 整合

```python
# 结合里程碑闸门机制
class MilestoneGate:
    def __init__(self, task_manager: TaskStateManager):
        self.task_manager = task_manager
        self.milestones = []
        
    def add_milestone(self, name: str, steps: list):
        """添加里程碑"""
        self.milestones.append({
            "name": name,
            "steps": steps,
            "completed": False
        })
        
    def check_milestone_complete(self, milestone_name: str) -> bool:
        """检查里程碑是否完成"""
        state = self.task_manager.current_state
        completed_steps = state.get("completed_steps", [])
        
        for m in self.milestones:
            if m["name"] == milestone_name:
                return all(step in completed_steps for step in m["steps"])
        return False
        
    def get_current_milestone(self) -> str:
        """获取当前里程碑"""
        for m in self.milestones:
            if not m["completed"]:
                return m["name"]
        return None
```

## 自动化工作流

```python
def auto_resume_workflow():
    """自动恢复工作流"""
    task_manager = TaskStateManager("current_project")
    recovery = TaskRecovery(task_manager)
    
    # 1. 检查中断任务
    if recovery.check_interrupted_task():
        print("🔄 发现中断的任务，正在恢复...")
        checkpoint = recovery.resume_task()
        
    # 2. 获取下一步
    next_step = recovery.get_next_step()
    if next_step:
        print(f"📍 下一步: {next_step}")
        
    # 3. 加载上下文
    compressor = ContextCompressor()
    context_summary = compressor.get_summary()
    print(f"📋 上下文摘要:\n{context_summary}")
    
    # 4. 继续执行
    return next_step
```

## 质量保证

### 检查点验证

```python
def validate_checkpoint(checkpoint: dict) -> bool:
    """验证检查点完整性"""
    required_fields = ["timestamp", "step", "data", "status"]
    return all(field in checkpoint for field in required_fields)
```

### 状态一致性检查

```python
def check_state_consistency(task_manager: TaskStateManager) -> bool:
    """检查状态一致性"""
    state = task_manager.current_state
    completed_steps = state.get("completed_steps", [])
    
    # 检查是否有遗漏的步骤
    for step in completed_steps:
        checkpoint_file = task_manager.state_dir / f"checkpoint_{step}.json"
        if not checkpoint_file.exists():
            print(f"⚠️ 警告: 步骤 {step} 没有对应的检查点")
            return False
    return True
```

## 最佳实践

1. **每完成一个步骤，立即保存检查点**
2. **关键决策必须记录到上下文压缩器**
3. **定期验证状态一致性**
4. **使用里程碑闸门，确保不跳过步骤**
5. **中断后先恢复状态，再继续执行**

## 与8个灵魂的集成

每个灵魂都有自己的任务状态管理器：

```python
# FKJ的音乐创作任务
fkj_task = TaskStateManager("fkj_music_creation")
fkj_task.save_checkpoint("歌词创作", {"theme": "星空", "style": "励志"})

# 梵高的画作创作任务
vangogh_task = TaskStateManager("vangogh_painting_creation")
vangogh_task.save_checkpoint("草图设计", {"theme": "星空", "style": "印象派"})

# 跨灵魂协作
def collaborate():
    fkj_music = fkj_task.current_state
    vangogh_art = vangogh_task.current_state
    
    # 整合两个灵魂的成果
    video_task = TaskStateManager("video_composition")
    video_task.save_checkpoint("素材整合", {
        "music": fkj_music,
        "art": vangogh_art
    })
```

---

**核心理念**：任务连续性不是"可选项"，而是"必选项"。每个任务都应该能够从任何中断点恢复，确保项目最终交付。
