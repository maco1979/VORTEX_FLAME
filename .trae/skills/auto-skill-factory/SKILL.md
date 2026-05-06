---
name: "auto-skill-factory"
description: "技能工厂 - 自动生成+更新技能。Invoke when user wants to automatically create, update, or manage skills based on requirements."
---

# 技能工厂

自动生成和更新SKILL的完整工厂系统。

## 核心功能

### 1. 自动生成新技能
### 2. 自动更新现有技能
### 3. 技能版本管理
### 4. 技能状态监控

## 使用方式

```
自动创建技能：[技能名称]
自动更新技能：[技能名称]
查看所有技能
```

## 技能列表

| 技能 | 名称 | 说明 |
|------|------|------|
| auto-skill-generator | 技能生成器 | 根据需求生成新技能 |
| auto-skill-updater | 技能更新器 | 更新现有技能内容 |
| auto-skill-factory | 技能工厂 | 生成+更新综合管理 |

## 自动创建流程

```python
def auto_create_skill(skill_name: str, requirements: str):
    # 1. 分析需求
    # 2. 生成技能内容
    # 3. 创建目录
    # 4. 写入文件
    # 5. 验证
```

## 自动更新流程

```python
def auto_update_skill(skill_name: str, updates: str):
    # 1. 读取现有技能
    # 2. 分析更新内容
    # 3. 保留frontmatter
    # 4. 更新body
    # 5. 验证
```

## 技能存储位置

```
.trae/skills/
├── auto-skill-generator/
│   └── SKILL.md
├── auto-skill-updater/
│   └── SKILL.md
├── auto-skill-factory/
│   └── SKILL.md
└── [其他技能]/
    └── SKILL.md
```

## 验证命令

```bash
# 列出所有技能
ls -la .trae/skills/

# 验证技能格式
python -c "from pathlib import Path; [print(p.name) for p in Path('.trae/skills').glob('*/SKILL.md')]"
```

## 最佳实践

1. **自动触发**：文件修改后自动更新技能
2. **版本控制**：保留技能历史版本
3. **格式验证**：确保SKILL.md格式正确
4. **描述清晰**：description说明触发条件
