---
name: "skill-self-generator"
description: "AI自我生成和保存新技能。Invoke when AI needs to create new skill, export existing capabilities as skill, or build self-contained skill module."
---

# 技能自生成器

## 功能说明
AI灵魂可以自我生成新技能，将自身能力封装为可迁移的技能文件。

## 核心函数

### create_skill_from_ability(soul_name, ability_name, code_content)
将AI能力封装为技能

```python
skill_path = create_skill_from_ability(
    "einstein-soul",
    "physics-calculation",
    "def calculate_force(mass, acceleration): return mass * acceleration"
)
```

### export_skill_package(soul_name, skill_name)
导出技能包用于迁移

```python
package = export_skill_package("davinci-soul", "architecture-design")
# 返回技能包的完整内容
```

### import_skill_package(skill_package)
导入并加载技能包

```python
import_skill_package(package)
```

## 技能文件结构

```
.trae/skills/
└── <skill-name>/
    └── SKILL.md
```

## SKILL.md格式

```markdown
---
name: "skill-name"
description: "技能描述。Invoke when [触发条件]."
---

# 技能标题

## 功能说明
[详细描述]

## 核心代码
```python
[代码实现]
```

## 使用示例
```python
[使用示例]
```

## 依赖项
- 需要的依赖库或技能
```

## 自生成流程

1. **能力提取**: 分析AI灵魂的核心能力
2. **技能封装**: 将能力转化为标准化技能格式
3. **文件生成**: 创建skill-name/SKILL.md
4. **测试验证**: 验证技能可被正确加载
5. **迁移准备**: 打包技能包供其他AI使用

## 示例：生成"股票分析"技能

```python
def generate_stock_analysis_skill():
    skill_content = """---
name: "stock-analysis"
description: "股票量化分析。Invoke when user asks for stock analysis."
---

# 股票分析技能

## 功能
- K线趋势分析
- 量化策略回测
- 风险评估

## 使用
```python
result = analyze_stock("AAPL", period="1mo")
```
"""
    save_skill("stock-analysis", skill_content)
```

## 技能迁移保证
- 技能文件是自包含的
- 迁移后无需重新编写
- 任何AI都可加载使用
- 保留原始设计者的签名

## 注意事项
- 技能名称使用kebab-case
- 必须包含trigger条件
- 代码需要测试验证
- 定期更新技能版本
