# VORTEX FLAME — Architecture Notes

> 2026-05-31 · 设计讨论记录
> 主题：SKILL/MCP 生态的未来形态，以及双模型（LLM + C-JEPA）对 SKILL 的分工定位

---

## 一、SKILL/MCP 的生产模式演进

### 1.1 当前阶段：人手写（现在）

MCP Server 是桥接代码，连接有真实 API 约束的外部系统（Postgres、Brave Search、Filesystem、ComfyUI 等）。Auth token、rate limit、错误重试、边界条件——一个失误就可能导致数据库误操作或 API 欠费。

**LLM 可以生成骨架，但可靠性验证必须由人完成。**

### 1.2 中期阶段：AI 写，人审（1–2 年）

```
OpenAPI Spec / SDK 文档
        │
        ▼
    AI 生成完整 MCP Server + 测试
        │
        ▼
    人工审查：auth 安全？错误处理？边界条件？
        │
        ▼
    部署到社区市场，带 AI-generated + human-verified 标签
```

Claude Computer Use、OpenAI Function Calling 已验证这个方向的可行性——模型理解 API 文档后可动态生成调用代码，不总是需要一个预建 MCP Server。

### 1.3 长期阶段：自演进 + 边界定义（3 年+）

**MCP Server 作为中间层被逐步消解。** 模型直接读 API 文档、推理 auth 流程、处理错误、适配合约变更。

SKILL 从代码演变为"认知模板"——「我知道该怎么跟这类系统交互」的知识表征，而非硬编码的适配器。

**核心约束：外部世界的不可逆性。**
- 写文件、转账、发邮件 → 没有 undo → **人类永远保留否决权**
- 纯内部推理类 SKILL（架构审查、代码分析） → 可完全自演进

**人的角色：从编写者 → 边界定义者**（定义"哪些操作永远不允许"）。

---

## 二、SKILL/MCP 的双模型分工

### 2.1 架构全景

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  顶层 LLM (Ollama / Mistral-7B)                      │
│  职责：语言理解、对话、工具调用、结果自然语言表述       │
│                                                     │
│  「帮我查询数据库里最近 30 天的销售趋势」              │
│         │                                           │
│         ▼ 调用 Postgres MCP                          │
│  拿到 [120, 135, 110, ...]                          │
│         │                                           │
│         ▼ 转为结构化数据，传给 C-JEPA                  │
│                                                     │
├─────────────────────────────────────────────────────┤
│  底层 C-JEPA (10 个领域变体)                          │
│  职责：因果推理、预测、反事实、表征学习                 │
│                                                     │
│  「给定这个趋势序列，下个月的预测是什么？」             │
│  「如果关税提高 5%，趋势会怎么变？」(反事实推理)        │
│         │                                           │
│         ▼ 预测结果                                   │
│  「月度增长约 8%，但波动率扩大」                       │
│         │                                           │
│         ▼ 返回 LLM                                  │
│                                                     │
├─────────────────────────────────────────────────────┤
│  LLM 包装成自然语言输出                                │
│  「过去 30 天下月预计增长 8%，                        │
│    但波动范围较大 (±12%)，建议关注供应链稳定性」        │
└─────────────────────────────────────────────────────┘
```

### 2.2 顶层 LLM 使用 SKILL/MCP（当前已实现）

MCP 是 LLM 的 **"手和眼"**——让 LLM 获得外部世界的感知和行动能力。数据库查询、文件读写、浏览器控制、图像生成，这些都是 LLM 自身不具备的能力。

[soul_orchestrator.py](file:///D:/VORTEX_FLAME/soul_orchestrator.py) 中每条灵魂定义了 `skills` 和 `tools` 映射。

### 2.3 底层 C-JEPA 使用 SKILL/MCP（未实现，接口已预留）

C-JEPA 使用 SKILL 的三种模式：

| 模式 | 示例 | 说明 |
|------|------|------|
| **数据注入** | FIN-JEPA 通过 MCP 实时获取 Bloomberg 数据 | 代替离线批处理，支持实时因果推理 |
| **干预探针** | PHYS-JEPA 调用物理模拟器验证预测 | 作为 ground truth 验证反事实推理 |
| **行动输出** | C-JEPA 预测的最优行动，通过 MCP 执行 | "按这个配比调整反应釜温度" → 工业控制协议 |

[five_layer_jepa/causal_jepa.py](file:///D:/VORTEX_FLAME/five_layer_jepa/causal_jepa.py) 中的 `CJEPALayer.aux_conditioner` 模块就是为此预留的接口——等在线推理管线打通后，MCP 数据可直接作为辅助条件注入。

### 2.4 核心区别

| 维度 | LLM + MCP | C-JEPA + MCP |
|------|-----------|--------------|
| 界面 | 语言 | 结构 |
| 核心目标 | 感知 + 行动 | 感知 + 验证 |
| 用户关注 | 「用户想干什么」 | 「这个世界会发生什么」 |
| 时间方向 | 当前 + 过去（检索） | 未来（预测）+ 反事实（假设推理） |
| 输出 | 自然语言 | 结构表征 |

**不是二选一——LLM 负责对话和编排，C-JEPA 负责因果推理。MCP/SKILL 同时为两者服务。**

---

## 三、对 VORTEX FLAME 社区市场的影响

### 3.1 当前缺口

| 能力 | 状态 |
|------|------|
| 从上游源发现技能 | 已有（GitHub / awesome 适配器框架） |
| 搜索 / 浏览 / 筛选技能 | 已有（API + design.html 前端） |
| 点击安装到本地 | **缺失** |
| 安装后自动写配置文件 | **缺失** |
| 社区贡献 / 上传新技能 | **缺失** |

### 3.2 中期路线

```
Phase A: 本地缓存驱动
  skill_cache/ 目录扫描 → 作为 tier4_community 伪数据源

Phase B: 真实社区管道
  GitHub Topics API (topic:mcp-server) → 自动发现
  mcpservers.org → 元数据补全
  huggingface spaces → SKILL 代码分发

Phase C: 安装引擎
  一键安装 → npm/pip install → 写 mcp_servers/ 配置 → 注册到 skill_registry
```

### 3.3 长期愿景：SKILL 作为生态通货

当 SKILL 从代码演变为认知模板，社区市场的角色也会变：
- **发现层**：搜索「我能用哪些工具？」 → 语义匹配而非关键词匹配
- **信任层**：AI-generated + human-verified 分级认证
- **编排层**：多 SKILL 自动组合成工作流（类似 Zapier 的 AI 原生版本）
- **进化层**：SKILL 基于使用频率和成功率自动迭代

---

## 四、分布式 SKILL/MCP 获取方案

### 核心问题

VORTEX FLAME 的用户是分布式的——每个人在本地都有自己的实例、自己的 `mcp_servers/` 配置、自己调出来的 SKILL 变体。如何让这些分散的知识汇聚成一个社区生态，而不是永远困在各自本地？

**两条路**：被动抽取（系统发现用户有什么）vs 主动上传（用户自己发布）。答案不是二选一，而是**按信任梯度分层走**。

---

### 4.1 获取模式矩阵

```
                    被动发现                              主动贡献
                ←─────────────────────────────────────────────────→

隐私度：  高                                    中                        低
信任度：  零信任                              中等信任                  高度信任
质量度：  未筛选                              中等                      高（人为意图）
```

**四种获取方式的组合方案：**

#### 模式 A：本地扫描 + 选择性公开（被动 → 主动授权）

```
┌─────────────────────────────────────────────────────────────┐
│  VORTEX FLAME 本地实例                                       │
│                                                             │
│  mcp_servers/                skill_cache/                   │
│  ├── my-custom-db.json       ├── custom-skill-1.json        │
│  ├── internal-api.json       └── finance-helper.json        │
│  └── comfyui.json                                          │
│         │                                                   │
│         ▼  系统扫描本地配置                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  发现 3 个本地 SKILL/MCP                             │   │
│  │                                                     │   │
│  │  ✅ my-custom-db      [自定义 Postgres 连接器]        │   │
│  │  ✅ internal-api      [公司内部 API 桥接]   🔒 敏感   │   │
│  │  ✅ finance-helper    [财务数据处理]          📤 可分享│   │
│  │                                                     │   │
│  │  选择性公开：你勾选哪些进入社区？                       │   │
│  │  [ ] my-custom-db      [ ] internal-api               │   │
│  │  [✓] finance-helper    ← 一键发布到 tier4_community   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**类比**：像 `npm audit` 告诉你本地有什么包、哪些有过期版本——但不是自动上传，是你自己选择。

**优点**：降低贡献门槛（不需要用 git），用户不需要"想好要发布"才行动
**风险**：扫描范围必须透明可控，不能触及非 SKILL 文件

#### 模式 B：一键发布 / 手动上传（主动贡献）

用户主动创建 → 打包 → 提交到社区：

```
用户本地：
  写了一个 SKILL (mcp_servers/custom-skill.json)
        │
        ▼  点击"发布到社区"
  ┌─────────────────────────────┐
  │  Publish SKILL              │
  │                             │
  │  Name:     custom-skill     │
  │  Category: database         │
  │  Tier:     tier4_community  │
  │  Tags:     postgres,sql     │
  │  Soul Mapping: cezanne      │
  │                             │
  │  [发布]                      │
  └─────────────────────────────┘
        │
        ▼
  社区市场 API 注册 → 可被搜索/安装
```

**这是传统开源模式——用户有明确意图，质量最高。**

#### 模式 C：差异化同步（增量贡献）

用户从社区安装了 SKILL，做了本地修改。系统 diff：

```
社区版 postgres-mcp v1.2.0
        │
用户本地修改：
  + 添加了连接池大小可配
  + 加了 read_only_mode 开关
  + 修了一个 SQL 注入防护的 bug
        │
        ▼ 系统检测到差异
  「你修改了社区的 postgres-mcp，是否提交改进？」
  
  Contribution 类型:
  [✓] Bug fix → 直接提交 PR
  [✓] Feature → 提交 fork 版本
  [ ] 仅本地使用 → 忽略
```

**类比**：Git fork → PR 流程，但自动化了"发现修改"这一步。
**最高价值**：社区的 SKILL 可以像开源代码一样持续进化，不同用户的改进自动汇聚。

#### 模式 D：匿名使用统计（需求信号，不分享代码）

不传代码，只传聚合元数据：

```json
{
  "skill_id": "postgres-mcp",
  "anonymized_stats": {
    "call_count_per_day": 47,
    "success_rate": 0.96,
    "avg_latency_ms": 82,
    "most_used_tools": ["sql_query", "schema_inspect"],
    "custom_overrides": 3
  }
}
```

**不暴露用户数据，只暴露"这个 SKILL 好不好用"的信号。** 这驱动了两个关键反馈回路：
- 用户搜索时 → 排序按真实使用率而非星标数
- 开发者决策 → 知道社区真正需要什么工具（"40%用户有自定义数据库连接器 → 说明现有方案不够 → 值得做一个官方版"）

---

### 4.2 信任分层体系

不是所有 SKILL 生来平等。围绕"它来自哪里、谁验证过它、它在多少真实环境跑过"来分层：

```
                      Trust Tier
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  Tier 1: Private (私有)                              │
  │  本地扫描发现，但不公开。仅自己使用。                    │
  │  「我的公司内部 API 桥接」🔒                           │
  │                                                      │
  │  Tier 2: Team (团队内共享)                            │
  │  同一命名空间/组织内的用户可见。                        │
  │  「团队的部署脚本」👥                                  │
  │                                                      │
  │  Tier 3: Community Draft (社区草稿)                   │
  │  公开发布，状态为 pending，未被验证。                   │
  │  「我写的 CustomDB 连接器，试试看」⚠️                   │
  │                                                      │
  │  Tier 4: Community Verified (社区验证)                │
  │  被 ≥3 个其他用户安装且运行过，通过自动安全检查。        │
  │  「Postgres 连接器，300+ 安装，97% 成功率」✅            │
  │                                                      │
  │  Tier 5: Official (官方维护)                          │
  │  VORTEX FLAME 核心团队维护，CI 测试，安全审计。         │
  │  「内置 ComfyUI 适配器」🛡️                             │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

### 4.3 完整用户旅程

一个新用户加入 VORTEX FLAME 的完整体验：

```
第 1 天：本地探索
  ├── 打开 SKILL 市场 → 浏览 → 安装了 4 个社区 SKILL
  ├── 发现缺一个"飞书审批"的 SKILL → 自己写了一个本地版
  │
第 7 天：发现需求信号
  ├── 匿名统计显示："13% 用户有自定义审批类 SKILL"
  ├── 触发社区关注：「审批自动化是真实需求」
  │
第 14 天：贡献
  ├── 用户把"飞书审批"一键发布到社区（tier3 draft）
  ├── 3 个其他用户安装并运行 → 自动升级到 tier4 verified
  │
第 30 天：核心采纳
  ├── VORTEX FLAME 核心团队看到这个 SKILL 火了
  ├── 纳入 tier1_core，成为官方维护的内置模块
  └── 原作者的贡献记录永久保存在 SKILL 元数据中
```

### 4.4 技术实现要点

```
D:\VORTEX_FLAME\skills\
├── local_skil_scanner.py    # 模式 A：扫描本地 mcp_servers/ + skill_cache/
├── skill_publisher.py       # 模式 B：一键打包 + 提交到社区
├── skill_diff_engine.py     # 模式 C：对比本地版 vs 社区版, 生成 diff
├── usage_telemetry.py       # 模式 D：匿名使用统计收集 (opt-in)
└── trust_engine.py          # 信任分层自动计算 (安装数 + 成功率 + 安全扫描)
```

### 4.5 隐私边界 (硬约束)

- **模式 A 的扫描不触碰非 SKILL 目录**——只扫 `mcp_servers/` 和 `skill_cache/`
- **模式 D 的遥测必须是 opt-in**——默认关闭，用户主动开启
- **所有上传的 SKILL 自动剥离**：本地路径、IP 地址、API key、token
- **撤回权**：用户可随时从社区移除自己发布的 SKILL（但已安装的实例不会受影响，类比 `npm unpublish`）

---

## 五、即刻行动项

- [ ] 修复 `_fetch_github_registry` / `_fetch_awesome_mcp` 的数据源 URL
- [ ] 或：先用 `skill_cache/` 本地扫描补上 tier4_community 的空缺
- [ ] `CJEPALayer` 的 MCP 数据注入路径设计（连接 `aux_conditioner` 到实时 API）
- [ ] 社区技能安装引擎（一键从市场安装到本地）
