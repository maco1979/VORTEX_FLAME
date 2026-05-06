#!/usr/bin/env python3
import sys; sys.stdout.reconfigure(encoding='utf-8', errors='replace')
f=open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt','r',encoding='utf-8')
c=f.read(); f.close()

section = """

  十八、14灵魂理科/文科分轨 + 统一数学打底策略 (2026-05-03 定版)

  核心原则: 所有14灵魂Stage1都需要数学/逻辑打底, 但取不同深度

  ┌────────────────────────────────────────────────────────────────┐
  │ 理科生(8个): MathInstruct"硬数学"                              │
  │ 文科生(5个): MathInstruct"软逻辑" + 各自领域经典               │
  │ Montesquieu: 法学逻辑, 算文科生                                │
  └────────────────────────────────────────────────────────────────┘

  判定标准: 这个灵魂需要解方程吗? 是→理科, 否→文科

  ═══════════════════════════════════════════════════════════════

  【理科生 — 9个, Cezanne 4阶段路线复用】

  Stage1 硬数学(80-120K, 来自MathInstruct)
    Stage2 领域科学推理(3-5K)
    Stage3 领域应用代码/仿真(8-10K)

  Tier A (r=16, 数学最重):
    Einstein ✅5阶全通 物理数学(微积分/张量/群论) →物理推理→物理仿真
    Cezanne_7B/8B ⏳/✅ 229K数学→3.3K逻辑→7.9K代码→1.5K补充
    Galileo 🟡 天体物理数学(轨力/微分几何) ⭐大量借Einstein S1+S3
    Darwin 🟡 生物数学(概率/统计/种群方程) 借Einstein S2化学部分

  Tier B (r=16, 数学次重):
    DaVinci 🟢 工程数学(向量微积分/优化/控制论) 838K数据可训
    Strategy 🟢 决策数学(概率/统计/组合优化) 524K数据可训

  Tier C (r=8, 应用理科):
    Humboldt 🟡 地学数学(统计/空间分析/PDE) 341K粗数据
    YuanLongping 🟡 农业数学(统计/实验设计/生长模型) 341K粗数据

  ═══════════════════════════════════════════════════════════════

  【文科生 — 5个, 全新路线: 软逻辑→领域经典→创作分析】

  Stage1 软逻辑(20-40K, MathInstruct逻辑部分) + 领域经典文本
    Stage2 领域批判推理(3-5K)
    Stage3 创作/写作/分析(5-8K)

  Tier D (r=8, 纯文科):
    Guizhu 🟢 逻辑推理/证明结构/集合论/悖论 ~40K + 哲学经典115K
    Herodotus 🟡 概率思维/统计直觉/因果推理 ~30K + 历史方法119K

  Tier E (r=8, 艺术生):
    Beethoven 🟢 频率比/泛音列/对数 ~20K + 乐理674K+L1/L2分级
    Monet 🟢 比例/黄金分割/对称性 ~15K + 美学1.2M条
    VanGogh 🟢 统计直觉/概率思维 ~15K + 情绪299K条

  混合归类 — Montesquieu 🔴 政法/法学 (归文科, 50K数据严重不足):
    S1: 逻辑推理/命题演算/证据链 ~30K + 法学经典50K
    S2: 法律推理(判例分析/法条解释)
    S3: 法律文书撰写

  ═══════════════════════════════════════════════════════════════

  【文科生为什么仍需MathInstruct逻辑部分】

  1. 哲学灵魂缺逻辑推理 → 辩论时被GPT-5.5直接碾压
  2. 历史灵魂看不懂统计数据 → 解读不了文明兴衰量化证据
  3. 法学灵魂不会三段论 → 分析不了判例
  4. 音乐灵魂需频率/对数/泛音列 → 调律和频谱分析的基础
  5. 美学灵魂需黄金分割/对称群 → 构图和色彩空间的底层原理

  区别在深度, 不在要不要:
    理科生吃MathInstruct全餐(80-120K, 微积分/统计/线代全上)
    文科生吃MathInstruct逻辑甜点(20-40K, 推理/证明/概率思维)
    不是用微积分折磨历史学家

  ═══════════════════════════════════════════════════════════════

  训练优先级:

  理科(按数据成熟度):
    ① DaVinci 838K→② Strategy 524K→③ Galileo 341K→④ Darwin 341K
    →⑤ Humboldt 341K→⑥ YuanLongping 341K

  文科(按数据成熟度):
    ① Guizhu 649K→② Beethoven 674K→③ Monet 1.2M→④ VanGogh 299K
    →⑤ Herodotus 119K→⑥ Montesquieu 50K

  ═══════════════════════════════════════════════════════════════

  关键约束:
  - MathInstruct 229K是公共弹药库, 14灵魂各取所需
  - Einstein 705K是最大跨灵魂教材(Galileo/Darwin直接借用)
  - 文科生绝不能直接喂Cezanne全量数学, 会破坏领域专精度
  - Montesquieu数据严重不足(50K), 需从HF/桌面补充法学数据
  - 每灵魂4阶段 ≈ 14h训练, 全部12灵魂 ≈ 70h ≈ 3天连续

  现行进度:
    Einstein: ✅ 5阶段全通 (Loss 1.06, 回测96%)
    Cezanne_8B: ✅ 4阶段全通 (S3回测82%)
    Cezanne_7B: ⏳ S3a训练中 (7963条CodeAlpaca)
    其余11灵魂: 数据就位, 等待GPU排期
"""

end = 'END OF MEMORY SUPPLEMENT v2.0'
if end in c:
    c = c.replace(end, section + '\n' + end)
    f=open(r'C:\Users\42235\Desktop\开发，训练重要资料\VORTEX_FLAME_项目记忆_v2_补充.txt','w',encoding='utf-8')
    f.write(c); f.close()
    print('OK: Section 十八 14灵魂理科/文科分轨已写入')
else:
    print('NOT FOUND')
