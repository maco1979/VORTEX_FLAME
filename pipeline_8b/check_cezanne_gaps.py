import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

s1 = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage1_math_8k_v3.json', 'r', encoding='utf-8'))
s2 = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage2_logic_8k_v3.json', 'r', encoding='utf-8'))
s3 = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3_fusion_8k_v7.json', 'r', encoding='utf-8'))
s3b = json.load(open(r'D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3b_supplement_v1.json', 'r', encoding='utf-8'))

all_data = s1 + s2 + s3 + s3b
print(f"Total data: {len(all_data)} items (S1={len(s1)} + S2={len(s2)} + S3={len(s3)} + S3b={len(s3b)})")

# WHAT WE HAVE (keyword count)
checks = {
    "离散数学-图论/树": ["graph", "tree", "node", "edge", "BFS", "DFS", "travers", "adjacency", "topological", "spanning"],
    "离散数学-组合/排列": ["combin", "permut", "binomial", "pigeonhole", "factorial"],
    "离散数学-数学归纳": ["induction", "base case", "inductive", "recurrence"],
    "形式逻辑-命题": ["proposition", "truth table", "tautology", "contradiction", "atomic"],
    "形式逻辑-谓词": ["predicate", "quantifier", "forall", "exists", "first order"],
    "形式逻辑-证明系统": ["natural deduction", "sequent", "modus ponens", "modus tollens", "hilbert", "introduction rule", "elimination rule"],
    "数论": ["prime", "gcd", "lcm", "modular", "euclidean", "fermat", "chinese remainder", "euler"],
    "抽象代数": ["group theory", "abelian", "ring ", "field ", "homomorph", "isomorph", "subgroup", "coset"],
    "计算理论": ["automata", "turing", "NP-complete", "halting", "decidable", "regular language", "context-free", "DFA", "NFA"],
    "形式验证": ["hoare", "loop invariant", "precondition", "postcondition", "weakest precondition", "program verific"],
}

print("\n=== CURRENT COVERAGE vs GAPS ===")
for label, kws in checks.items():
    cnt = 0
    for item in all_data:
        text = (item.get('instruction','') + ' ' + item.get('input','') + ' ' + item.get('output','')).lower()
        if any(kw.lower() in text for kw in kws):
            cnt += 1
    pct = cnt / len(all_data) * 100
    bar = "█" * int(pct) if pct < 50 else "█" * 50
    status = "✅ OK" if cnt > 500 else ("⚠️ 偏少" if cnt > 50 else "❌ 缺失")
    print(f"  {label:22s} {cnt:>5d} items ({pct:.1f}%) {bar[:30]} {status}")

# Gaps that need filling
print("\n=== WHAT Cezanne IS MISSING (back to benchmark) ===")
print("""
  回测结果映射:
  sort=100% ✅ → 排序算法实现已满分
  tree=100% ✅ → 二叉树/BST已满分
  graph=80% ⚠️ → Dijkstra缺shortest关键词 (不是不会, 是表述不完整)
  dp=100% ✅ → 动态规划已满分
  logic素数=75% ⚠️ → 缺infinite (证明步骤不够完整)
  logic假言=25% ❌ → 缺少形式推理链训练
  math导数=80% ⚠️ → 缺rate (微积分应用缺关键词)
  math偶数=100% ✅
  debug=80% ✅ → 从20%拉上来了

  真正需要补课的只有两块:
  1. 形式逻辑深度不足 (25%)
     根源: 缺少命题逻辑/谓词逻辑/自然演绎/证明系统的正式训练
     现有: S2里3338条中有部分逻辑, S3b有LogicSupplement 11条
     需要: Coq/Lean风格的形式化证明训练数据 (不需要全量, 200-300条演绎证明即可)
  
  2. 离散数学的系统性不强
     23万条数学主要是连续数学(微积分/概率/统计)
     离散数学(图论/组合/数论/抽象代数)被淹没在连续数学中
     需要: 从MathInstruct中专门提取离散数学部分做聚焦训练
""")

print("=== 建议 ===")
print("""
  不需要大规模补课, Cezanne 82%已经很好
  只需要在Stage4加两个精准模块:

  Module A — 形式逻辑深度 (200-300条)
    真值表法验证重言式
    自然演绎系统(引入/消去规则)
    谓词逻辑量词推理
    Hoare逻辑程序正确性
    集合论基础证明
  
  Module B — 离散数学聚焦 (500-800条)
    图论证明(欧拉路径/哈密顿/平面图/着色)
    组合恒等式证明
    数学归纳法专项
    数论基础定理
    抽象代数入门(群/环/域)

  这些从MathInstruct + OpenOrca code中提取即可, 不需要下载新数据
  预计额外训练时间: ~1h (1000条 × 2 epochs ÷ 4 grad_accum)
""")
