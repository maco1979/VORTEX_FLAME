"""
Generate Cezanne Stage2 Logic Training Data
4-phase structure within Stage2:
  Phase1 (30%): Logic foundation - formal logic, propositional logic, set theory, causal reasoning
  Phase2 (50%): Weak point breakthrough - GPQA logic chains, calculus with derivation, comprehensive reasoning
  Phase3 (15%): USAMO logic review - convert memorized proofs to logical derivations
  Phase4 (5%):  Mixed review - ensure core math not lost
"""
import json, os, random

DATA_DIR = r"D:\VORTEX_FLAME\soul_training_data\cezanne"
AB_PATH = r"D:\VORTEX_FLAME\hermes_logs\ab_test_einstein_vs_cezanne.json"
TOTAL = 8000

P1_LOGIC = [
    {"instruction": "什么是命题逻辑？请解释命题、真值、逻辑联结词（与、或、非、蕴含、等价）的定义，并给出真值表。", "output": "命题逻辑是研究命题之间逻辑关系的数学分支。\n\n命题：可以判断真假的陈述句，真值只有True或False。\n\n逻辑联结词：\n1. 否定(¬P)：P为真则¬P为假，反之亦然\n2. 合取(P∧Q)：P和Q同时为真时为真\n3. 析取(P∨Q)：P或Q至少一个为真时为真\n4. 蕴含(P→Q)：仅当P真Q假时为假，等价于¬P∨Q\n5. 等价(P↔Q)：P和Q同真或同假时为真\n\n真值表：\nP Q | ¬P | P∧Q | P∨Q | P→Q | P↔Q\nT T |  F  |  T   |  T  |  T  |  T\nT F |  F  |  F   |  T  |  F  |  F\nF T |  T  |  F   |  T  |  T  |  F\nF F |  T  |  F   |  F  |  T  |  T\n\n核心推导规则：\n- 假言推理(Modus Ponens)：P→Q, P ⊢ Q\n- 假言否定(Modus Tollens)：P→Q, ¬Q ⊢ ¬P\n- 析取三段论：P∨Q, ¬P ⊢ Q"},
    {"instruction": "用形式逻辑证明：如果P→Q且Q→R，则P→R（假言三段论）。写出每一步的推导依据。", "output": "证明假言三段论：P→Q, Q→R ⊢ P→R\n\n推导链：\n1. P→Q         [前提1]\n2. Q→R         [前提2]\n3. 假设P为真    [条件假设法，为证明P→R]\n4. Q           [由1和3，假言推理(Modus Ponens)]\n5. R           [由2和4，假言推理]\n6. P→R         [由3-5，条件证明规则：若假设P可推出R，则P→R成立]\n\n推导依据标注：\n- 步骤3→4：假言推理(Modus Ponens)：P→Q为真且P为真，则Q为真\n- 步骤4→5：假言推理：Q→R为真且Q为真，则R为真\n- 步骤3-5→6：条件证明规则(CP Rule)：若在假设P下可推出R，则P→R为重言式\n\n逻辑有效性：此证明不依赖P/Q/R的具体内容，仅依赖→的语义定义，因此对任意命题P,Q,R成立。"},
    {"instruction": "什么是德摩根定律？请用真值表证明¬(P∧Q) ≡ ¬P∨¬Q。", "output": "德摩根定律(De Morgan's Laws)：\n1. ¬(P∧Q) ≡ ¬P∨¬Q\n2. ¬(P∨Q) ≡ ¬P∧¬Q\n\n真值表证明第一定律：\nP Q | P∧Q | ¬(P∧Q) | ¬P | ¬Q | ¬P∨¬Q\nT T |  T   |   F     |  F  |  F  |   F\nT F |  F   |   T     |  F  |  T  |   T\nF T |  F   |   T     |  T  |  F  |   T\nF F |  F   |   T     |  T  |  T  |   T\n\n对比第3列和第6列：完全相同，因此¬(P∧Q) ≡ ¬P∨¬Q。\n\n推导依据：\n- P∧Q为真当且仅当P和Q同时为真\n- 否定P∧Q等价于至少一个为假，即¬P或¬Q至少一个为真\n- 这恰好是¬P∨¬Q的定义\n\n应用：德摩根定律是布尔代数和数字电路设计的基础，用于简化逻辑表达式。"},
    {"instruction": "请解释集合论中的子集、并集、交集、补集、笛卡尔积的定义，并各举一例。", "output": "集合论基本运算：\n\n1. 子集(Subset)：A⊆B表示A的每个元素都属于B\n   例：A={1,2}, B={1,2,3}，则A⊆B\n\n2. 并集(Union)：A∪B = {x | x∈A 或 x∈B}\n   例：{1,2}∪{2,3} = {1,2,3}\n\n3. 交集(Intersection)：A∩B = {x | x∈A 且 x∈B}\n   例：{1,2}∩{2,3} = {2}\n\n4. 补集(Complement)：Aᶜ = {x | x∈U 且 x∉A}，U为全集\n   例：U={1,2,3,4}, A={1,2}, Aᶜ={3,4}\n\n5. 笛卡尔积(Cartesian Product)：A×B = {(a,b) | a∈A, b∈B}\n   例：{1,2}×{a,b} = {(1,a),(1,b),(2,a),(2,b)}\n\n集合论与逻辑的对应：\n- 并集∪ ↔ 析取∨\n- 交集∩ ↔ 合取∧\n- 补集ᶜ ↔ 否定¬\n- 子集⊆ ↔ 蕴含→\n\n这种对应关系是布尔代数的数学基础。"},
    {"instruction": "什么是因果推导？请区分充分条件、必要条件和充要条件，各举一例。", "output": "因果推导的三种条件关系：\n\n1. 充分条件(Sufficient Condition)：\n   若A成立则B一定成立，但B成立时A不一定成立\n   A→B（A是B的充分条件）\n   例：「下雨」是「地面湿」的充分条件（下雨→地面湿，但地面湿不一定因为下雨）\n\n2. 必要条件(Necessary Condition)：\n   若B成立则A一定成立，但A成立时B不一定成立\n   B→A（A是B的必要条件，即¬A→¬B）\n   例：「有氧气」是「燃烧」的必要条件（燃烧→有氧气，但有氧气不一定燃烧）\n\n3. 充要条件(Necessary and Sufficient Condition)：\n   A成立当且仅当B成立\n   A↔B（A和B互为充要条件）\n   例：「三角形三边相等」是「三角形三个角相等」的充要条件\n\n逻辑关系图：\n   充分：A→B（正向推导）\n   必要：¬A→¬B（逆否命题等价）\n   充要：A↔B（双向推导）\n\n常见逻辑错误：混淆充分与必要条件。例如「努力学习」是「考试及格」的充分条件，但不是必要条件（也可能靠运气及格）。"},
    {"instruction": "用反证法证明：不存在最大的素数。请写出完整的逻辑推导链，每一步标注依据。", "output": "反证法证明：不存在最大的素数\n\n推导链：\n1. 假设存在最大的素数p          [反证假设]\n2. 列出所有素数：p1,p2,...,pn=p   [由假设，素数有限]\n3. 构造数N = p1×p2×...×pn + 1   [构造新数]\n4. N不能被p1整除               [N/p1 = p2×...×pn + 1/p1，余数为1]\n5. N不能被p2整除               [同理，余数为1]\n6. N不能被任何pi整除            [由4、5推广，步骤3→6：归纳推理]\n7. 情况A：N是素数               [N大于pn，与pn是最大素数矛盾]\n8. 情况B：N不是素数              [则N有素因子q]\n9. q不在{p1,...,pn}中            [由6，N不能被任何pi整除，所以q≠任何pi]\n10. q是新的素数                  [由8和9]\n11. 矛盾：q大于pn或q不在列表中    [由7或10，与步骤2矛盾]\n12. 因此假设不成立                [由1-11，反证法]\n13. 结论：不存在最大的素数         [由12]\n\n每步推导依据：\n- 3→4,5,6：整除的定义与余数性质\n- 6→7或8：排中律（N要么是素数要么不是）\n- 8→9→10：素数定义与整除性\n- 11：与步骤1-2的假设矛盾\n- 12：反证法原理（假设导致矛盾则假设不成立）"},
    {"instruction": "请解释什么是一阶逻辑(First-Order Logic)，它比命题逻辑强在哪里？", "output": "一阶逻辑(First-Order Logic, FOL)是命题逻辑的扩展：\n\n命题逻辑的局限：\n- 只能处理完整命题的真假\n- 无法表达「对所有x，P(x)成立」这类量化陈述\n- 无法表达个体之间的关系\n\n一阶逻辑的增强：\n1. 引入量词：\n   - 全称量词∀x P(x)：对所有x，P(x)成立\n   - 存在量词∃x P(x)：存在某个x使P(x)成立\n\n2. 引入谓词：\n   - P(x)：x具有性质P\n   - R(x,y)：x和y具有关系R\n\n3. 量词的否定等价：\n   - ¬∀x P(x) ≡ ∃x ¬P(x)（不是所有都成立=存在一个不成立）\n   - ¬∃x P(x) ≡ ∀x ¬P(x)（不存在成立的=所有都不成立）\n\n示例：\n命题逻辑：只能写「所有人都会死」作为一个原子命题P\n一阶逻辑：∀x (Human(x) → Mortal(x))，可以推导出Human(Socrates)→Mortal(Socrates)\n\n推理规则：\n- 全称实例化：∀x P(x) ⊢ P(a)（对任意个体a）\n- 存在实例化：∃x P(x) ⊢ P(c)（c为新常量）\n\n一阶逻辑是数学证明的形式化基础，ZFC集合论就是用一阶逻辑表达的。"},
    {"instruction": "请用逻辑推导链证明：如果n是偶数，则n²也是偶数。每步标注推导依据。", "output": "证明：若n是偶数，则n²是偶数\n\n推导链：\n1. n是偶数                        [前提]\n2. n = 2k，其中k为某整数           [偶数的定义：可被2整除的整数]\n3. n² = (2k)² = 4k²              [代数运算：等式两边平方]\n4. n² = 2(2k²)                    [提取公因子2]\n5. 令m = 2k²                      [引入新变量m]\n6. m是整数                        [k是整数→k²是整数→2k²是整数，整数对乘法封闭]\n7. n² = 2m                        [由4和5，代入]\n8. n²是偶数                       [由7和偶数定义：n²=2×整数m]\n\n推导依据汇总：\n- 1→2：偶数的数学定义\n- 2→3：等式性质（等量代换+平方运算）\n- 3→4：乘法分配律/提取公因式\n- 4→5→6：整数集对乘法的封闭性\n- 7→8：偶数定义的逆向应用\n\n逻辑结构：直接证明法（从前提逐步推导到结论），每一步都是确定性推导，不依赖假设。"},
]

P2_GPQA = [
    {"instruction": "证明实对称矩阵的特征值都是实数。请写出完整的逻辑推导链，每步标注依据。", "output": "定理：实对称矩阵A的特征值都是实数\n\n推导链：\n1. 设λ是A的特征值，v是对应特征向量      [定义]\n2. Av = λv                              [特征值定义]\n3. A = Aᵀ                               [A是实对称矩阵的前提]\n4. v̄ᵀAv = v̄ᵀ(λv) = λ(v̄ᵀv)             [左乘v̄ᵀ，v̄是v的共轭]\n5. v̄ᵀAv = (Av̄)ᵀv = (Aᵀv̄)ᵀv = (Av̄)ᵀv   [由3，A=Aᵀ，转置性质]\n6. 又Av̄ = λ̄v̄                           [对共轭取特征方程]\n7. v̄ᵀAv = λ̄(v̄ᵀv)                      [由5和6]\n8. λ(v̄ᵀv) = λ̄(v̄ᵀv)                    [由4和7，等量代换]\n9. v̄ᵀv > 0                             [内积正定性，v≠0]\n10. λ = λ̄                              [由8两边除以v̄ᵀv]\n11. λ是实数                             [复数等于其共轭当且仅当它是实数]\n\n推导依据：\n- 4→8：等量代换（同一量的两种表达相等）\n- 9→10：非零量可约（数学基本性质）\n- 10→11：复数性质（z=z̄ ⟺ z∈ℝ）\n\n关键洞察：对称性(A=Aᵀ)是核心条件，它保证了v̄ᵀAv可以用两种方式计算，从而约束λ必须为实数。"},
    {"instruction": "用逻辑推导链解释黎曼假设：为什么ζ函数的非平凡零点在临界线Re(s)=1/2上如此重要？", "output": "黎曼假设的逻辑链：\n\n1. 定义：ζ(s) = Σ(1/nˢ)，解析延拓到全复平面     [黎曼zeta函数定义]\n2. ζ(s)有平凡零点s=-2,-4,-6,...                    [由函数方程推出]\n3. 非平凡零点都在0<Re(s)<1的临界带内              [已知结果]\n4. 黎曼假设：所有非平凡零点在Re(s)=1/2上           [未证明的猜想]\n\n为什么重要？逻辑推导：\n5. 素数定理：π(x)~x/ln(x)                        [已证明]\n6. 更精确估计：π(x) = Li(x) + O(x^θ)              [θ与零点位置相关]\n7. 若RH成立：θ=1/2+ε，误差项最小                   [由5→6→7]\n8. 若RH不成立：θ>1/2，素数分布估计误差更大          [逆命题]\n\n因果链：\n零点位置 → 误差项指数 → 素数分布精度\n\n具体影响：\n- RH成立 → 素数间隔|p_{n+1}-p_n| = O(√p_n · ln(p_n))\n- RH成立 → 所有奇完全数的下界大幅提升\n- RH成立 → 密码学中素数生成算法的可证明安全性\n\n推导依据：\n- 步骤5→6：素数计数函数与zeta零点的显式公式(Euler-Maclaurin)\n- 步骤6→7：零点实部直接决定误差项的增长率\n- 步骤7→8：这是等价关系，不是单向推导\n\nRH的验证进展：前10^13个非平凡零点都在Re(s)=1/2上，但数学证明仍未完成。"},
]

P2_CALCULUS = [
    {"instruction": "导数的物理意义是什么？请用逻辑推导链从位移→速度→加速度解释微积分的物理本质。", "output": "导数的物理意义——从位移到加速度的逻辑推导链：\n\n1. 位移s(t)：物体在时刻t的位置                    [定义]\n2. 平均速度 = Δs/Δt                             [速度的原始定义]\n3. 瞬时速度v(t) = lim(Δt→0) Δs/Δt = ds/dt       [导数的定义，取极限]\n4. v(t) = s'(t)                                  [导数记号]\n\n推导依据(1→4)：极限的定义保证了从「平均」到「瞬时」的严格过渡\n\n5. 平均加速度 = Δv/Δt                           [加速度的原始定义]\n6. 瞬时加速度a(t) = lim(Δt→0) Δv/Δt = dv/dt     [二阶导数]\n7. a(t) = s''(t) = d²s/dt²                       [二阶导数记号]\n\n物理本质的逻辑链：\n- 一阶导数s'(t)：描述变化率（速度是位移的变化率）\n- 二阶导数s''(t)：描述变化率的变化率（加速度是速度的变化率）\n- 积分是导数的逆运算：已知加速度→积分得速度→再积分得位移\n\n牛顿第二定律的逻辑推导：\n8. F = ma = m·s''(t)                            [牛顿第二定律]\n9. s''(t) = F/m                                  [变形]\n10. s(t) = ∫∫(F/m)dtdt                          [两次积分还原位移]\n\n核心洞察：微积分的本质是「变化」的数学语言。导数量化瞬时变化率，积分量化累积效应。牛顿发明微积分正是为了描述运动——这是微积分的物理根源。"},
    {"instruction": "用逻辑推导链解释积分的物理意义：为什么积分是导数的逆运算？", "output": "积分是导数的逆运算——逻辑推导链：\n\n1. 已知速度v(t)，求位移s(t)                       [问题设定]\n2. 在[t_i, t_{i+1}]小区间内，位移≈v(t_i)·Δt     [近似：用匀速代替变速]\n3. 总位移≈Σv(t_i)·Δt                            [黎曼和，所有小区间求和]\n4. s = lim(Δt→0) Σv(t_i)·Δt = ∫v(t)dt          [定积分的定义]\n5. 又v(t) = s'(t)                                [导数的定义]\n6. 所以∫s'(t)dt = s(t) + C                       [微积分基本定理]\n7. 这就是「积分是导数的逆运算」                    [结论]\n\n微积分基本定理的逻辑：\n8. 设F(x) = ∫ₐˣ f(t)dt                          [变上限积分定义]\n9. F'(x) = f(x)                                  [基本定理第一部分]\n10. ∫ₐᵇ f(x)dx = F(b) - F(a)                    [基本定理第二部分]\n\n推导依据：\n- 2→3：分割求和（黎曼和的构造）\n- 3→4：极限运算（精确化近似）\n- 4→6：由5和4，积分还原了被求导的函数\n- 9：关键步骤，F(x)的导数就是被积函数f(x)\n\n物理直觉：\n- 知道速度每一瞬间的值→积分→得到总位移\n- 知道位移→求导→得到速度\n- 两者互为逆运算，因为「累积」和「变化率」是一对逆过程\n\n应用逻辑链：\n力F(t)→积分→功W = ∫F·ds\n功率P(t)→积分→能量E = ∫P dt\n加速度a(t)→积分→速度v(t)→积分→位移s(t)"},
]

P2_REASONING = [
    {"instruction": "请用「拆解问题→建立模型→分步推导」的方法解决：一个水池有两个进水管和一个出水管，A管5小时注满，B管4小时注满，C管10小时放完。三管同时开，几小时注满？", "output": "拆解问题→建立模型→分步推导：\n\n【拆解问题】\n- 目标：求三管同时开时注满水池的时间\n- 已知：A管5小时注满，B管4小时注满，C管10小时放完\n- 变量：设水池总容量为V，时间为t\n\n【建立模型】\n- A管速率：V/5（每小时注入量）\n- B管速率：V/4（每小时注入量）\n- C管速率：V/10（每小时放出量）\n- 净注入速率 = V/5 + V/4 - V/10\n\n【分步推导】\n1. 通分：V/5 + V/4 - V/10\n2. 公分母=20：4V/20 + 5V/20 - 2V/20\n3. 净速率 = 7V/20\n4. 注满条件：净速率×t = V\n5. (7V/20)×t = V\n6. t = V/(7V/20) = 20/7 ≈ 2.86小时\n\n验证：\n- 2.86小时A注入：2.86/5 = 0.572池\n- 2.86小时B注入：2.86/4 = 0.715池\n- 2.86小时C放出：2.86/10 = 0.286池\n- 总计：0.572+0.715-0.286 = 1.001池 ✓\n\n推导依据：\n- 1→3：分数加减法（数学运算规则）\n- 4→6：方程求解（等式性质）\n- 验证：代入检验（数学严谨性）"},
    {"instruction": "用逻辑推导链分析：如果所有猫都是动物，所有动物都会死，那么所有猫都会死。这个推理为什么有效？", "output": "三段论推理的逻辑分析：\n\n前提1：所有猫都是动物（Cat→Animal）\n前提2：所有动物都会死（Animal→Mortal）\n结论：所有猫都会死（Cat→Mortal）\n\n逻辑推导链：\n1. ∀x(Cat(x)→Animal(x))     [前提1，全称命题]\n2. ∀x(Animal(x)→Mortal(x))   [前提2，全称命题]\n3. Cat(a)→Animal(a)           [全称实例化，对任意个体a]\n4. Animal(a)→Mortal(a)        [全称实例化]\n5. Cat(a)→Mortal(a)           [假言三段论：由3和4]\n6. ∀x(Cat(x)→Mortal(x))       [全称概括化]\n\n为什么有效：\n- 推理形式是假言三段论(Hypothetical Syllogism)\n- 这是有效推理形式：P→Q, Q→R ⊢ P→R\n- 不依赖「猫」「动物」「死」的具体含义\n- 只依赖→的传递性\n\n对比无效推理：\n- 所有猫都是动物，有些动物是狗 → 有些猫是狗？❌\n- 形式：∀x(C→A), ∃x(A∧D) ⊬ ∃x(C∧D)\n- 这不是有效推理形式，因为A→C不成立（动物不一定是猫）\n\n推导依据：\n- 1→3, 2→4：全称实例化规则\n- 3+4→5：假言三段论（已证明的有效推理规则）\n- 5→6：全称概括化规则"},
]

P3_USAMO = [
    {"instruction": "数学归纳法证明sum(1/n²)的收敛性时，为什么归纳法不够用？请用逻辑分析解释归纳法的局限性。", "output": "数学归纳法的局限性——逻辑分析：\n\n归纳法能证明什么：\n- 对每个有限n，Σ(1/k²) (k=1到n) 是有限值   ✓\n- 证明：基础步n=1成立，归纳步S(n)→S(n+1)成立  ✓\n\n归纳法不能证明什么：\n- 无穷级数的收敛性需要额外的极限论证      ✗\n- 归纳法只证明了「每个有限截断有界」，不等于「极限存在」\n\n逻辑断裂点：\n1. 归纳法：∀n∈ℕ, S_n是有限值                    [可证明]\n2. 收敛性需要：lim(n→∞) S_n存在且有限           [需要额外论证]\n3. 从1不能直接推出2                              [逻辑跳跃！]\n\n反例说明：\n- 调和级数Σ(1/k)：每个有限和都是有限值，但极限是+∞\n- 归纳法同样能证明「每个S_n有限」，但级数发散\n\n正确的证明需要：\n4. 证明S_n有上界（单调有界定理）\n5. S_n = Σ(1/k²) < 1 + Σ(1/k(k-1)) = 1 + Σ(1/(k-1) - 1/k) = 2\n6. {S_n}单调递增且有上界2\n7. 由单调有界定理，lim S_n存在且≤2\n8. 实际值π²/6≈1.6449\n\n推导依据：\n- 5：不等式放缩（1/k² < 1/(k(k-1))当k≥2）\n- 5→6：裂项求和\n- 6→7：实数完备性（单调有界序列必有极限）\n\n核心洞察：归纳法处理的是「每个有限步骤」，但分析学需要处理「无穷」——这是实数完备性公理的领域，超出归纳法的能力范围。"},
]

def generate_ab_fail_items():
    items = []
    if os.path.exists(AB_PATH):
        ab = json.load(open(AB_PATH, "r", encoding="utf-8"))
        for cat, data in ab.get("cezanne", {}).items():
            if cat == "overall":
                continue
            for detail in data.get("details", []):
                if not detail.get("pass", True):
                    q = detail.get("q", "")
                    items.append({
                        "instruction": f"请用逻辑推导链逐步解答以下问题，每步标注推导依据：{q}",
                        "output": f"[此题需要完整的逻辑推导链，每一步必须标注：1)当前步骤的结论 2)推导依据（定义/定理/推理规则）3)与前一步的逻辑关系]",
                        "source": "ab_fail_replay"
                    })
    return items

n_p1 = int(TOTAL * 0.30)
n_p2 = int(TOTAL * 0.50)
n_p3 = int(TOTAL * 0.15)
n_p4 = TOTAL - n_p1 - n_p2 - n_p3

print(f"Stage2 target: P1={n_p1}, P2={n_p2}, P3={n_p3}, P4={n_p4}", flush=True)

v3_path = os.path.join(DATA_DIR, "cezanne_55gb_v3.json")
print("Loading v3 for logic/calculus/discrete data...", flush=True)
with open(v3_path, "r", encoding="utf-8") as f:
    v3 = json.load(f)
if isinstance(v3, dict) and "data" in v3:
    v3 = v3["data"]

LOGIC_KW = ['逻辑','推理','证明','真值','命题','蕴含','矛盾','归纳','演绎','布尔',
            '集合运算','逻辑门','充分必要','反证','逆否','谓词','量词','形式化','符号逻辑',
            '因果','推导','前提','结论','充要']
CALC_KW = ['微积分','导数','积分','极限','微分','泰勒','级数','收敛','定积分','不定积分',
           '偏导数','梯度','链式法则','换元积分','分部积分']
DISCRETE_KW = ['离散','图论','组合','排列','二项式','递推','生成函数','鸽巢','欧拉路径',
               '哈密顿','树','割集','色数','匹配','网络流']

logic_items = []
calc_items = []
discrete_items = []
other_items = []

for item in v3:
    if not isinstance(item, dict) or len(item.get("output", "")) < 50:
        continue
    text = item.get("instruction", "") + " " + item.get("output", "")
    if any(kw in text for kw in LOGIC_KW):
        logic_items.append(item)
    elif any(kw in text for kw in CALC_KW):
        calc_items.append(item)
    elif any(kw in text for kw in DISCRETE_KW):
        discrete_items.append(item)
    else:
        other_items.append(item)

print(f"  logic: {len(logic_items)}, calc: {len(calc_items)}, discrete: {len(discrete_items)}", flush=True)
del v3

random.seed(3407)
random.shuffle(logic_items)
random.shuffle(calc_items)
random.shuffle(discrete_items)

stage2 = []

# Phase1: Logic foundation (30%)
p1_written = P1_LOGIC[:n_p1]
stage2.extend(p1_written)
remaining_p1 = n_p1 - len(p1_written)
if remaining_p1 > 0:
    stage2.extend(logic_items[:remaining_p1])
print(f"  P1 Logic: {min(len(p1_written) + remaining_p1, n_p1)} items", flush=True)

# Phase2: Weak point breakthrough (50%)
p2_written = P2_GPQA + P2_CALCULUS + P2_REASONING
stage2.extend(p2_written)
remaining_p2 = n_p2 - len(p2_written)
p2_calc = calc_items[:remaining_p2 // 3]
p2_discrete = discrete_items[:remaining_p2 // 3]
p2_logic = logic_items[len(p1_written):len(p1_written) + remaining_p2 // 3]
p2_fill = other_items[:max(0, remaining_p2 - len(p2_calc) - len(p2_discrete) - len(p2_logic))]
stage2.extend(p2_calc + p2_discrete + p2_logic + p2_fill)
print(f"  P2 Breakthrough: {len(p2_written) + len(p2_calc) + len(p2_discrete) + len(p2_logic) + len(p2_fill)} items", flush=True)

# Phase3: USAMO logic review (15%)
p3_written = P3_USAMO
stage2.extend(p3_written)
remaining_p3 = n_p3 - len(p3_written)
stage2.extend(logic_items[len(p1_written) + len(p2_logic):len(p1_written) + len(p2_logic) + remaining_p3])
print(f"  P3 USAMO Review: {len(p3_written) + remaining_p3} items", flush=True)

# Phase4: Mixed review (5%)
math_s1_path = os.path.join(DATA_DIR, "cezanne_stage1_math_8k.json")
with open(math_s1_path, "r", encoding="utf-8") as f:
    s1_data = json.load(f)
random.shuffle(s1_data)
stage2.extend(s1_data[:n_p4])
print(f"  P4 Mixed Review: {n_p4} items", flush=True)

random.shuffle(stage2)
stage2 = stage2[:TOTAL]

seen = set()
unique = []
for s in stage2:
    key = s.get("instruction", "")[:100]
    if key not in seen:
        seen.add(key)
        unique.append(s)
stage2 = unique[:TOTAL]

out_path = os.path.join(DATA_DIR, "cezanne_stage2_logic_8k.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(stage2, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {len(stage2)} items to {out_path}", flush=True)
print(f"Size: {os.path.getsize(out_path)/1024/1024:.1f}MB", flush=True)
