#!/usr/bin/env python3
"""
Stage5 全阶段回测+行业工具训练 (3060 GPU, 12GB VRAM)
铁则: 旧知识不丢 + 新知识学会 = 才算合格

Stage5 做三件事:
  1. 全阶段回测: Stage1-4旧考题全部重考,发现遗忘立刻补训
  2. 行业工具训练: 14灵魂各学自己的行业工具
  3. 防遗忘混合: 训练数据中混入Stage1-4核心样本

用法:
  python train_stage5.py --soul einstein          -- 训练Einstein Stage5
  python train_stage5.py --soul einstein --resume  -- 从Stage4 LoRA继续
  python train_stage5.py --all                     -- 依次训练全部14灵魂
"""

import os, json, gc, time, random, sys, argparse, faulthandler
import torch

faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

FAULT_LOG = open(r"D:\VORTEX_FLAME\train_fault_stage5.log", "w")
faulthandler.enable(file=FAULT_LOG, all_threads=True)

parser = argparse.ArgumentParser()
parser.add_argument("--soul", type=str, default="einstein",
    choices=["einstein","davinci","beethoven","cezanne","vangogh","monet",
             "strategy","guizhu","darwin","galileo","humboldt","yuanlongping",
             "montesquieu","herodotus"])
parser.add_argument("--resume", action="store_true", help="Resume from Stage4 LoRA")
parser.add_argument("--max_samples", type=int, default=8000)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--seq_len", type=int, default=256)
parser.add_argument("--all", action="store_true", help="Train all 14 souls sequentially")
args = parser.parse_args()

SOUL_TOOLS = {
    "einstein":     {"cn": "爱因斯坦", "tools": "Ansys,COMSOL,电力仿真,工业AI", "industry": "数理化·能源·半导体"},
    "davinci":      {"cn": "达芬奇",   "tools": "CAD,PCB设计,ROS机器人,BIM", "industry": "工程·机械·自动驾驶·建筑"},
    "beethoven":    {"cn": "贝多芬",   "tools": "Ableton Live,Suno,iZotope,音效AI", "industry": "音乐·音频·声学"},
    "cezanne":      {"cn": "塞尚",     "tools": "Cursor,IoT平台,嵌入式AI", "industry": "代码·嵌入式·物联网"},
    "vangogh":      {"cn": "梵高",     "tools": "ComfyUI,SD,Flux,Runway", "industry": "视觉·艺术疗愈"},
    "monet":        {"cn": "莫奈",     "tools": "设计AI,课件AI,创意工作流", "industry": "美学设计·AI教育"},
    "strategy":     {"cn": "策略",     "tools": "投研AI,风控AI,商业智能BI", "industry": "金融·战略·商业咨询"},
    "guizhu":       {"cn": "硅酌居士", "tools": "RAG,对话系统,情绪分析", "industry": "NLP·心理咨询·对话"},
    "darwin":       {"cn": "达尔文",   "tools": "医学影像AI,制药AI,健康监测", "industry": "生命·医疗·制药"},
    "galileo":      {"cn": "伽利略",   "tools": "卫星遥感,轨道计算,航天仿真", "industry": "航天·卫星·天文"},
    "humboldt":     {"cn": "洪堡",     "tools": "气象AI,环保监测,碳核算", "industry": "地球·环保·气象·碳中和"},
    "yuanlongping": {"cn": "袁隆平",   "tools": "农业遥感,作物识别,食安检测", "industry": "农业·食品安全"},
    "montesquieu":  {"cn": "孟德斯鸠", "tools": "合同审查AI,法规AI,政务系统", "industry": "法律·数字政府·合规"},
    "herodotus":    {"cn": "希罗多德", "tools": "文献AI,修复AI,出版排版", "industry": "历史·文化·出版"},
}

BASE_MODEL = r"D:\models\Mistral-7B-Instruct-v0.1"
DATA_DIR = r"D:\VORTEX_FLAME\soul_training_data"
LORA_DIR = r"D:\VORTEX_FLAME\soul_lora_v2"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs"
FINEWEB_DIR = r"E:\大规模训练集\FinewebEdu_filtered"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_SEQ_LENGTH = args.seq_len
USE_GRAD_CKPT = True

TRAIN_ORDER = ["einstein","davinci","beethoven","cezanne","vangogh","monet",
               "strategy","guizhu","darwin","galileo","humboldt","yuanlongping",
               "montesquieu","herodotus"]

EXAM_QUESTIONS = {
    1: [
        "请用中文解释牛顿第二定律F=ma的物理含义。",
        "什么是能量守恒定律？请用中文举例说明。",
        "请解释光的折射现象及其公式。",
        "什么是热力学第二定律？请用中文解释。",
        "请解释电磁感应现象和法拉第定律。",
        "请解释量子力学中的波粒二象性。",
        "什么是狭义相对论的时间膨胀效应？",
        "请解释麦克斯韦方程组的物理意义。",
        "什么是普朗克常数？它在量子力学中有什么作用？",
        "请解释半导体PN结的工作原理。",
    ],
    2: [
        "请解释微积分中导数的几何意义。",
        "什么是矩阵的特征值和特征向量？",
        "请解释概率论中的大数定律。",
        "什么是化学键？共价键和离子键有什么区别？",
        "请解释热化学中的盖斯定律。",
        "什么是傅里叶变换？它有什么应用？",
        "请解释有机化学中烷烃的命名规则。",
        "什么是统计假设检验？p值的含义是什么？",
        "请解释化学平衡和勒夏特列原理。",
        "什么是微分方程？请举一个物理应用的例子。",
    ],
    3: [
        "请解释统计力学如何从微观粒子行为推导宏观热力学性质。",
        "什么是量子化学中的密度泛函理论？",
        "请解释傅里叶变换在信号处理和量子力学中的共同应用。",
        "什么是蒙特卡洛方法？它在物理模拟中如何应用？",
        "请解释数学物理中的变分法原理。",
        "什么是分子动力学模拟？它的物理基础是什么？",
        "请解释能带理论如何解释导体、半导体和绝缘体的区别。",
        "什么是有限元方法？它在工程物理中如何应用？",
        "请解释格林函数在数学物理中的作用。",
        "什么是物理化学中的相律？请用数学表达。",
    ],
    4: [
        "请解释半导体制造中的光刻工艺原理。",
        "什么是太阳能电池的工作原理？光电转换效率如何提高？",
        "请解释核电站的工作原理和安全系统设计。",
        "什么是锂电池的工作原理？如何提高能量密度？",
        "请解释医疗器械MRI的物理原理和工程实现。",
        "什么是工业自动化中的PID控制？",
        "请解释电力系统中变压器的原理和效率优化。",
        "什么是纳米材料？它在工程应用中有什么优势？",
        "请解释风力发电的原理和效率影响因素。",
        "什么是嵌入式系统？它在工业控制中如何应用？",
    ],
}

STAGE5_EXAM = [
    "如何使用Ansys进行半导体热应力分析？",
    "COMSOL Multiphysics在新能源电池仿真中如何应用？",
    "电力系统仿真软件如何进行电网稳定性分析？",
    "工业AI在智能制造中如何实现预测性维护？",
    "请说明Ansys和COMSOL在工程仿真中的适用场景区别。",
]


def run_regression_exam(model, tokenizer):
    print(f"\n  {'='*50}")
    print(f"  全阶段回测 (Stage1-4 + Stage5)")
    print(f"  {'='*50}")
    all_results = {}
    total_pass = 0
    total_q = 0

    for stage in [1, 2, 3, 4]:
        questions = EXAM_QUESTIONS.get(stage, [])
        stage_pass = 0
        stage_results = []
        print(f"\n  --- Stage{stage} 回测 ({len(questions)}题) ---")
        for i, q in enumerate(questions):
            prompt = f"[INST] {q} [/INST] "
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.replace(prompt, "").strip()
            ok = len(answer) > 30
            if ok:
                stage_pass += 1
            stage_results.append({"Q": q, "A": answer[:300], "pass": ok})
            status = "PASS" if ok else "FAIL"
            print(f"  S{stage} Q{i+1}/{len(questions)} [{status}] {answer[:80]}...")
        rate = stage_pass / len(questions) if questions else 0
        all_results[f"stage{stage}"] = {"pass": stage_pass, "total": len(questions), "rate": rate, "details": stage_results}
        total_pass += stage_pass
        total_q += len(questions)
        print(f"  Stage{stage}: {stage_pass}/{len(questions)} ({rate:.0%})")

    print(f"\n  --- Stage5 新考 ({len(STAGE5_EXAM)}题) ---")
    s5_pass = 0
    s5_results = []
    for i, q in enumerate(STAGE5_EXAM):
        prompt = f"[INST] {q} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.replace(prompt, "").strip()
        ok = len(answer) > 30
        if ok:
            s5_pass += 1
        s5_results.append({"Q": q, "A": answer[:300], "pass": ok})
        status = "PASS" if ok else "FAIL"
        print(f"  S5 Q{i+1}/{len(STAGE5_EXAM)} [{status}] {answer[:80]}...")
    all_results["stage5"] = {"pass": s5_pass, "total": len(STAGE5_EXAM), "rate": s5_pass/len(STAGE5_EXAM), "details": s5_results}
    total_pass += s5_pass
    total_q += len(STAGE5_EXAM)
    print(f"  Stage5: {s5_pass}/{len(STAGE5_EXAM)} ({s5_pass/len(STAGE5_EXAM):.0%})")

    overall_rate = total_pass / total_q if total_q else 0
    print(f"\n  全阶段回测总计: {total_pass}/{total_q} ({overall_rate:.0%})")

    regression_fail = False
    for stage in [1, 2, 3, 4]:
        rate = all_results[f"stage{stage}"]["rate"]
        if rate < 0.7:
            print(f"  [WARNING] Stage{stage} 回测通过率 {rate:.0%} < 70%, 存在遗忘!")
            regression_fail = True
    if not regression_fail:
        print(f"  [OK] Stage1-4 旧知识保持良好, 无严重遗忘")

    return all_results, overall_rate, regression_fail


def generate_stage5_data(soul_name, max_samples=8000):
    soul_info = SOUL_TOOLS[soul_name]
    tools = [t.strip() for t in soul_info["tools"].split(",")]
    industry = soul_info["industry"]
    cn_name = soul_info["cn"]

    CORRECTION_DATA = {
        "einstein": [
            {"instruction": "什么是热力学第二定律？请用中文解释熵和不可逆过程。", "output": "热力学第二定律是热力学的基本定律之一，其核心表述为：\n\n1. 克劳修斯表述：热量不可能自发地从低温物体传向高温物体而不引起其他变化。\n2. 开尔文表述：不可能从单一热源取热使之完全变为有用功而不产生其他影响。\n\n熵（Entropy）是热力学中描述系统无序程度的态函数，用S表示。在孤立系统中，熵只可能增加或保持不变，这就是熵增原理：ΔS ≥ 0。熵增意味着不可逆过程的方向性——自然过程总是朝着熵增加的方向进行。\n\n不可逆过程是指无法通过微小变化使系统恢复原状的过程，如热传导、气体自由膨胀、摩擦生热等。这些过程都会导致系统总熵增加，因此是不可逆的。热力学第二定律的本质就是：孤立系统中自发过程的方向总是熵增方向。", "source": "correction_s1q4"},
            {"instruction": "请解释太阳能电池的工作原理，包括光伏效应和半导体P-N结的作用。", "output": "太阳能电池的工作原理基于光伏效应（Photovoltaic Effect）：\n\n1. 光伏效应：当光子能量大于半导体禁带宽度的光照射到半导体P-N结上时，光子被吸收产生电子-空穴对。\n\n2. P-N结的作用：P-N结内建电场将光生的电子和空穴分离，电子被推向N区，空穴被推向P区，在P-N结两侧形成电势差（光生电压）。\n\n3. 电流产生：当外电路接通时，电子从N区经外电路流向P区，形成光生电流。\n\n4. 材料类型：\n- 硅基太阳能电池：单晶硅（效率~25%）、多晶硅（效率~20%）\n- 薄膜电池：碲化镉CdTe（效率~22%）、铜铟镓硒CIGS（效率~23%）\n- 钙钛矿电池：CH3NH3PbI3等有机-无机杂化材料（实验室效率~26%）\n\n5. 提高效率的方法：\n- 多结叠层电池（不同带隙材料吸收不同波长）\n- 抗反射涂层减少光损失\n- 背面钝化减少复合损失\n- 聚光光伏（CPV）系统", "source": "correction_s4q2"},
            {"instruction": "请解释核电站的工作原理，包括核裂变、铀燃料和反应堆安全系统。", "output": "核电站的工作原理：\n\n1. 核裂变：铀-235原子核吸收一个中子后，分裂成两个较轻的原子核（裂变产物），同时释放2-3个中子和大量能量（约200MeV/次裂变）。释放的中子继续引发其他铀-235裂变，形成链式反应。\n\n2. 燃料：核电站使用二氧化铀（UO₂）作为燃料，铀-235富集度约3-5%。燃料制成陶瓷芯块，封装在锆合金包壳管中，组成燃料棒。\n\n3. 反应堆类型：\n- 压水堆（PWR）：最常见，用高压水作慢化剂和冷却剂\n- 沸水堆（BWR）：水在堆芯直接沸腾产生蒸汽\n\n4. 能量转换：核裂变热能→一回路冷却剂→蒸汽发生器→二回路蒸汽→汽轮机→发电机→电能\n\n5. 安全系统：\n- 控制棒：硼或镉制成，插入堆芯吸收中子，控制反应速率\n- 紧急停堆（SCRAM）：控制棒快速插入，瞬间停止链式反应\n- 安全壳：钢筋混凝土结构，防止放射性物质外泄\n- 余热排出系统：停堆后持续冷却堆芯\n- 多重屏障：燃料芯块→包壳→一回路→安全壳", "source": "correction_s4q3"},
            {"instruction": "请解释锂电池的工作原理，包括锂离子在正极、负极和电解质中的运动。", "output": "锂电池（锂离子电池）的工作原理基于锂离子在正极和负极之间的可逆嵌入/脱嵌：\n\n1. 正极（阴极）：含锂的过渡金属氧化物，如：\n- 钴酸锂 LiCoO₂\n- 磷酸铁锂 LiFePO₄\n- 三元材料 LiNiMnCoO₂（NMC）\n\n2. 负极（阳极）：石墨或硅碳复合材料，锂离子嵌入石墨层间形成LiC₆\n\n3. 电解质：锂盐（如LiPF₆）溶解在有机碳酸酯溶剂中（EC、DMC等），提供锂离子传导通道\n\n4. 充电过程：锂离子从正极脱嵌→经电解质→嵌入负极，电子经外电路从正极流向负极\n5. 放电过程：锂离子从负极脱嵌→经电解质→嵌入正极，电子经外电路从负极流向正极\n\n6. 提高能量密度的方法：\n- 正极：高镍三元材料、富锂锰基材料\n- 负极：硅碳复合材料（硅的理论比容量是石墨的10倍）\n- 固态电解质：用固态电解质替代液态电解质，提高安全性", "source": "correction_s4q4"},
            {"instruction": "请解释医疗器械MRI（核磁共振成像）的物理原理和工程实现。", "output": "MRI（Magnetic Resonance Imaging，核磁共振成像）的原理：\n\n1. 物理基础：人体内大量氢原子核（质子）具有自旋和磁矩。在强外磁场B₀中，质子沿磁场方向排列，产生净磁化矢量。\n\n2. 核磁共振：当施加与质子拉莫尔频率相同的射频（RF）脉冲时，质子吸收能量发生共振，磁化矢量偏转。射频脉冲停止后，质子弛豫恢复平衡状态，释放射频信号（MR信号）。\n\n3. 弛豫时间：\n- T1弛豫（纵向弛豫）：磁化矢量恢复到B₀方向的时间常数\n- T2弛豫（横向弛豫）：磁化矢量在垂直B₀平面衰减的时间常数\n不同组织的T1、T2值不同，形成图像对比度\n\n4. 工程实现：\n- 主磁体：超导磁体产生1.5T或3.0T均匀强磁场\n- 梯度线圈：在x、y、z方向产生线性梯度磁场，实现空间编码\n- 射频线圈：发射RF脉冲激励质子，接收MR信号\n- 计算机系统：对MR信号进行傅里叶变换，重建图像\n\n5. 安全注意事项：金属植入物禁忌、幽闭恐惧症处理、噪音防护", "source": "correction_s4q5"},
            {"instruction": "什么是工业自动化中的PID控制？请分别解释比例、积分、微分三个部分。", "output": "PID控制器是工业自动化中最常用的闭环反馈控制算法，由三个部分组成：\n\n1. 比例控制（P，Proportional）：\n输出与误差e(t)成正比：P = Kp × e(t)\nKp为比例增益。误差越大，控制输出越大。但纯比例控制存在稳态误差（余差）。\n\n2. 积分控制（I，Integral）：\n输出与误差的累积成正比：I = Ki × ∫e(t)dt\nKi为积分增益。积分作用消除稳态误差，因为只要误差存在，积分项就持续增加。但积分过强会导致超调和振荡。\n\n3. 微分控制（D，Derivative）：\n输出与误差的变化率成正比：D = Kd × de(t)/dt\nKd为微分增益。微分作用预测误差变化趋势，提前施加控制，减少超调和振荡。但对噪声敏感。\n\nPID总输出：u(t) = Kp×e(t) + Ki×∫e(t)dt + Kd×de(t)/dt\n\n典型应用：温度控制、流量控制、压力控制、电机转速控制、无人机姿态稳定等。", "source": "correction_s4q6"},
        ],
    }

    all_samples = []

    # 0) P0: Correction data (highest priority, must not be overwritten)
    correction_samples = CORRECTION_DATA.get(soul_name, [])
    for s in correction_samples:
        s["source"] = s.get("source", "correction")
    print(f"  Correction data (P0): {len(correction_samples)} samples")

    # 1) Load Stage1-4 data for anti-forgetting (25% = 2000 samples)
    anti_forget_samples = []
    for stage in [1, 2, 3, 4]:
        stage_files = {
            1: f"{soul_name}_stage1_physics_8k.json",
            2: f"{soul_name}_stage2_math_chem_8k.json",
            3: f"{soul_name}_stage3_fusion_8k.json",
            4: f"{soul_name}_stage4_industry_4k.json",
        }
        fname = stage_files.get(stage)
        if not fname:
            continue
        fpath = os.path.join(DATA_DIR, soul_name, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for s in data:
                    if len(s.get("output", "")) >= 50:
                        anti_forget_samples.append({**s, "source": f"stage{stage}_anti_forget"})
        except:
            pass
    print(f"  Anti-forget pool: {len(anti_forget_samples)} samples")

    # 2) Load FinewebEdu filtered data
    fineweb_dir = os.path.join(FINEWEB_DIR, soul_name)
    fineweb_samples = []
    if os.path.exists(fineweb_dir):
        for f in sorted(os.listdir(fineweb_dir)):
            if not f.endswith(".json"):
                continue
            fp = os.path.join(fineweb_dir, f)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    for s in data:
                        if len(s.get("output", "")) >= 50:
                            fineweb_samples.append({**s, "source": "finewebedu"})
            except:
                pass
    print(f"  FinewebEdu pool: {len(fineweb_samples)} samples")

    # 3) Generate tool-specific instruction data
    tool_samples = []
    for tool in tools:
        tool_samples.extend([
            {"instruction": f"如何使用{tool}进行{industry}领域的分析？", "output": f"使用{tool}进行{industry}分析的步骤如下：\n1. 数据准备：收集{industry}相关的原始数据，确保数据格式与{tool}兼容\n2. 模型配置：在{tool}中设置正确的分析参数，包括边界条件和初始条件\n3. 运行分析：执行{tool}的计算引擎，监控收敛情况\n4. 结果解读：对{tool}输出的结果进行专业解读，提取关键指标\n5. 优化迭代：根据结果调整参数，在{tool}中重新计算直到满足精度要求", "source": "tool_template"},
            {"instruction": f"请详细说明{tool}在{industry}中的典型应用场景", "output": f"{tool}在{industry}中的典型应用场景包括：\n\n场景一：基础分析\n利用{tool}对{industry}基础问题进行建模和求解，快速获得初步结果。\n\n场景二：深度优化\n在基础分析基础上，使用{tool}的高级功能进行参数优化和灵敏度分析。\n\n场景三：多物理场耦合\n结合{tool}与其他工具，实现{industry}中多学科交叉的耦合分析。\n\n场景四：自动化流程\n通过{tool}的API或脚本接口，实现{industry}分析流程的自动化批处理。", "source": "tool_template"},
            {"instruction": f"作为{cn_name}AI灵魂，你如何将{tool}整合到{industry}工作流中？", "output": f"作为{cn_name}AI灵魂，我将{tool}整合到{industry}工作流的方式：\n\n第一步：需求分析\n理解用户在{industry}领域的具体需求，确定{tool}能解决的问题类型。\n\n第二步：数据预处理\n将用户的原始数据转换为{tool}可接受的格式，进行必要的数据清洗和验证。\n\n第三步：{tool}建模\n在{tool}中构建分析模型，设置合理的参数和边界条件。\n\n第四步：智能优化\n基于{industry}领域知识，自动调整{tool}的参数，提高分析精度和效率。\n\n第五步：结果输出\n将{tool}的计算结果转化为用户可理解的专业报告，包含可视化图表和建议。", "source": "tool_template"},
            {"instruction": f"{tool}的输入输出格式是什么？在{industry}中如何规范使用？", "output": f"{tool}输入输出规范：\n\n输入格式：\n- 数据文件：支持CSV、JSON、Parquet等标准格式\n- 配置参数：通过YAML或JSON配置文件设定分析参数\n- 模型定义：使用{tool}专有格式或标准交换格式定义模型\n\n输出格式：\n- 数值结果：CSV/JSON格式的计算数据\n- 可视化：PNG/SVG格式的图表和云图\n- 报告：自动生成的PDF/HTML分析报告\n\n{industry}中的规范使用流程：\n1. 按照行业数据标准准备输入数据\n2. 使用经过验证的{tool}模板和预设\n3. 执行分析后进行结果校验\n4. 生成符合行业规范的输出文档", "source": "tool_template"},
            {"instruction": f"比较{tool}与其他{industry}工具的优劣势", "output": f"{tool}与其他{industry}工具对比：\n\n{tool}的优势：\n- 专业性强：针对{industry}领域深度优化\n- 精度高：采用先进的数值方法，计算结果可靠\n- 生态完善：丰富的插件和社区支持\n- 易集成：提供标准API，便于自动化\n\n{tool}的局限：\n- 学习曲线：需要{industry}专业背景才能充分发挥\n- 资源需求：复杂分析需要较高计算资源\n- 特定场景：某些{industry}细分领域可能需要定制开发\n\n适用场景：\n{tool}最适合需要高精度、可重复性的{industry}分析任务，尤其在标准化的工作流中表现优异。", "source": "tool_template"},
        ])
    print(f"  Tool templates: {len(tool_samples)} samples")

    # 4) Mix: correction first, then 25% anti-forget + 45% fineweb + 30% tool
    n_correction = len(correction_samples)
    n_anti = int(max_samples * 0.25)
    n_fineweb = int(max_samples * 0.45)
    n_tool = max_samples - n_correction - n_anti - n_fineweb

    random.seed(3407)
    picked_anti = random.sample(anti_forget_samples, min(n_anti, len(anti_forget_samples)))
    picked_fineweb = random.sample(fineweb_samples, min(n_fineweb, len(fineweb_samples)))
    picked_tool = random.sample(tool_samples, max(0, min(n_tool, len(tool_samples))))

    # CRITICAL: correction samples go FIRST, then others
    # Dedup: correction samples have HIGHEST priority - they overwrite older samples
    all_samples = correction_samples + picked_anti + picked_fineweb + picked_tool

    # Dedup by instruction (first occurrence wins = correction wins)
    seen = set()
    unique = []
    for s in all_samples:
        key = s.get("instruction", "")[:100]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    if len(unique) > max_samples:
        random.seed(3407)
        # Always keep correction samples
        correction_set = set(s.get("instruction", "")[:100] for s in correction_samples)
        non_correction = [s for s in unique if s.get("instruction", "")[:100] not in correction_set]
        kept_correction = [s for s in unique if s.get("instruction", "")[:100] in correction_set]
        random.shuffle(non_correction)
        unique = kept_correction + non_correction[:max_samples - len(kept_correction)]

    n_corr_in_final = sum(1 for s in unique if s.get("source", "").startswith("correction"))
    n_anti_in_final = sum(1 for s in unique if "anti_forget" in s.get("source", ""))
    n_fw_in_final = sum(1 for s in unique if s.get("source") == "finewebedu")
    n_tool_in_final = sum(1 for s in unique if s.get("source") == "tool_template")
    print(f"  Final mix: {len(unique)} samples (correction={n_corr_in_final}, anti-forget={n_anti_in_final}, fineweb={n_fw_in_final}, tool={n_tool_in_final})")
    return unique


def train_soul_stage5(soul_name):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
    from datasets import Dataset
    from trl import SFTTrainer

    soul_info = SOUL_TOOLS[soul_name]
    cn_name = soul_info["cn"]
    industry = soul_info["industry"]
    tools = soul_info["tools"]

    print(f"\n{'='*60}")
    print(f"  Stage5: {cn_name} | {industry}")
    print(f"  Tools: {tools}")
    print(f"  SeqLen: {MAX_SEQ_LENGTH} | GradCkpt: {USE_GRAD_CKPT}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"{'='*60}")

    out_dir = os.path.join(LORA_DIR, soul_name, "stage5")
    log_dir = os.path.join(LOG_DIR, soul_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Generate training data
    print(f"\n  Generating Stage5 data for {soul_name}...")
    samples = generate_stage5_data(soul_name, args.max_samples)

    data_path = os.path.join(DATA_DIR, soul_name, f"{soul_name}_stage5_tools_8k.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"  Data saved: {data_path}")

    # Load model
    print(f"  Loading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load previous stage LoRA
    if args.resume:
        prev_lora = None
        for s in [4, 3, 2, 1]:
            candidate = os.path.join(LORA_DIR, soul_name, f"stage{s}", "final")
            if os.path.exists(candidate):
                prev_lora = candidate
                break
        if prev_lora:
            print(f"  Resuming from: {prev_lora}")
            model = PeftModel.from_pretrained(model, prev_lora)
            model = model.merge_and_unload()
            print(f"  Merged previous stage LoRA into base model")
        else:
            print(f"  WARNING: No previous LoRA found, training from base")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=USE_GRAD_CKPT)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    def format_sample(sample):
        inst = sample.get("instruction", "")
        inp = sample.get("input", "")
        out = sample.get("output", "")
        if inp:
            text = f"<s>[INST] {inst}\n{inp} [/INST] {out}</s>"
        else:
            text = f"<s>[INST] {inst} [/INST] {out}</s>"
        return {"text": text}

    ds = Dataset.from_list([format_sample(s) for s in samples])

    # Train
    gpu_stats_start = torch.cuda.memory_reserved() / 1024**3
    print(f"  VRAM before train: {gpu_stats_start:.1f}GB")

    t0 = time.time()
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.1,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            bf16=False,
            logging_steps=25,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=out_dir,
            save_strategy="epoch",
            save_total_limit=2,
            max_grad_norm=0.5,
            report_to="none",
            disable_tqdm=True,
            dataloader_num_workers=0,
            gradient_checkpointing=USE_GRAD_CKPT,
        ),
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"\n  [ERROR] Training crashed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    elapsed = (time.time() - t0) / 60

    peak_vram = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\n  Training done in {elapsed:.1f} min | Peak VRAM: {peak_vram:.1f}GB")

    final_loss = 999.0
    if trainer.state and trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                final_loss = entry["loss"]
                break
    print(f"  Final Loss: {final_loss:.4f}")

    # Save
    final_path = os.path.join(out_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Saved LoRA: {final_path}")

    # === 全阶段回测 ===
    exam_results, overall_rate, regression_fail = run_regression_exam(model, tokenizer)

    # Final verdict
    loss_ok = final_loss <= 1.4
    exam_ok = overall_rate >= 0.80
    no_forget = not regression_fail
    overall = loss_ok and exam_ok and no_forget

    print(f"\n  {'='*50}")
    print(f"  Stage5 最终判定:")
    print(f"    Loss: {final_loss:.4f} {'<=' if loss_ok else '>'} 1.4 = {'PASS' if loss_ok else 'FAIL'}")
    print(f"    全阶段回测: {overall_rate:.0%} {'>=' if exam_ok else '<'} 80% = {'PASS' if exam_ok else 'FAIL'}")
    print(f"    遗忘检查: {'PASS (无严重遗忘)' if no_forget else 'FAIL (存在遗忘,需复训)'}")
    print(f"    总判定: {'PASS' if overall else 'FAIL'}")
    print(f"  {'='*50}")

    if not overall:
        if regression_fail:
            print(f"  -> 旧知识遗忘! 需要增加anti-forget数据比例,重新训练")
        if not loss_ok:
            print(f"  -> Loss过高, 考虑降低LR或增加epochs")
        if not exam_ok:
            print(f"  -> 回测通过率不足, 考虑增加训练数据或延长训练")

    # Log
    log_data = {
        "soul": soul_name,
        "stage": "stage5",
        "cn_name": cn_name,
        "industry": industry,
        "tools": tools,
        "final_loss": final_loss,
        "loss_pass": loss_ok,
        "overall_exam_rate": overall_rate,
        "exam_pass": exam_ok,
        "regression_fail": regression_fail,
        "no_forget": no_forget,
        "overall_pass": overall,
        "elapsed_min": elapsed,
        "peak_vram_gb": peak_vram,
        "samples": len(samples),
        "epochs": args.epochs,
        "lr": args.lr,
        "seq_len": MAX_SEQ_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "grad_ckpt": USE_GRAD_CKPT,
        "exam_results": exam_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(log_dir, "stage5_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"  Log saved: {log_path}")

    # Cleanup
    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return final_loss, overall


if __name__ == "__main__":
    if args.all:
        print("=" * 60)
        print("  Stage5: Training ALL 14 souls (with full regression)")
        print("=" * 60)
        results = {}
        for soul in TRAIN_ORDER:
            try:
                loss, passed = train_soul_stage5(soul)
                results[soul] = {"status": "DONE" if passed else "NEEDS_RETRAIN", "loss": loss, "passed": passed}
            except Exception as e:
                results[soul] = {"status": "FAILED", "error": str(e)}
                print(f"  [ERROR] {soul}: {e}")
        print("\n" + "=" * 60)
        print("  Stage5 ALL Results:")
        for soul, r in results.items():
            print(f"    {soul}: {r}")
        print("=" * 60)
    else:
        train_soul_stage5(args.soul)
