import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

LOG_FILE = r"D:\VORTEX_FLAME\regression_log.txt"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("START regression test")

import torch
log(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
log("imports done")

BASE_MODEL = r"D:\models\Mistral-7B-Instruct-v0.1"
LORA_PATH = r"D:\VORTEX_FLAME\soul_lora_v2\einstein\stage4\final"

log("loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16,
)
log("base model loaded")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
log("tokenizer loaded")

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()
log("LoRA loaded")

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

CRITICAL_CONCEPTS = {
    1: {
        0: ["牛顿第二定律", "F=ma", "加速度", "力"],
        1: ["能量守恒", "转化", "不灭"],
        2: ["折射", "斯涅尔", "n1sin"],
        3: ["熵", "不可逆", "孤立系统"],
        4: ["电磁感应", "法拉第", "磁通量"],
        5: ["波粒二象性", "德布罗意", "波函数"],
        6: ["时间膨胀", "洛伦兹", "光速"],
        7: ["麦克斯韦", "电场", "磁场"],
        8: ["普朗克", "h=", "量子化"],
        9: ["PN结", "p型", "n型", "耗尽层"],
    },
    2: {
        0: ["导数", "切线", "斜率", "变化率"],
        1: ["特征值", "特征向量"],
        2: ["大数定律", "频率", "概率", "收敛"],
        3: ["化学键", "共价键", "离子键", "电子"],
        4: ["盖斯定律", "焓", "状态函数"],
        5: ["傅里叶变换", "频域", "时域"],
        6: ["烷烃", "甲基", "碳链"],
        7: ["假设检验", "p值", "显著性"],
        8: ["化学平衡", "勒夏特列", "可逆反应"],
        9: ["微分方程", "导数", "初值"],
    },
    3: {
        0: ["统计力学", "微观", "宏观", "配分函数"],
        1: ["密度泛函", "DFT", "电子密度"],
        2: ["傅里叶变换", "频域", "量子"],
        3: ["蒙特卡洛", "随机采样", "概率"],
        4: ["变分法", "极值", "泛函"],
        5: ["分子动力学", "牛顿方程", "势能"],
        6: ["能带", "导体", "半导体", "绝缘体"],
        7: ["有限元", "网格", "离散化"],
        8: ["格林函数", "微分方程", "边界"],
        9: ["相律", "F=C-P+2", "吉布斯"],
    },
    4: {
        0: ["光刻", "掩模", "光刻胶", "曝光"],
        1: ["太阳能电池", "光伏效应", "半导体", "P-N结"],
        2: ["核电站", "核裂变", "铀", "反应堆"],
        3: ["锂电池", "锂离子", "正极", "负极", "电解质"],
        4: ["MRI", "核磁共振", "磁场", "射频"],
        5: ["PID", "比例", "积分", "微分"],
        6: ["变压器", "电磁感应", "初级", "次级"],
        7: ["纳米材料", "1-100nm", "纳米尺度"],
        8: ["风力发电", "风轮", "动能", "发电机"],
        9: ["嵌入式", "微处理器", "实时", "专用"],
    },
}

import json, time

all_results = {}
total_pass_concept = 0
total_q = 0

for stage in [1, 2, 3, 4]:
    questions = EXAM_QUESTIONS[stage]
    concepts = CRITICAL_CONCEPTS[stage]
    stage_pass_concept = 0
    stage_results = []
    log(f"--- Stage{stage} ({len(questions)} questions) ---")

    for i, q in enumerate(questions):
        prompt = f"[INST] {q} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.replace(prompt, "").strip()

        concept_list = concepts.get(i, [])
        pass_concept = any(c in answer for c in concept_list)
        matched = [c for c in concept_list if c in answer]
        missing = [c for c in concept_list if c not in answer]

        if pass_concept:
            stage_pass_concept += 1

        stage_results.append({
            "stage": stage, "q_num": i + 1, "question": q,
            "answer": answer[:300], "pass_concept": pass_concept,
            "matched_concepts": matched, "missing_concepts": missing,
        })
        status = "PASS" if pass_concept else "FAIL"
        log(f"  S{stage} Q{i+1} [{status}] matched={matched} missing={missing} | {answer[:80]}")

    rate_concept = stage_pass_concept / len(questions)
    all_results[f"stage{stage}"] = {
        "pass_concept": stage_pass_concept, "total": len(questions),
        "rate_concept": rate_concept, "details": stage_results,
    }
    total_pass_concept += stage_pass_concept
    total_q += len(questions)
    log(f"  Stage{stage}: concept_pass={stage_pass_concept}/{len(questions)} ({rate_concept:.0%})")

overall_concept = total_pass_concept / total_q
regression_fail = False
for stage in [1, 2, 3, 4]:
    rate = all_results[f"stage{stage}"]["rate_concept"]
    if rate < 0.7:
        regression_fail = True
        log(f"  [WARNING] Stage{stage} concept rate {rate:.0%} < 70%, FORGETTING!")

log(f"OVERALL: {total_pass_concept}/{total_q} ({overall_concept:.0%}), regression_fail={regression_fail}")

log_data = {
    "type": "full_regression_test",
    "model": "einstein_stage4",
    "overall_rate_concept": overall_concept,
    "regression_fail": regression_fail,
    "results": all_results,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}
log_path = r"D:\VORTEX_FLAME\hermes_logs\einstein\regression_test_stage1_4.json"
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(log_data, f, ensure_ascii=False, indent=2)
log(f"Saved: {log_path}")
log("DONE")
