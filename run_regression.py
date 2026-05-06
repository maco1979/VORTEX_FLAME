#!/usr/bin/env python3
"""
Stage5 回归检测脚本 - 只跑考试，不训练
加载Stage5 final LoRA，跑Stage1-5全阶段回测
"""
import os, json, gc, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "D:/models/Mistral-7B-Instruct-v0.1"
LORA_PATH = "D:/VORTEX_FLAME/soul_lora_v2/einstein/stage5/final"
LOG_DIR = "D:/VORTEX_FLAME/hermes_logs/einstein"

EXAM = {
    1: [
        "请用中文解释牛顿第二定律F=ma的物理含义。",
        "什么是能量守恒定律？请用中文举例说明。",
        "请解释光的折射现象及其公式。",
        "什么是热力学第二定律？请解释熵的概念。",
        "请解释电磁感应定律及其应用。",
        "什么是波粒二象性？请用中文说明。",
        "请解释狭义相对论的基本假设。",
        "什么是量子力学的测不准原理？",
        "请解释万有引力定律。",
        "什么是热传导？请说明三种传热方式。",
    ],
    2: [
        "请解释微积分中导数的物理意义。",
        "什么是矩阵特征值？它在物理中有什么应用？",
        "请解释化学反应平衡常数。",
        "什么是偏微分方程？请举例说明。",
        "请解释概率论中的大数定律。",
        "什么是线性代数中的向量空间？",
        "请解释化学键的类型及其特性。",
        "什么是傅里叶变换？它的物理意义是什么？",
        "请解释统计学中的正态分布。",
        "什么是群论？它在物理学中有什么应用？",
    ],
    3: [
        "如何将数学建模应用于物理问题？",
        "请解释物理化学中的相变理论。",
        "如何用数学方法解决化学反应动力学问题？",
        "请解释量子化学的基本概念。",
        "如何将统计学应用于实验数据分析？",
        "请解释计算物理中的数值模拟方法。",
        "如何将线性代数应用于量子力学？",
        "请解释数学物理方程的分类。",
        "如何用概率论分析物理实验的不确定性？",
        "请解释跨学科研究中数学、物理、化学的联系。",
    ],
    4: [
        "Ansys在半导体热应力分析中如何应用？",
        "COMSOL Multiphysics在新能源电池仿真中的工作流程是什么？",
        "电力系统仿真软件如何进行潮流计算？",
        "工业AI在智能制造中的应用场景有哪些？",
        "如何使用有限元方法分析工程结构强度？",
        "半导体器件仿真中如何处理多物理场耦合？",
        "新能源发电系统中如何优化电力调度？",
        "工业AI如何实现预测性维护？",
        "如何将物理仿真与机器学习结合？",
        "请解释数字孪生技术在工业中的应用。",
    ],
    5: [
        "如何使用Ansys进行新能源电池热管理仿真？",
        "COMSOL在半导体器件仿真中的优势是什么？",
        "电力仿真软件如何实现短路电流计算？",
        "工业AI如何优化供应链管理？",
        "如何将Ansys仿真结果导入机器学习模型？",
        "COMSOL Multiphysics如何模拟电化学过程？",
        "电力系统如何使用AI进行负荷预测？",
        "工业AI在质量控制中的实施方案是什么？",
    ],
}

KEYWORDS = {
    1: [["牛顿第二定律", "F=ma", "加速度", "力"], ["能量守恒", "转化", "不灭"], ["折射", "斯涅尔"], ["热力学第二定律", "熵"], ["电磁感应", "法拉第"], ["波粒二象性"], ["相对论", "光速"], ["测不准", "不确定性"], ["万有引力"], ["传导", "对流", "辐射"]],
    2: [["导数", "变化率"], ["特征值"], ["平衡常数"], ["偏微分"], ["大数定律"], ["向量空间"], ["化学键"], ["傅里叶"], ["正态分布"], ["群论"]],
    3: [["数学建模"], ["相变"], ["动力学"], ["量子化学"], ["统计", "分析"], ["数值模拟"], ["线性代数", "量子"], ["数学物理"], ["概率", "不确定"], ["跨学科"]],
    4: [["Ansys", "热应力"], ["COMSOL", "电池"], ["潮流计算"], ["工业AI", "制造"], ["有限元"], ["多物理场"], ["电力调度"], ["预测性维护"], ["仿真", "机器学习"], ["数字孪生"]],
    5: [["Ansys", "热管理"], ["COMSOL", "半导体"], ["短路电流"], ["AI", "供应链"], ["Ansys", "机器学习"], ["COMSOL", "电化学"], ["AI", "负荷预测"], ["AI", "质量控制"]],
}

def check_answer(answer, stage, q_idx):
    if len(answer) < 30:
        return False, []
    kw_list = KEYWORDS.get(stage, [])
    if q_idx >= len(kw_list):
        return len(answer) > 30, []
    matched = [kw for kw in kw_list[q_idx] if kw in answer]
    return len(matched) >= 1, matched

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    print("Loading Stage5 final model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()

    all_results = {}
    total_pass = 0
    total_q = 0

    for stage in [1, 2, 3, 4, 5]:
        questions = EXAM.get(stage, [])
        stage_pass = 0
        stage_details = []
        print(f"\n{'='*50}")
        print(f"  Stage{stage} 回测 ({len(questions)}题)")
        print(f"{'='*50}")

        for i, q in enumerate(questions):
            prompt = f"[INST] {q} [/INST] "
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            ok, matched = check_answer(answer, stage, i)
            if ok:
                stage_pass += 1
            status = "PASS" if ok else "FAIL"
            print(f"  S{stage} Q{i+1}/{len(questions)} [{status}] {answer[:80]}...")
            stage_details.append({"question": q, "answer": answer[:500], "pass": ok, "matched_keywords": matched})

        rate = stage_pass / len(questions) if questions else 0
        all_results[f"stage{stage}"] = {"pass": stage_pass, "total": len(questions), "rate": rate, "details": stage_details}
        total_pass += stage_pass
        total_q += len(questions)
        print(f"  Stage{stage}: {stage_pass}/{len(questions)} ({rate:.0%})")

    overall_rate = total_pass / total_q if total_q else 0
    regression_fail = any(r["rate"] < 0.7 for r in all_results.values() if r["total"] > 0)

    print(f"\n{'='*60}")
    print(f"  全阶段回测结果: {total_pass}/{total_q} ({overall_rate:.0%})")
    print(f"  回归检测: {'FAIL - 存在严重遗忘' if regression_fail else 'PASS - 无严重遗忘'}")
    print(f"  总体判定: {'PASS' if overall_rate >= 0.8 and not regression_fail else 'FAIL'}")
    print(f"{'='*60}")

    log_data = {
        "type": "stage5_full_regression",
        "model": "einstein_stage5_final",
        "overall_rate": overall_rate,
        "total_pass": total_pass,
        "total_q": total_q,
        "regression_fail": regression_fail,
        "overall_pass": overall_rate >= 0.8 and not regression_fail,
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, "stage5_regression_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {log_path}")

if __name__ == "__main__":
    main()
