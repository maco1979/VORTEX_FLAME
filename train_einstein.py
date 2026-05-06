#!/usr/bin/env python3
"""
Einstein 4阶段训练脚本 (Windows, NO unsloth)
底座: Mistral-7B-Instruct-v0.1 (D:\models\Mistral-7B-Instruct-v0.1)
数据: einstein_stage{N}_*_8k.json (每阶段8000条精准筛选)
方法: QLoRA 4-bit + LoRA (标准HuggingFace, 无unsloth依赖)

4阶段:
  Stage1 领域打底: 纯物理, Loss<=2.5, lr=2e-4, 3epochs
  Stage2 知识扩展: 数学+化学, Loss<=2.0, lr=2.5e-4, 3epochs
  Stage3 融合训练: 数理融合, Loss<=1.6, lr=2e-4, 2epochs
  Stage4 行业落地: 半导体/新能源/医疗, Loss<=1.4, lr=1.5e-4, 2epochs

用法:
  python train_einstein.py              -- 训练Stage1
  python train_einstein.py --stage 2    -- 训练Stage2
  python train_einstein.py --stage 3    -- 训练Stage3
  python train_einstein.py --stage 4    -- 训练Stage4
  python train_einstein.py --stage 1 --resume  -- 从上阶段LoRA继续
"""

import os, json, gc, time, random, sys, argparse, faulthandler

faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

FAULT_LOG = open(r"D:\VORTEX_FLAME\train_fault.log", "w")
faulthandler.enable(file=FAULT_LOG, all_threads=True)

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, default=1, choices=[1,2,3,4])
parser.add_argument("--resume", action="store_true", help="Resume from previous stage LoRA")
parser.add_argument("--max_samples", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--seq_len", type=int, default=None)
args = parser.parse_args()

STAGE = args.stage

STAGE_CONFIGS = {
    1: {"name": "stage1_physics",     "cn": "领域打底-纯物理",       "data": "einstein_stage1_physics_8k.json",     "lr": 2e-4,  "epochs": 3, "loss_threshold": 2.5, "exam_rate": 0.60},
    2: {"name": "stage2_math_chem",   "cn": "知识扩展-数学化学",     "data": "einstein_stage2_math_chem_8k.json",   "lr": 2.5e-4,"epochs": 3, "loss_threshold": 2.0, "exam_rate": 0.70},
    3: {"name": "stage3_fusion",      "cn": "融合训练-数理融合",     "data": "einstein_stage3_fusion_8k.json",      "lr": 2e-4,  "epochs": 2, "loss_threshold": 1.6, "exam_rate": 0.80},
    4: {"name": "stage4_industry",    "cn": "行业落地-半导体新能源",  "data": "einstein_stage4_industry_4k.json",    "lr": 1.5e-4,"epochs": 2, "loss_threshold": 1.4, "exam_rate": 0.80, "seq_len": 128, "grad_ckpt": True},
}

cfg = STAGE_CONFIGS[STAGE]

BASE_MODEL = r"D:\models\Mistral-7B-Instruct-v0.1"
DATA_DIR = r"D:\VORTEX_FLAME\soul_training_data\einstein"
LORA_DIR = r"D:\VORTEX_FLAME\soul_lora_v2\einstein"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs\einstein"

MAX_SAMPLES = args.max_samples or 8000
STAGE_EPOCHS = args.epochs or cfg["epochs"]
STAGE_LR = args.lr or cfg["lr"]
LOSS_THRESHOLD = cfg["loss_threshold"]
EXAM_RATE_THRESHOLD = cfg["exam_rate"]
MAX_SEQ_LENGTH = args.seq_len or cfg.get("seq_len", 256)
USE_GRAD_CKPT = cfg.get("grad_ckpt", False)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 1
GRAD_ACCUM = 8

OUT_DIR = os.path.join(LORA_DIR, f"stage{STAGE}")
DATA_FILE = os.path.join(DATA_DIR, cfg["data"])

LOG_FILE = open(os.path.join(r"D:\VORTEX_FLAME", f"train_stage{STAGE}.log"), "w", encoding="utf-8", buffering=1)

class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except:
                pass

sys.stdout = TeeWriter(sys.__stdout__, LOG_FILE)
sys.stderr = TeeWriter(sys.__stderr__, LOG_FILE)

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


def load_stage_data():
    print(f"  Loading: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print(f"  [FATAL] Data file not found: {DATA_FILE}")
        sys.exit(1)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and "data" in raw:
        items = raw["data"]
    else:
        items = [raw]

    print(f"  Raw items: {len(items):,}")

    filtered = []
    for item in items:
        if not isinstance(item, dict):
            continue
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        if not instruction or not output:
            continue
        if len(output) < 20 or len(output) > 3000:
            continue
        blacklist = ["不知道", "不会", "抱歉", "作为AI", "http:", "https:"]
        if any(b in output for b in blacklist):
            continue
        filtered.append(item)

    print(f"  After quality filter: {len(filtered):,}")

    if len(filtered) > MAX_SAMPLES:
        random.seed(42)
        filtered = random.sample(filtered, MAX_SAMPLES)

    print(f"  Final sample: {len(filtered):,}")
    return filtered


def format_sft(entry):
    instruction = entry.get("instruction", "").strip()
    input_text = entry.get("input", "").strip()
    output = entry.get("output", "").strip()
    if input_text:
        return {"text": f"<s>[INST] {instruction}\n{input_text} [/INST] {output}</s>"}
    return {"text": f"<s>[INST] {instruction} [/INST] {output}</s>"}


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


def run_exam(model, tokenizer, stage):
    questions = EXAM_QUESTIONS.get(stage, EXAM_QUESTIONS[1])
    model.eval()
    results = []
    correct = 0

    for i, q in enumerate(questions):
        prompt = f"[INST] {q} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.replace(prompt, "").strip()
        ok = len(answer) > 30
        if ok:
            correct += 1
        results.append({"Q": q, "A": answer[:300], "pass": ok})
        status = "PASS" if ok else "FAIL"
        print(f"  Q{i+1}/{len(questions)} [{status}] {answer[:80]}...")

    rate = correct / len(questions)
    print(f"  Exam: {correct}/{len(questions)} ({rate:.0%}), threshold={EXAM_RATE_THRESHOLD:.0%}")
    return rate, results


def main():
    print("=" * 70)
    print(f"  Einstein Stage{STAGE}: {cfg['cn']}")
    print(f"  Base: {BASE_MODEL}")
    print(f"  Data: {cfg['data']} ({MAX_SAMPLES:,} samples)")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"  LR: {STAGE_LR}, Epochs: {STAGE_EPOCHS}, SeqLen: {MAX_SEQ_LENGTH}, GradCkpt: {USE_GRAD_CKPT}")
    print(f"  Loss threshold: {LOSS_THRESHOLD}, Exam rate: {EXAM_RATE_THRESHOLD:.0%}")
    print(f"  Resume from prev: {args.resume}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  VRAM: Total={total/1024**3:.1f}GB, Free={free/1024**3:.1f}GB")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    data = load_stage_data()
    if not data:
        print("[FATAL] No training data!")
        return

    texts = [format_sft(item) for item in data]
    ds = Dataset.from_list(texts)
    print(f"  Dataset: {len(ds)} samples")

    torch.cuda.empty_cache()
    gc.collect()

    print("\n  Loading model with 4-bit quantization...", flush=True)
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
        torch_dtype=torch.float16,
    )

    if args.resume and STAGE > 1:
        prev_lora = os.path.join(LORA_DIR, f"stage{STAGE-1}", "final")
        if os.path.exists(prev_lora):
            print(f"  Loading previous LoRA: {prev_lora}", flush=True)
            model = PeftModel.from_pretrained(model, prev_lora)
            model = model.merge_and_unload()
            print(f"  Merged previous stage LoRA into base model")
        else:
            print(f"  [WARN] Previous LoRA not found: {prev_lora}, training from base")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("  Preparing model for k-bit training...", flush=True)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=USE_GRAD_CKPT,
    )

    print("  Applying LoRA...", flush=True)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")
    model.print_trainable_parameters()

    print(f"\n  Starting Stage{STAGE} training...", flush=True)
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
            num_train_epochs=STAGE_EPOCHS,
            learning_rate=STAGE_LR,
            fp16=True,
            bf16=False,
            logging_steps=25,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=OUT_DIR,
            save_strategy="epoch",
            save_total_limit=2,
            max_grad_norm=0.5,
            report_to="none",
            disable_tqdm=True,
            dataloader_num_workers=0,
            gradient_checkpointing=USE_GRAD_CKPT,
        ),
    )

    gpu_stats_start = torch.cuda.memory_reserved() / 1024**3
    print(f"  VRAM before train: {gpu_stats_start:.1f}GB")

    t0 = time.time()
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

    final_path = os.path.join(OUT_DIR, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Saved LoRA: {final_path}")

    loss_ok = final_loss <= LOSS_THRESHOLD
    print(f"\n  Loss gate: {final_loss:.4f} {'<=' if loss_ok else '>'} {LOSS_THRESHOLD} = {'PASS' if loss_ok else 'FAIL'}")

    print(f"\n  Running Stage{STAGE} exam...", flush=True)
    exam_rate, exam_results = run_exam(model, tokenizer, STAGE)

    exam_ok = exam_rate >= EXAM_RATE_THRESHOLD
    print(f"  Exam gate: {exam_rate:.0%} {'>=' if exam_ok else '<'} {EXAM_RATE_THRESHOLD:.0%} = {'PASS' if exam_ok else 'FAIL'}")

    overall = loss_ok and exam_ok
    print(f"\n  STAGE{STAGE} RESULT: {'PASS' if overall else 'FAIL'}")
    if not overall:
        if not loss_ok:
            print(f"  -> Loss too high, consider: reduce LR, increase epochs, check data quality")
        if not exam_ok:
            print(f"  -> Exam failed, consider: more relevant data, longer training")

    log_data = {
        "stage": f"stage{STAGE}",
        "soul": "einstein",
        "stage_name": cfg["cn"],
        "final_loss": final_loss,
        "loss_threshold": LOSS_THRESHOLD,
        "loss_pass": loss_ok,
        "exam_rate": exam_rate,
        "exam_threshold": EXAM_RATE_THRESHOLD,
        "exam_pass": exam_ok,
        "overall_pass": overall,
        "trainable_params": trainable,
        "peak_vram_gb": peak_vram,
        "elapsed_min": elapsed,
        "samples": len(ds),
        "epochs": STAGE_EPOCHS,
        "lr": STAGE_LR,
        "lora_r": LORA_R,
        "seq_len": MAX_SEQ_LENGTH,
        "resume": args.resume,
        "exam_results": exam_results,
    }
    log_path = os.path.join(LOG_DIR, f"stage{STAGE}_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"  Log saved: {log_path}")

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    if overall:
        if STAGE < 4:
            print(f"  STAGE{STAGE} PASSED! Ready for Stage{STAGE+1}")
            print(f"  Next: python train_einstein.py --stage {STAGE+1} --resume")
        else:
            print(f"  ALL 4 STAGES PASSED! Einstein training complete!")
            print(f"  LoRA saved: {LORA_DIR}")
    else:
        print(f"  STAGE{STAGE} FAILED - needs retry or parameter adjustment")
        print(f"  Retry: python train_einstein.py --stage {STAGE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
