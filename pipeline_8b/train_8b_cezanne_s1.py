#!/usr/bin/env python3
"""
Cezanne_PRO 8B Stage1 - Math Foundation Training
Soul: Cezanne_PRO (8B version of Cezanne)
Base: D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024 (post brain surgery)
Data: cezanne_stage1_math_8k.json (same as 7B Cezanne)
"""
import os, sys, json, gc, time, random, faulthandler
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

os.makedirs(r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b", exist_ok=True)
FAULT_LOG = open(r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b\fault_s1.log", "w")
faulthandler.enable(file=FAULT_LOG, all_threads=True)

BASE_MODEL = r"D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024"
DATA_DIR = r"D:\VORTEX_FLAME\soul_training_data\cezanne"
LORA_DIR = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne_8b"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_SEQ_LENGTH = 128
EPOCHS = 3
LR = 3e-4
USE_GRAD_CKPT = True
MAX_SAMPLES = 8000

EXAM_Q = [
    "What is the derivative of x^3 + 2x^2 - 5x + 1? Show the step by step process.",
    "Explain the concept of a matrix eigenvalue and how to compute it.",
    "What is the difference between a convergent series and a divergent series? Give examples.",
    "Explain the fundamental theorem of calculus and its significance.",
    "What is a probability distribution? Explain normal distribution.",
    "How do you solve a system of linear equations using matrix methods?",
    "Explain the concept of mathematical induction with an example.",
    "What is the difference between permutation and combination?",
    "Explain what a limit is in calculus and how to compute it.",
    "What is a vector space? Give the definition and axioms.",
]

KEYWORDS = [
    ["derivative", "3x^2", "4x", "differenti"],
    ["eigenvalue", "eigenvector", "characteristic"],
    ["convergent", "divergent", "series", "limit"],
    ["fundamental theorem", "integral", "antiderivative"],
    ["probability", "distribution", "normal", "Gaussian"],
    ["linear equations", "matrix", "augmented", "Gaussian elimination"],
    ["induction", "base case", "inductive step"],
    ["permutation", "combination", "order", "arrangement"],
    ["limit", "approach", "epsilon", "infinity"],
    ["vector space", "axiom", "scalar", "addition"],
]


def load_cezanne_data(max_samples):
    all_samples = []
    s1_path = os.path.join(DATA_DIR, "cezanne_stage1_math_8k.json")
    with open(s1_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and len(item.get("output", "")) >= 50:
                all_samples.append(item)
    print(f"  Loaded stage1 data: {len(all_samples)} items", flush=True)

    random.seed(3407)
    random.shuffle(all_samples)

    seen = set()
    unique = []
    for s in all_samples:
        key = s.get("instruction", "")[:100]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique[:max_samples]


def run_exam(model, tokenizer):
    print(f"\n  {'='*50}")
    print(f"  Cezanne 8B Stage1 Exam")
    print(f"  {'='*50}")
    passed = 0
    for i, q in enumerate(EXAM_Q):
        prompt = f"[INST] {q} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        kw_list = KEYWORDS[i] if i < len(KEYWORDS) else []
        matched = [kw for kw in kw_list if kw.lower() in answer.lower()]
        ok = len(matched) >= 1 or len(answer) > 100
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        safe_answer = answer[:80].encode('ascii', errors='replace').decode()
        print(f"  Q{i+1}/10 [{status}] {safe_answer}...")
    rate = passed / len(EXAM_Q)
    print(f"\n  Result: {passed}/10 ({rate:.0%})")
    return rate


def main():
    from transformers import Mistral3ForConditionalGeneration, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import Dataset
    from trl import SFTTrainer

    os.makedirs(LORA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    out_dir = os.path.join(LORA_DIR, "stage1")

    print(f"{'='*60}")
    print(f"  Cezanne 8B Stage1: Math Foundation")
    print(f"  Base: {BASE_MODEL}")
    print(f"  Samples: {MAX_SAMPLES}, Epochs: {EPOCHS}, LR: {LR}")
    print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    print(f"{'='*60}")

    print("  Loading data...")
    samples = load_cezanne_data(MAX_SAMPLES)
    print(f"  Loaded: {len(samples)} unique samples")

    print("  Loading model...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = Mistral3ForConditionalGeneration.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.enable_input_require_grads()

    lang_target_modules = []
    for name, _ in model.named_modules():
        if "language_model" in name and any(name.endswith(t) for t in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]):
            lang_target_modules.append(name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=lang_target_modules,
        bias="none",
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def format_sample(sample):
        inst = sample.get("instruction", "")
        inp = sample.get("input", "")
        out = sample.get("output", "")
        text = f"<s>[INST] {inst}\n{inp} [/INST] {out}</s>" if inp else f"<s>[INST] {inst} [/INST] {out}</s>"
        return {"text": text}

    ds = Dataset.from_list([format_sample(s) for s in samples])

    t0 = time.time()
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        max_seq_length=MAX_SEQ_LENGTH, dataset_text_field="text",
        dataset_num_proc=1, packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.05, num_train_epochs=EPOCHS,
            learning_rate=LR, fp16=True,
            logging_steps=25, optim="adamw_torch",
            weight_decay=0.01, lr_scheduler_type="cosine",
            seed=3407, output_dir=out_dir,
            save_strategy="epoch", save_total_limit=2,
            max_grad_norm=0.5, report_to="none", disable_tqdm=True,
            dataloader_num_workers=0, gradient_checkpointing=USE_GRAD_CKPT,
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

    final_path = os.path.join(out_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Saved LoRA: {final_path}")

    rate = run_exam(model, tokenizer)

    loss_ok = final_loss <= 2.5
    exam_ok = rate >= 0.6
    overall = loss_ok and exam_ok
    print(f"\n  Loss={final_loss:.4f}({'OK' if loss_ok else 'HIGH'}) Exam={rate:.0%}({'OK' if exam_ok else 'LOW'}) Overall={'PASS' if overall else 'FAIL'}")

    log_data = {
        "soul": "Cezanne_PRO", "stage": "stage1", "cn_name": "Cezanne_PRO",
        "final_loss": final_loss, "exam_rate": rate, "overall_pass": overall,
        "elapsed_min": elapsed, "peak_vram_gb": peak_vram,
        "samples": len(samples), "epochs": EPOCHS, "lr": LR, "seq_len": MAX_SEQ_LENGTH,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "base_model": BASE_MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, "stage1_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
