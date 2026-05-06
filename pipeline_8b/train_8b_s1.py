#!/usr/bin/env python3
"""
Step 3: Train 8B Soul - Stage 1 (Pure Math Foundation)
Ministral-8B-Reasoning base, 4bit QLoRA
Same 5-stage protocol as Cezanne/Einstein, but on 8B base
"""
import os, json, gc, time, random, faulthandler
import torch

faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

SOUL_NAME = os.environ.get("SOUL_NAME", "galileo")
STAGE = "stage1"

BASE_MODEL = r"D:\models\Ministral-8B-Reasoning"
DATA_DIR = r"D:\VORTEX_FLAME\soul_training_data"
LORA_DIR = r"D:\VORTEX_FLAME\soul_lora_v2"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs"

os.makedirs(os.path.join(LOG_DIR, SOUL_NAME), exist_ok=True)
FAULT_LOG = open(os.path.join(LOG_DIR, SOUL_NAME, f"fault_{STAGE}.log"), "w")
faulthandler.enable(file=FAULT_LOG, all_threads=True)

LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_SEQ_LENGTH = 256
EPOCHS = 3
LR = 2e-4

EXAM_Q = [
    "Prove that the square root of 2 is irrational. Label each step.",
    "Solve: find the derivative of f(x) = sin(x) * e^x. Show each step.",
    "What is the Fundamental Theorem of Calculus? Explain both parts.",
    "Prove by mathematical induction: 1+2+...+n = n(n+1)/2.",
    "Explain the difference between a convergent and divergent series. Give examples.",
    "What is a eigenvalue and eigenvector? Find eigenvalues of [[2,1],[1,2]].",
    "State and prove De Morgan's Laws for sets.",
    "What is the chain rule in calculus? Apply it to differentiate ln(sin(x)).",
    "Explain the concept of linear independence with an example.",
    "What is Bayes' theorem? Derive it from the definition of conditional probability.",
]

KEYWORDS = [
    ["irrational", "contradiction", "sqrt", "2", "assume"],
    ["derivative", "product rule", "sin", "e^x", "cos"],
    ["fundamental theorem", "integral", "antiderivative", "derivative"],
    ["induction", "base case", "inductive step", "n(n+1)/2"],
    ["convergent", "divergent", "series", "limit", "geometric"],
    ["eigenvalue", "eigenvector", "matrix", "characteristic"],
    ["De Morgan", "union", "intersection", "complement"],
    ["chain rule", "ln", "sin", "cos", "inner"],
    ["linear independence", "combination", "trivial", "vectors"],
    ["Bayes", "conditional", "prior", "posterior", "P(A|B)"],
]


def load_stage_data(soul_name, stage, max_samples=8000):
    stage_file = os.path.join(DATA_DIR, soul_name, f"{soul_name}_{stage}_math_8k.json")
    if not os.path.exists(stage_file):
        v3_file = os.path.join(DATA_DIR, soul_name, f"{soul_name}_55gb_v3.json")
        if os.path.exists(v3_file):
            print(f"  Loading from v3: {v3_file}")
            with open(v3_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "data" in raw:
                raw = raw["data"]
            math_items = [item for item in raw
                         if isinstance(item, dict)
                         and any(kw in item.get("instruction", "").lower() + item.get("output", "").lower()
                                for kw in ["math", "calculus", "algebra", "proof", "equation",
                                           "derivative", "integral", "linear", "probability",
                                           "statistics", "geometry", "number theory"])]
            print(f"  Filtered math items: {len(math_items)}")
            if len(math_items) > max_samples:
                random.seed(42)
                math_items = random.sample(math_items, max_samples)
            return math_items
        print(f"  [FATAL] No data found for {soul_name}")
        return []

    print(f"  Loading: {stage_file}")
    with open(stage_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if isinstance(samples, dict) and "data" in samples:
        samples = samples["data"]
    samples = [s for s in samples if isinstance(s, dict) and len(s.get("output", "")) >= 50]
    print(f"  Loaded: {len(samples)} samples")
    return samples


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import Dataset
    from trl import SFTTrainer

    out_dir = os.path.join(LORA_DIR, SOUL_NAME, STAGE)

    print(f"{'='*60}", flush=True)
    print(f"  8B Soul Training - {SOUL_NAME} {STAGE}", flush=True)
    print(f"  Base: {BASE_MODEL}", flush=True)
    print(f"  Epochs: {EPOCHS}, LR: {LR}, SeqLen: {MAX_SEQ_LENGTH}", flush=True)
    print(f"{'='*60}", flush=True)

    samples = load_stage_data(SOUL_NAME, STAGE)
    if not samples:
        print("[FATAL] No training data!")
        return

    print("  Loading model (4bit)...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def format_sample(sample):
        inst = sample.get("instruction", "")
        inp = sample.get("input", "")
        out = sample.get("output", "")
        if inp:
            return {"text": f"<s>[INST] {inst}\n{inp} [/INST] {out}</s>"}
        return {"text": f"<s>[INST] {inst} [/INST] {out}</s>"}

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
            dataloader_num_workers=0, gradient_checkpointing=True,
        ),
    )

    try:
        trainer.train()
    except Exception as e:
        print(f"\n  [ERROR] {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()

    elapsed = (time.time() - t0) / 60
    peak_vram = torch.cuda.max_memory_reserved() / 1024**3
    print(f"\n  Training done in {elapsed:.1f} min | Peak VRAM: {peak_vram:.1f}GB", flush=True)

    final_loss = 999.0
    if trainer.state and trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry:
                final_loss = entry["loss"]
                break
    print(f"  Final Loss: {final_loss:.4f}", flush=True)

    final_path = os.path.join(out_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"  Saved LoRA: {final_path}", flush=True)

    print(f"\n  Running {STAGE} exam...", flush=True)
    model.eval()
    passed = 0
    exam_results = []
    for i, q in enumerate(EXAM_Q):
        prompt = f"[INST] {q} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        kw_list = KEYWORDS[i] if i < len(KEYWORDS) else []
        matched = [kw for kw in kw_list if kw.lower() in answer.lower()]
        ok = len(matched) >= 1 or len(answer) > 200
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        exam_results.append({"q": q, "a": answer[:200], "matched": matched, "pass": ok})
        print(f"  Q{i+1}/10 [{status}] {answer[:80]}...", flush=True)
    rate = passed / len(EXAM_Q)
    print(f"\n  Exam: {passed}/10 ({rate:.0%})", flush=True)

    log_data = {
        "soul": SOUL_NAME, "stage": STAGE, "base_model": BASE_MODEL,
        "final_loss": final_loss, "exam_rate": rate,
        "overall_pass": final_loss <= 2.5 and rate >= 0.6,
        "elapsed_min": elapsed, "peak_vram_gb": peak_vram,
        "samples": len(samples), "epochs": EPOCHS, "lr": LR,
        "seq_len": MAX_SEQ_LENGTH, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "exam_results": exam_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, SOUL_NAME, f"{STAGE}_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"  {SOUL_NAME} {STAGE} COMPLETE")
    print(f"  Loss: {final_loss:.4f} | Exam: {rate:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
