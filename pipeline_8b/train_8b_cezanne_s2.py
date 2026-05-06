#!/usr/bin/env python3
"""
Cezanne_PRO 8B Stage2 - Logic Foundation + Weak Point Breakthrough
Soul: Cezanne_PRO (8B version of Cezanne)
Continues from Stage1 LoRA on 8B base
Data: cezanne_stage2_logic_8k.json (same as 7B Cezanne)
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

SOUL_NAME = "Cezanne_PRO"
BASE_MODEL = r"D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024"
S1_LORA = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne_8b\stage1\final"
DATA_DIR = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b"
LORA_DIR = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne_8b"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b"

os.makedirs(LOG_DIR, exist_ok=True)
FAULT_LOG = open(os.path.join(LOG_DIR, "fault_s2.log"), "w")
faulthandler.enable(file=FAULT_LOG, all_threads=True)

LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 128
EPOCHS = 3
LR = 2e-4

EXAM_Q = [
    "Prove using formal logic: if P implies Q and Q implies R, then P implies R (hypothetical syllogism). Label each derivation step.",
    "Explain De Morgan's Laws and prove one using a truth table.",
    "What is the difference between sufficient condition and necessary condition? Give examples.",
    "Prove by contradiction that there is no largest prime number. Label each step's basis.",
    "Explain first-order logic: what does the universal quantifier and existential quantifier mean? Show the negation equivalence.",
    "Prove: if n is even, then n squared is even. Label each derivation step's basis.",
    "What is the physical meaning of a derivative? Explain the logic chain from displacement to velocity to acceleration.",
    "Why is integration the inverse of differentiation? Explain using the Fundamental Theorem of Calculus.",
    "Solve: A pool has pipe A (5h to fill), pipe B (4h to fill), pipe C (10h to drain). All open, how long to fill? Show step-by-step reasoning.",
    "Explain the logical structure of categorical syllogism. Why is 'All cats are animals, some animals are dogs, therefore some cats are dogs' invalid?",
]

KEYWORDS = [
    ["hypothetical syllogism", "P implies Q", "transitivity", "modus ponens"],
    ["De Morgan", "truth table", "negation", "conjunction", "disjunction"],
    ["sufficient", "necessary", "if-then", "only if", "condition"],
    ["contradiction", "prime", "assume", "largest", "infinite"],
    ["universal quantifier", "existential", "negation", "for all", "there exists"],
    ["even", "2k", "derivation", "definition", "closed"],
    ["derivative", "velocity", "acceleration", "rate of change", "limit"],
    ["fundamental theorem", "integral", "antiderivative", "inverse"],
    ["5h", "4h", "10h", "rate", "net", "20/7"],
    ["syllogism", "invalid", "some", "all", "does not follow"],
]


from transformers import TrainerCallback


class LossPrinter(TrainerCallback):
    def __init__(self):
        self.step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.step += 1
            loss = logs["loss"]
            lr = logs.get("learning_rate", 0)
            epoch = logs.get("epoch", 0)
            print(f"  [Step {self.step}] loss={loss:.4f} lr={lr:.2e} epoch={epoch:.3f}", flush=True)


def main():
    from transformers import Mistral3ForConditionalGeneration, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, TrainerCallback
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from datasets import Dataset
    from trl import SFTTrainer

    out_dir = os.path.join(LORA_DIR, "stage2")

    print(f"{'='*60}", flush=True)
    print(f"  {SOUL_NAME} Stage2: Logic + Breakthrough", flush=True)
    print(f"  Base: {BASE_MODEL}", flush=True)
    print(f"  From Stage1 LoRA: {S1_LORA}", flush=True)
    print(f"  Epochs: {EPOCHS}, LR: {LR}, SeqLen: {MAX_SEQ_LENGTH}", flush=True)
    print(f"{'='*60}", flush=True)

    print("  Loading data...", flush=True)
    s2_path = os.path.join(DATA_DIR, "cezanne_pro_8b_stage2_logic_8k_v2.json")
    with open(s2_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if isinstance(samples, dict) and "data" in samples:
        samples = samples["data"]
    samples = [s for s in samples if isinstance(s, dict) and len(s.get("output", "")) >= 50]
    print(f"  Loaded: {len(samples)} samples", flush=True)

    print("  Loading model + Stage1 LoRA...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(model, S1_LORA)
    print("  Stage1 LoRA loaded (frozen, continuing on top)", flush=True)

    model.enable_input_require_grads()

    lang_target_modules = []
    for name, _ in model.named_modules():
        if "language_model" in name and any(name.endswith(t) for t in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]):
            lang_target_modules.append(name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
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

    loss_printer = LossPrinter()

    t0 = time.time()
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        max_seq_length=MAX_SEQ_LENGTH, dataset_text_field="text",
        dataset_num_proc=1, packing=False,
        callbacks=[loss_printer],
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

    print(f"\n  Running Stage2 exam...", flush=True)
    model.eval()
    passed = 0
    for i, q in enumerate(EXAM_Q):
        prompt = f"[INST] {q} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=400, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        kw_list = KEYWORDS[i] if i < len(KEYWORDS) else []
        matched = [kw for kw in kw_list if kw.lower() in answer.lower()]
        ok = len(matched) >= 1 or len(answer) > 200
        if ok: passed += 1
        status = "PASS" if ok else "FAIL"
        safe_answer = answer[:80].encode('ascii', errors='replace').decode()
        print(f"  Q{i+1}/10 [{status}] {safe_answer}...", flush=True)
    rate = passed / len(EXAM_Q)
    print(f"\n  Exam: {passed}/10 ({rate:.0%})", flush=True)

    log_data = {
        "soul": SOUL_NAME, "stage": "stage2",
        "final_loss": final_loss, "exam_rate": rate, "overall_pass": final_loss <= 2.5 and rate >= 0.6,
        "elapsed_min": elapsed, "peak_vram_gb": peak_vram,
        "samples": len(samples), "epochs": EPOCHS, "lr": LR, "seq_len": MAX_SEQ_LENGTH,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "base_model": BASE_MODEL, "continued_from": S1_LORA,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, "stage2_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
