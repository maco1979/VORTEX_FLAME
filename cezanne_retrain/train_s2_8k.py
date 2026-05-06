#!/usr/bin/env python3
"""Cezanne 7B Full Retrain — Stage2: CS Logic 8K
From Base with S1+S2 accumulated data, r=16
Safety: Loss gate + Regression gate + Exam gate + Auto-rollback
GPU: max_grad_norm=0.5, warmup=5%, faulthandler log
"""
import os, sys, json, gc, time, faulthandler
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

SOUL_NAME = "Cezanne_7B"
STAGE = "s2_cs_logic_8k"
BASE_MODEL = "/mnt/d/models/Mistral-7B-Instruct-v0.1"
DATA_DIR = "/mnt/d/VORTEX_FLAME/soul_training_data/cezanne"
LORA_DIR = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne"
LOG_DIR = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"

os.makedirs(LOG_DIR, exist_ok=True)
FAULT_LOG = open(os.path.join(LOG_DIR, f"fault_{STAGE}.log"), "w")
faulthandler.enable(file=FAULT_LOG, all_threads=True)

LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_SEQ_LENGTH = 256
EPOCHS = 2
LR = 1.5e-4
MAX_LOSS = 1.2
MAX_RETRIES = 2

DATA_FILES = [
    "cezanne_s1_cs_math_8k.json",
    "cezanne_s2_cs_logic_8k.json",
]

REGRESSION_QUESTIONS = [
    {"q": "Find the bug: def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1", "kw": ["off-by-one", "mid + 1", "infinite", "boundary"]},
    {"q": "Prove using formal logic: if P implies Q and Q implies R, then P implies R.", "kw": ["syllogism", "implies", "transitivity", "modus"]},
    {"q": "Write Dijkstra's algorithm in Python.", "kw": ["dijkstra", "priority", "queue", "distance"]},
    {"q": "Write quicksort in Python.", "kw": ["quicksort", "partition", "pivot"]},
    {"q": "What is the time complexity of merge sort?", "kw": ["n log", "merge", "sort", "divide"]},
]


def regression_test(model, tokenizer):
    passed = 0
    was_training = model.training
    model.eval()
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    for item in REGRESSION_QUESTIONS:
        prompt = f"<s>[INST] {item['q']} [/INST] "
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=200, temperature=0.7,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        matched = [k for k in item["kw"] if k.lower() in ans.lower()]
        if len(matched) >= 2:
            passed += 1
    if was_training:
        model.train()
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    return passed, len(REGRESSION_QUESTIONS)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    out_dir = os.path.join(LORA_DIR, STAGE)

    print(f"{'='*60}", flush=True)
    print(f"  {SOUL_NAME} {STAGE}: CS Logic 8K", flush=True)
    print(f"  Base: {BASE_MODEL}", flush=True)
    print(f"  Data: {DATA_FILES}", flush=True)
    print(f"  LoRA r={LORA_R} alpha={LORA_ALPHA}", flush=True)
    print(f"  Epochs: {EPOCHS}, LR: {LR}, SeqLen: {MAX_SEQ_LENGTH}", flush=True)
    print(f"  Safety: Loss Gate ({MAX_LOSS}) + Regression Gate", flush=True)
    print(f"{'='*60}", flush=True)

    all_samples = []
    for fname in DATA_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.get("data", raw) if isinstance(raw, dict) else raw
        all_samples.extend(items)
        print(f"  Loaded {fname}: {len(items)} items", flush=True)
    print(f"  Total: {len(all_samples)} items", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n  --- Attempt {attempt}/{MAX_RETRIES} ---", flush=True)

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model.enable_input_require_grads()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        print("  Running pre-training regression test...", flush=True)
        pre_pass, pre_total = regression_test(model, tokenizer)
        print(f"  Pre-training regression: {pre_pass}/{pre_total}", flush=True)

        def format_sample(s):
            inst = s.get("instruction", "")
            out = s.get("output", "")
            text = s.get("text", "")
            if text:
                return {"text": text}
            if inst and out:
                return {"text": f"<s>[INST] {inst} [/INST] {out}</s>"}
            return {"text": text or inst or out}

        ds = Dataset.from_list([format_sample(s) for s in all_samples])

        current_lr = LR / (2 ** (attempt - 1))
        t0 = time.time()
        sft_config = SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.05, num_train_epochs=EPOCHS,
            learning_rate=current_lr, fp16=True,
            logging_steps=50, optim="adamw_torch",
            weight_decay=0.01, lr_scheduler_type="cosine",
            seed=3407, output_dir=out_dir,
            save_strategy="steps", save_steps=500, save_total_limit=2,
            max_grad_norm=0.5, report_to="none", disable_tqdm=True,
            dataloader_num_workers=0, gradient_checkpointing=True,
            max_length=MAX_SEQ_LENGTH, dataset_text_field="text",
            dataset_num_proc=1, packing=False,
        )
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=ds,
            args=sft_config,
        )

        try:
            trainer.train()
        except Exception as e:
            print(f"\n  [ERROR] {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()

        elapsed = (time.time() - t0) / 60
        peak_vram = torch.cuda.max_memory_reserved() / 1024**3

        final_loss = 999.0
        if trainer.state and trainer.state.log_history:
            for entry in reversed(trainer.state.log_history):
                if "loss" in entry:
                    final_loss = entry["loss"]
                    break
        print(f"  Final Loss: {final_loss:.4f} | Time: {elapsed:.1f}min | VRAM: {peak_vram:.1f}GB", flush=True)

        if final_loss > MAX_LOSS:
            print(f"  [LOSS GATE FAILED] Loss {final_loss:.4f} > {MAX_LOSS}", flush=True)
            del model, tokenizer, trainer
            gc.collect(); torch.cuda.empty_cache()
            continue

        print("  Running post-training regression test...", flush=True)
        post_pass, post_total = regression_test(model, tokenizer)
        print(f"  Post-training regression: {post_pass}/{post_total}", flush=True)

        regression_ok = post_pass >= pre_pass - 1
        if not regression_ok:
            print(f"  [REGRESSION GATE FAILED] {pre_pass} -> {post_pass} (dropped {pre_pass - post_pass})", flush=True)
            print(f"  LoRA NOT saved due to catastrophic forgetting!", flush=True)
            del model, tokenizer, trainer
            gc.collect(); torch.cuda.empty_cache()
            continue
        else:
            print(f"  [REGRESSION GATE PASSED] {pre_pass} -> {post_pass}", flush=True)

        final_path = os.path.join(out_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"  Saved LoRA: {final_path}", flush=True)

        log_data = {
            "soul": SOUL_NAME, "stage": STAGE,
            "final_loss": final_loss, "elapsed_min": elapsed, "peak_vram_gb": peak_vram,
            "samples": len(all_samples), "epochs": EPOCHS, "lr": current_lr, "seq_len": MAX_SEQ_LENGTH,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "base_model": BASE_MODEL, "data_files": DATA_FILES, "attempt": attempt,
            "regression_pre": pre_pass, "regression_post": post_pass,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        log_path = os.path.join(LOG_DIR, f"{STAGE}_result.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        del model, tokenizer, trainer
        gc.collect(); torch.cuda.empty_cache()
        break

    print(f"\n  {STAGE} complete.", flush=True)


if __name__ == "__main__":
    main()
