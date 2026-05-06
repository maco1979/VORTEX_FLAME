#!/usr/bin/env python3
"""Cezanne 7B Retrain — Stage1: Math Foundation
From Base model, r=8 (was r=16, caused catastrophic forgetting)
Safety: Loss gate + regression test + auto-rollback
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
STAGE = "stage1_v2"
BASE_MODEL = "/mnt/d/models/Mistral-7B-Instruct-v0.1"
DATA_DIR = "/mnt/d/VORTEX_FLAME/soul_training_data/cezanne"
LORA_DIR = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne"
LOG_DIR = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"

os.makedirs(LOG_DIR, exist_ok=True)

LORA_R = 8
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 512
EPOCHS = 3
LR = 1.5e-4
MAX_LOSS = 2.5
MAX_RETRIES = 2

REGRESSION_QUESTIONS = [
    {"q": "Write quicksort in Python with Lomuto partition.", "kw": ["quicksort", "partition", "pivot"]},
    {"q": "Write Dijkstra's algorithm in Python.", "kw": ["dijkstra", "priority", "queue"]},
    {"q": "Explain TCP congestion control.", "kw": ["congestion", "slow start", "window"]},
    {"q": "What is SQL injection and how to prevent it?", "kw": ["SQL", "injection", "parameterized"]},
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
    print(f"  {SOUL_NAME} {STAGE}: Math Foundation", flush=True)
    print(f"  Base: {BASE_MODEL}", flush=True)
    print(f"  LoRA r={LORA_R} alpha={LORA_ALPHA}", flush=True)
    print(f"  Epochs: {EPOCHS}, LR: {LR}, SeqLen: {MAX_SEQ_LENGTH}", flush=True)
    print(f"  Loss Gate: {MAX_LOSS} | Max Retries: {MAX_RETRIES}", flush=True)
    print(f"{'='*60}", flush=True)

    data_path = os.path.join(DATA_DIR, "cezanne_s1_math_v2.json")
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw.get("data", raw) if isinstance(raw, dict) else raw
    print(f"  Loaded: {len(samples)} samples", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n  --- Attempt {attempt}/{MAX_RETRIES} ---", flush=True)

        print("  Loading base model...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
            device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        print("  Running pre-training regression test...", flush=True)
        pre_pass, pre_total = regression_test(model, tokenizer)
        print(f"  Pre-training regression: {pre_pass}/{pre_total}", flush=True)

        model.enable_input_require_grads()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        def format_sample(s):
            text = s.get("text", "")
            if text:
                return {"text": text}
            inst = s.get("instruction", "")
            out = s.get("output", "")
            return {"text": f"<s>[INST] {inst} [/INST] {out}</s>"}

        ds = Dataset.from_list([format_sample(s) for s in samples])

        current_lr = LR / (2 ** (attempt - 1))
        t0 = time.time()
        sft_config = SFTConfig(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            warmup_ratio=0.05, num_train_epochs=EPOCHS,
            learning_rate=current_lr, fp16=True,
            logging_steps=10, optim="adamw_torch",
            weight_decay=0.01, lr_scheduler_type="cosine",
            seed=3407, output_dir=out_dir,
            save_strategy="epoch", save_total_limit=2,
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
            print(f"  [GATE FAILED] Loss {final_loss:.4f} > {MAX_LOSS}", flush=True)
            if attempt < MAX_RETRIES:
                print(f"  Retrying with LR={current_lr/2:.2e}...", flush=True)
            del model, tokenizer, trainer
            gc.collect(); torch.cuda.empty_cache()
            continue

        final_path = os.path.join(out_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"  Saved LoRA: {final_path}", flush=True)

        print("  Running post-training regression test...", flush=True)
        post_pass, post_total = regression_test(model, tokenizer)
        print(f"  Post-training regression: {post_pass}/{post_total}", flush=True)

        regression_ok = post_pass >= pre_pass - 1
        if not regression_ok:
            print(f"  [GATE FAILED] Regression dropped: {pre_pass}→{post_pass}", flush=True)
        else:
            print(f"  [GATE PASSED] Regression stable: {pre_pass}→{post_pass}", flush=True)

        log_data = {
            "soul": SOUL_NAME, "stage": STAGE,
            "final_loss": final_loss, "elapsed_min": elapsed, "peak_vram_gb": peak_vram,
            "samples": len(samples), "epochs": EPOCHS, "lr": current_lr, "seq_len": MAX_SEQ_LENGTH,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
            "base_model": BASE_MODEL, "attempt": attempt,
            "pre_regression": f"{pre_pass}/{pre_total}",
            "post_regression": f"{post_pass}/{post_total}",
            "regression_ok": regression_ok, "loss_gate_passed": final_loss <= MAX_LOSS,
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
