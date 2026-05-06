#!/usr/bin/env python3
"""Cezanne 7B Stage4 — Rescue keywords + arena debug, from Stage3b"""
import os, sys, json, gc, time, faulthandler
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

SOUL_NAME = "Cezanne_7B"
BASE_MODEL = r"E:\models\Mistral-7B-Instruct-v0.1-Cezanne"
S2_LORA = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage2\final"  # S2=74% is 7B's best state
DATA_PATH = r"D:\VORTEX_FLAME\arena_results\arena_stage4_data.json"
LORA_DIR = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs\cezanne"

os.makedirs(LOG_DIR, exist_ok=True)

LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 2
MAX_SEQ_LENGTH = 128
EPOCHS = 1
LR = 1e-4


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from datasets import Dataset
    from trl import SFTTrainer

    out_dir = os.path.join(LORA_DIR, "stage4")

    print(f"{'='*60}", flush=True)
    print(f"  {SOUL_NAME} Stage4: Rescue + Arena", flush=True)
    print(f"  From Stage2 LoRA: {S2_LORA}", flush=True)
    print(f"  Epochs: {EPOCHS}, LR: {LR}", flush=True)
    print(f"{'='*60}", flush=True)

    print("  Loading data...", flush=True)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"  Loaded: {len(samples)} samples (lightweight rescue)", flush=True)

    print("  Loading model + Stage2 LoRA...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = PeftModel.from_pretrained(model, S2_LORA)
    print("  Stage2 LoRA loaded (frozen)", flush=True)

    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
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
            logging_steps=20, optim="adamw_torch",
            weight_decay=0.01, lr_scheduler_type="cosine",
            seed=3407, output_dir=out_dir,
            save_strategy="epoch", save_total_limit=2,
            max_grad_norm=0.5, report_to="none", disable_tqdm=True,
            dataloader_num_workers=0, gradient_checkpointing=True,
        ),
    )

    trainer.train()

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

    log_data = {
        "soul": SOUL_NAME, "stage": "stage4",
        "final_loss": final_loss, "elapsed_min": elapsed, "peak_vram_gb": peak_vram,
        "samples": len(samples), "epochs": EPOCHS, "lr": LR, "seq_len": MAX_SEQ_LENGTH,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "base_model": BASE_MODEL, "continued_from": S2_LORA,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, "stage4_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
