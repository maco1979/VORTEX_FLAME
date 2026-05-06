#!/usr/bin/env python3
"""Cezanne 7B Stage3 — Computer Science (CSAPP/MIT/CS149)
Continues from Stage2 LoRA (best stable state)
Key fix: r=8 (down from 16) to preserve base capabilities
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
BASE_MODEL = r"/mnt/d/models/Mistral-7B-Instruct-v0.1"
S2_LORA = r"/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/stage2/final"
DATA_DIR = r"/mnt/d/VORTEX_FLAME/soul_training_data/cezanne"
LORA_DIR = r"/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne"
LOG_DIR = r"/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"

os.makedirs(LOG_DIR, exist_ok=True)

LORA_R = 8
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LENGTH = 256
EPOCHS = 2
LR = 1.5e-4


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    out_dir = os.path.join(LORA_DIR, "stage3_cs")

    print(f"{'='*60}", flush=True)
    print(f"  {SOUL_NAME} Stage3: Computer Science", flush=True)
    print(f"  Base: {BASE_MODEL}", flush=True)
    print(f"  From Stage2 LoRA: {S2_LORA}", flush=True)
    print(f"  LoRA r={LORA_R} alpha={LORA_ALPHA} (reduced from r=16)", flush=True)
    print(f"  Epochs: {EPOCHS}, LR: {LR}, SeqLen: {MAX_SEQ_LENGTH}", flush=True)
    print(f"{'='*60}", flush=True)

    print("  Loading data...", flush=True)
    s3_path = os.path.join(DATA_DIR, "cezanne_stage3_cs_v1.json")
    with open(s3_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    samples = [s for s in samples if isinstance(s, dict) and len(s.get("output", "")) >= 50]
    print(f"  Loaded: {len(samples)} samples", flush=True)

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
    print("  Stage2 LoRA loaded", flush=True)

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
    sft_config = SFTConfig(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=0.05, num_train_epochs=EPOCHS,
        learning_rate=LR, fp16=True,
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
        "soul": SOUL_NAME, "stage": "stage3_cs",
        "final_loss": final_loss, "elapsed_min": elapsed, "peak_vram_gb": peak_vram,
        "samples": len(samples), "epochs": EPOCHS, "lr": LR, "seq_len": MAX_SEQ_LENGTH,
        "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        "base_model": BASE_MODEL, "continued_from": S2_LORA,
        "data_source": "CSAPP+MIT6.S081+MIT6.006+MIT6.824+CS149+PFRA",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_path = os.path.join(LOG_DIR, "stage3_cs_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    del model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
