#!/usr/bin/env python3
"""8B Stage4 — S3b(82%) + 84 rescue+CS items, PEFT stack, Windows safe"""
import os, sys, json, gc, time, faulthandler, multiprocessing
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SOUL = "Cezanne_8B_PRO_S4"
BASE = r"D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024"
S3B  = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne_8b\stage3b\final"
DATA = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b\cezanne_pro_8b_stage4_v1.json"
OUT  = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne_8b\stage4"
LOG  = r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b"
os.makedirs(LOG, exist_ok=True)

LORA_R, LORA_ALPHA = 8, 16
BATCH, ACCUM = 1, 4
SEQ_LEN, EPOCHS, LR = 128, 1, 1e-4

print("=" * 60, flush=True)
print(f"  {SOUL} Stage4", flush=True)
print(f"  VRAM: 3060 12GB | 8B 4bit ~4GB | S3b ~1.5GB | train ~2.5GB = ~8GB safe", flush=True)
print("=" * 60, flush=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    from transformers import AutoTokenizer, TrainingArguments
    from transformers import Mistral3ForConditionalGeneration
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from datasets import Dataset
    from trl import SFTTrainer

    with open(DATA, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples", flush=True)

    print("  Loading 8B base (4bit)...", flush=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    print("  Loading S3b LoRA (82%)...", flush=True)
    model = PeftModel.from_pretrained(model, S3B)
    print("  S3b=82% loaded (kept as PEFT, no merge)", flush=True)

    print("  Adding Stage4 LoRA...", flush=True)
    model.add_adapter("stage4", LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none"))
    model.set_adapter("stage4")
    model.print_trainable_parameters()

    def fmt(s):
        inst = s.get("instruction",""); inp = s.get("input",""); out = s.get("output","")
        if inp: return {"text": f"<s>[INST] {inst}\n{inp} [/INST] {out}</s>"}
        return {"text": f"<s>[INST] {inst} [/INST] {out}</s>"}

    ds = Dataset.from_list([fmt(s) for s in samples])

    t0 = time.time()
    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=ds,
        max_seq_length=SEQ_LEN, dataset_text_field="text",
        dataset_num_proc=1, packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
            warmup_ratio=0.05, num_train_epochs=EPOCHS, learning_rate=LR, fp16=True,
            logging_steps=10, optim="adamw_torch", weight_decay=0.01,
            lr_scheduler_type="cosine", seed=3407, output_dir=OUT,
            save_strategy="epoch", save_total_limit=2, max_grad_norm=0.5,
            report_to="none", disable_tqdm=True, dataloader_num_workers=0,
            gradient_checkpointing=True))
    trainer.train()

    elapsed = (time.time() - t0) / 60
    vram = torch.cuda.max_memory_reserved() / 1024**3
    loss = next((e["loss"] for e in reversed(trainer.state.log_history) if "loss" in e), 999)
    print(f"\n  Done {elapsed:.1f}min VRAM={vram:.1f}GB Loss={loss:.4f}", flush=True)

    fp = os.path.join(OUT, "final")
    os.makedirs(fp, exist_ok=True)
    model.save_pretrained(fp); tok.save_pretrained(fp)
    print(f"  Saved: {fp}", flush=True)

    json.dump({
        "soul":SOUL,"stage":"stage4","loss":loss,"elapsed":elapsed,
        "vram":vram,"samples":len(samples),"base":BASE,"s3b":S3B,
        "time":time.strftime("%Y-%m-%d %H:%M:%S")
    }, open(os.path.join(LOG,"stage4_result.json"),"w",encoding="utf-8"), indent=2)

    del model, trainer, tok; gc.collect(); torch.cuda.empty_cache()
