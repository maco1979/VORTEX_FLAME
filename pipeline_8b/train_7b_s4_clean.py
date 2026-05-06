#!/usr/bin/env python3
"""7B Clean Stage4 — S2(74%) merged + 74 rescue items, Windows safe"""
import os, sys, json, gc, time, faulthandler, multiprocessing
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
faulthandler.enable()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"

SOUL = "Cezanne_7B_S4"
BASE = r"E:\models\Mistral-7B-Instruct-v0.1-Cezanne"
S2   = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage2\final"
DATA = r"D:\VORTEX_FLAME\arena_results\arena_stage4_data.json"
OUT  = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage4"
LOG  = r"D:\VORTEX_FLAME\hermes_logs\cezanne"
os.makedirs(LOG, exist_ok=True)

LORA_R, LORA_ALPHA = 16, 32
BATCH, ACCUM = 1, 2
SEQ_LEN, EPOCHS, LR = 128, 1, 1e-4

print("=" * 60, flush=True)
print(f"  {SOUL} Clean Stage4", flush=True)
print(f"  Base: {BASE}", flush=True)
print(f"  S2:   {S2}", flush=True)
print(f"  Data: {DATA}", flush=True)
print("=" * 60, flush=True)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from datasets import Dataset
    from trl import SFTTrainer

    with open(DATA, "r", encoding="utf-8") as f:
        samples = json.load(f)
    print(f"  {len(samples)} samples", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = PeftModel.from_pretrained(model, S2)
    print("  S2=74% loaded", flush=True)
    model = model.merge_and_unload()
    print("  S2 merged into base", flush=True)
    model.enable_input_require_grads()

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none"))
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
            logging_steps=20, optim="adamw_torch", weight_decay=0.01,
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
        "soul":SOUL,"stage":"stage4_clean","loss":loss,"elapsed":elapsed,
        "vram":vram,"samples":len(samples),"base":BASE,"s2":S2,
        "time":time.strftime("%Y-%m-%d %H:%M:%S")
    }, open(os.path.join(LOG,"stage4_clean_result.json"),"w",encoding="utf-8"), indent=2)

    del model, trainer, tok; gc.collect(); torch.cuda.empty_cache()
