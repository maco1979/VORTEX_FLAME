#!/usr/bin/env python3
import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

from transformers import Mistral3ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig

model_dir = r"D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024"
log_path = r"D:\VORTEX_FLAME\pipeline_8b\inference_test_v2.txt"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = Mistral3ForConditionalGeneration.from_pretrained(model_dir, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

vram = torch.cuda.memory_reserved() / 1024**3
results = [f"VRAM: {vram:.1f}GB\n"]

prompts = [
    "What is 2+3?",
    "Explain what a process is in operating systems.",
    "Write a function to check if a number is prime.",
]
for p in prompts:
    inputs = tokenizer(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    results.append(f"Q: {p}")
    results.append(f"A: {answer[:300]}")
    results.append("---")

results.append("DONE")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

del model
torch.cuda.empty_cache()
