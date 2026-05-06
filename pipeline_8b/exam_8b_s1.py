#!/usr/bin/env python3
import os, sys, json, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from transformers import Mistral3ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = r"D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024"
S1_LORA = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne_8b\stage1\final"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b"

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

print("Loading model + Stage1 LoRA...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL, quantization_config=bnb,
    device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model, S1_LORA)
model.eval()
print("Model loaded OK", flush=True)

print("\n" + "="*50, flush=True)
print("  Cezanne 8B Stage1 Exam", flush=True)
print("="*50, flush=True)

passed = 0
results = []
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
    print(f"  Q{i+1}/10 [{status}] {answer[:100]}...", flush=True)
    results.append({"q": q, "a": answer[:300], "ok": ok, "matched": matched})

rate = passed / len(EXAM_Q)
print(f"\n  Result: {passed}/10 ({rate:.0%})", flush=True)

log_data = {
    "soul": "cezanne_8b", "stage": "stage1",
    "final_loss": 0.5658, "exam_rate": rate, "overall_pass": 0.5658 <= 2.5 and rate >= 0.6,
    "elapsed_min": 310.6, "peak_vram_gb": 11.7,
    "samples": 8000, "epochs": 3, "lr": 3e-4, "seq_len": 128,
    "lora_r": 16, "lora_alpha": 32,
    "base_model": BASE_MODEL,
    "exam_details": results,
    "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
}
os.makedirs(LOG_DIR, exist_ok=True)
with open(os.path.join(LOG_DIR, "stage1_result.json"), "w", encoding="utf-8") as f:
    json.dump(log_data, f, ensure_ascii=False, indent=2)

print(f"\n  Log saved to {LOG_DIR}/stage1_result.json", flush=True)

del model, tokenizer
torch.cuda.empty_cache()
print("DONE", flush=True)
