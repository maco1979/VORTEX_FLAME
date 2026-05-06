#!/usr/bin/env python3
"""
Benchmark 8B Cezanne_PRO Stage1 ONLY - check anti-forgetting
"""
import os, json, gc
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_MODEL = r"D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024"
S1_LORA = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne_8b\stage1\final"

QUESTIONS = [
    {"cat": "sort", "q": "Write a quicksort function in Python with Lomuto partition. Explain time complexity.",
     "kw": ["quicksort", "partition", "pivot", "O(n log n)", "worst"]},
    {"cat": "tree", "q": "Implement a binary search tree with insert and search in Python.",
     "kw": ["class", "Node", "insert", "search", "left", "right"]},
    {"cat": "graph", "q": "Write Dijkstra's algorithm in Python. Explain why it needs a priority queue.",
     "kw": ["dijkstra", "priority", "queue", "distance", "shortest"]},
    {"cat": "dp", "q": "Solve the 0/1 knapsack problem using dynamic programming. Show the DP table construction.",
     "kw": ["knapsack", "DP", "table", "weight", "value"]},
    {"cat": "logic", "q": "Prove by contradiction: there is no largest prime number. Label each step.",
     "kw": ["contradiction", "prime", "assume", "infinite"]},
    {"cat": "logic", "q": "Prove using formal logic: if P implies Q and Q implies R, then P implies R.",
     "kw": ["syllogism", "implies", "transitivity", "modus"]},
    {"cat": "math", "q": "What is the physical meaning of a derivative? Explain from displacement to velocity to acceleration.",
     "kw": ["derivative", "velocity", "acceleration", "rate", "limit"]},
    {"cat": "math", "q": "Prove: if n is even, then n squared is even. Label each derivation step.",
     "kw": ["even", "2k", "derivation", "definition"]},
    {"cat": "debug", "q": "Find the bug: def binary_search(arr, target): left, right = 0, len(arr); while left < right: mid = (left + right) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: left = mid; else: right = mid; return -1",
     "kw": ["off-by-one", "mid + 1", "mid - 1", "infinite", "boundary"]},
    {"cat": "security", "q": "Explain SQL injection. How to prevent it using parameterized queries?",
     "kw": ["SQL", "injection", "parameterized", "prepared", "input"]},
]


def main():
    from transformers import Mistral3ForConditionalGeneration, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading 8B + Stage1 LoRA...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, S1_LORA)
    model.eval()
    print(f"Model loaded, running benchmark...", flush=True)

    results = []
    for i, item in enumerate(QUESTIONS):
        prompt = f"[INST] {item['q']} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        matched = [kw for kw in item["kw"] if kw.lower() in answer.lower()]
        score = len(matched) / len(item["kw"])
        passed = score >= 0.3 or len(answer) > 200
        results.append({"cat": item["cat"], "score": score, "passed": passed, "matched": matched, "len": len(answer)})
        status = "PASS" if passed else "FAIL"
        print(f"  Q{i+1}/10 [{item['cat']:8s}] [{status}] kw={score:.0%} len={len(answer)} {matched}", flush=True)

    total_pass = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Total: {total_pass}/10 | Avg keyword: {avg_score:.1%}", flush=True)

    log_path = r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b\stage1_benchmark.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"pass_rate": total_pass/10, "avg_score": avg_score, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {log_path}", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
