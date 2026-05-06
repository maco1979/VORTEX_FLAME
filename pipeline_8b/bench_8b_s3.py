#!/usr/bin/env python3
"""Benchmark 8B Cezanne_PRO Stage3 final (4-stage LoRA chain) + compare with S1/S2"""
import os, json, gc, time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_MODEL = r"D:\models\Ministral-8B-Reasoning-Text-34L-ctx1024"
S3B_LORA = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne_8b\stage3b\final"

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

    # Load previous results for comparison
    s1_prev = json.load(open(r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b\stage1_benchmark.json", "r", encoding="utf-8"))
    s2_prev = json.load(open(r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b\stage2_benchmark.json", "r", encoding="utf-8"))

    print(f"Loading 8B + Stage3b LoRA (4-stage chain)...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(BASE_MODEL, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, S3B_LORA)
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

    # Compare with S1 and S2
    print(f"\n{'='*50}", flush=True)
    print(f"  S1 vs S2 vs S3 COMPARISON", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  {'Category':10s} {'S1':>6s} {'S2':>6s} {'S3':>6s} {'趋势':>8s}", flush=True)
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*8}", flush=True)

    s1_results = {r["cat"]: r for r in s1_prev["results"]}
    s2_results = {r["cat"]: r for r in s2_prev["results"]}
    s3_results = {r["cat"]: r for r in results}

    all_cats = ["sort","tree","graph","dp","logic","logic","math","math","debug","security"]
    seen = set()
    for i, cat in enumerate(all_cats):
        if cat in seen:
            cat_key = f"{cat}2"
        else:
            cat_key = cat
            seen.add(cat)
        s1s = s1_results.get(cat_key, s1_prev["results"][i]) if cat_key in s1_results else s1_prev["results"][i]
        s2s = s2_results.get(cat_key, s2_prev["results"][i]) if cat_key in s2_results else s2_prev["results"][i]
        s3s = results[i]
        s1v = s1s["score"] if isinstance(s1s, dict) else s1_prev["results"][i]["score"]
        s2v = s2s["score"] if isinstance(s2s, dict) else s2_prev["results"][i]["score"]
        s3v = s3s["score"]
        trend = "+" if s3v > s2v else ("-" if s3v < s2v else "=")
        print(f"  {results[i]['cat']:10s} {s1v:5.0%}  {s2v:5.0%}  {s3v:5.0%}  [{trend}]", flush=True)

    print(f"  {'AVG':10s} {s1_prev['avg_score']:5.0%}  {s2_prev['avg_score']:5.0%}  {avg_score:5.0%}", flush=True)

    # Save
    log_path = r"D:\VORTEX_FLAME\hermes_logs\cezanne_8b\stage3_benchmark.json"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "pass_rate": total_pass/10, "avg_score": avg_score, "results": results,
            "s1_avg": s1_prev["avg_score"], "s2_avg": s2_prev["avg_score"]
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {log_path}", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
