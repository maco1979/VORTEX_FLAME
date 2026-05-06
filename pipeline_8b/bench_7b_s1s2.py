#!/usr/bin/env python3
"""7B Cezanne Stage1 + Stage2 dual benchmark (serial, compare for anti-forgetting)"""
import os, json, gc, time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_MODEL = r"E:\models\Mistral-7B-Instruct-v0.1-Cezanne"
S1_LORA = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage1\final"
S2_LORA = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage2\final"
LOG_DIR = r"D:\VORTEX_FLAME\hermes_logs\cezanne"

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


def run_bench(lora_path, label):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n{'='*50}", flush=True)
    print(f"  7B {label}", flush=True)
    print(f"  LoRA: {lora_path}", flush=True)
    print(f"{'='*50}", flush=True)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    results = []
    for i, item in enumerate(QUESTIONS):
        prompt = f"<s>[INST] {item['q']} [/INST] "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt.replace("<s>",""), "").strip()
        matched = [kw for kw in item["kw"] if kw.lower() in answer.lower()]
        score = len(matched) / len(item["kw"])
        passed = score >= 0.3 or len(answer) > 200
        results.append({"cat": item["cat"], "score": score, "passed": passed, "matched": matched, "len": len(answer)})
        status = "PASS" if passed else "FAIL"
        print(f"  Q{i+1}/10 [{item['cat']:8s}] [{status}] kw={score:.0%} len={len(answer)} {matched}", flush=True)

    total_pass = sum(1 for r in results if r["passed"])
    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  {label}: {total_pass}/10 | Avg kw: {avg_score:.1%}", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)  # Let GPU fully release

    return {"label": label, "pass_rate": total_pass/10, "avg_score": avg_score, "results": results}


def main():
    s1 = run_bench(S1_LORA, "Stage1 (Math)")
    s2 = run_bench(S2_LORA, "Stage2 (Math+Logic)")

    print(f"\n{'='*50}", flush=True)
    print(f"  7B S1 vs S2 COMPARISON", flush=True)
    print(f"{'='*50}", flush=True)
    print(f"  {'Category':10s} {'S1':>6s} {'S2':>6s} {'变化':>8s}", flush=True)
    print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*8}", flush=True)

    for i, item in enumerate(QUESTIONS):
        s1v = s1["results"][i]["score"]
        s2v = s2["results"][i]["score"]
        diff = s2v - s1v
        if diff > 0.05: trend = f"+{diff:.0%}"
        elif diff < -0.05: trend = f"{diff:.0%}"
        else: trend = "="
        print(f"  {item['cat']:10s} {s1v:5.0%}  {s2v:5.0%}  [{trend}]", flush=True)

    print(f"  {'AVG':10s} {s1['avg_score']:5.0%}  {s2['avg_score']:5.0%}  [{s2['avg_score']-s1['avg_score']:+.0%}]", flush=True)

    # Check anti-forgetting
    math_cats = ["math", "math"]
    s1_math = sum(r["score"] for r in s1["results"] if r["cat"] == "math") / 2
    s2_math = sum(r["score"] for r in s2["results"] if r["cat"] == "math") / 2
    print(f"\n  Math avg: S1={s1_math:.0%} S2={s2_math:.0%} [{'OK, no forgetting' if s2_math >= s1_math * 0.9 else 'WARNING: possible forgetting'}]", flush=True)

    log_path = os.path.join(LOG_DIR, "stage1_stage2_benchmark.json")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"stage1": s1, "stage2": s2, "math_s1": s1_math, "math_s2": s2_math}, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {log_path}", flush=True)


if __name__ == "__main__":
    main()
