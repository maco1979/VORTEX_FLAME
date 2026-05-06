#!/usr/bin/env python3
"""7B Cezanne Full Diagnostic: Base + S1 + S2 + S3a + S3b + S4 Clean"""
import os, json, gc, time, sys
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE = r"/mnt/d/models/Mistral-7B-Instruct-v0.1"
LOG = r"/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"

LORAS = {
    "Base": None,
    "S1_Math": r"/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/stage1/final",
    "S2_Logic": r"/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/stage2/final",
    "S3a_Code": r"/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/stage3a/final",
    "S3b_Supp": r"/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/stage3b/final",
    "S4_Clean": r"/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/stage4/final",
}

QUESTIONS = [
    {"cat": "sort", "q": "Write quicksort in Python with Lomuto partition. Explain time complexity.",
     "kw": ["quicksort", "partition", "pivot", "O(n log n)", "worst"]},
    {"cat": "tree", "q": "Implement BST with insert and search in Python.",
     "kw": ["class", "Node", "insert", "search", "left", "right"]},
    {"cat": "graph", "q": "Write Dijkstra algorithm in Python. Explain why priority queue is needed.",
     "kw": ["dijkstra", "priority", "queue", "distance", "shortest"]},
    {"cat": "dp", "q": "Solve 0/1 knapsack with DP. Show the table construction.",
     "kw": ["knapsack", "DP", "table", "weight", "value"]},
    {"cat": "logic", "q": "Prove by contradiction: there is no largest prime number. Label each step.",
     "kw": ["contradiction", "prime", "assume", "infinite"]},
    {"cat": "logic2", "q": "Prove using formal logic: if P implies Q and Q implies R, then P implies R.",
     "kw": ["syllogism", "implies", "transitivity", "modus"]},
    {"cat": "math1", "q": "What is the physical meaning of a derivative? From displacement to velocity to acceleration.",
     "kw": ["derivative", "velocity", "acceleration", "rate", "limit"]},
    {"cat": "math2", "q": "Prove: if n is even, then n squared is even. Label each derivation step.",
     "kw": ["even", "2k", "derivation", "definition"]},
    {"cat": "debug", "q": "Find the bug: def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1",
     "kw": ["off-by-one", "mid + 1", "mid - 1", "infinite", "boundary"]},
    {"cat": "security", "q": "Explain SQL injection. How to prevent it using parameterized queries?",
     "kw": ["SQL", "injection", "parameterized", "prepared", "input"]},
    {"cat": "formal", "q": "Write a formal proof: for all integers n, if n^2 is even then n is even. Use contrapositive.",
     "kw": ["contrapositive", "even", "odd", "assume", "contradiction"]},
    {"cat": "hoare", "q": "Explain Hoare logic. Write a Hoare triple for a simple assignment x = x + 1.",
     "kw": ["hoare", "triple", "precondition", "postcondition", "invariant"]},
    {"cat": "os", "q": "Explain virtual memory and page tables. How does TLB speed up address translation?",
     "kw": ["virtual", "page", "TLB", "translation", "frame"]},
    {"cat": "tcp", "q": "Explain TCP congestion control. What is the difference between slow start and congestion avoidance?",
     "kw": ["congestion", "slow start", "window", "threshold", "avoidance"]},
    {"cat": "compile", "q": "Explain the phases of a compiler: lexing, parsing, semantic analysis, code generation.",
     "kw": ["lexer", "parser", "semantic", "code generation", "token"]},
]


def bench(lora_path, label):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n{'='*60}", flush=True)
    print(f"  7B Cezanne — {label}", flush=True)
    print(f"  LoRA: {lora_path if lora_path else 'None (base only)'}", flush=True)
    print(f"{'='*60}", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    results = []
    t0 = time.time()
    for i, item in enumerate(QUESTIONS):
        prompt = f"<s>[INST] {item['q']} [/INST] "
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=500, temperature=0.7, do_sample=True)
        ans = tokenizer.decode(out[0], skip_special_tokens=True).replace(
            prompt.replace("<s>", ""), ""
        ).strip()
        matched = [k for k in item["kw"] if k.lower() in ans.lower()]
        score = len(matched) / len(item["kw"])
        passed = score >= 0.3 or len(ans) > 200
        results.append(
            {"cat": item["cat"], "score": score, "passed": passed, "matched": matched, "len": len(ans)}
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"  Q{i+1:2d}/{len(QUESTIONS)} [{item['cat']:9s}] [{status}] kw={score:4.0%} len={len(ans):4d} {matched}",
            flush=True,
        )

    elapsed = time.time() - t0
    avg = sum(r["score"] for r in results) / len(results)
    pct = sum(1 for r in results if r["passed"])
    print(f"\n  {label}: {pct}/{len(QUESTIONS)} PASS | Avg kw: {avg:.1%} | {elapsed:.0f}s", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    return {"label": label, "avg": avg, "pass": pct, "total": len(QUESTIONS), "elapsed": elapsed, "results": results}


def main():
    all_results = {}
    for label, lora_path in LORAS.items():
        r = bench(lora_path, label)
        all_results[label] = r

    print(f"\n{'='*80}", flush=True)
    print(f"  7B CEZANNE FULL DIAGNOSTIC SUMMARY", flush=True)
    print(f"{'='*80}", flush=True)

    cats = [q["cat"] for q in QUESTIONS]
    header = f"  {'Category':10s}"
    for label in LORAS:
        header += f" {label:>10s}"
    print(header, flush=True)
    print(f"  {'-'*10}" + f" {'-'*10}" * len(LORAS), flush=True)

    for i, cat in enumerate(cats):
        row = f"  {cat:10s}"
        for label in LORAS:
            s = all_results[label]["results"][i]["score"]
            row += f" {s:9.0%}"
        print(row, flush=True)

    print(f"  {'AVG':10s}", end="", flush=True)
    for label in LORAS:
        print(f" {all_results[label]['avg']:9.1%}", end="", flush=True)
    print(flush=True)

    print(f"  {'PASS':10s}", end="", flush=True)
    for label in LORAS:
        r = all_results[label]
        print(f" {r['pass']:3d}/{r['total']:2d}   ", end="", flush=True)
    print(flush=True)

    s2_avg = all_results["S2_Logic"]["avg"]
    s3b_avg = all_results["S3b_Supp"]["avg"]
    s4_avg = all_results["S4_Clean"]["avg"]
    print(f"\n  S2->S3b delta: {s3b_avg - s2_avg:+.1%} ({'REGRESSION!' if s3b_avg < s2_avg else 'OK'})", flush=True)
    print(f"  S2->S4  delta: {s4_avg - s2_avg:+.1%} ({'REGRESSION!' if s4_avg < s2_avg else 'OK'})", flush=True)

    best_label = max(all_results, key=lambda k: all_results[k]["avg"])
    print(f"  Best stage: {best_label} ({all_results[best_label]['avg']:.1%})", flush=True)

    log_path = os.path.join(LOG, "full_diagnostic_7b.json")
    os.makedirs(LOG, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {log_path}", flush=True)


if __name__ == "__main__":
    main()
