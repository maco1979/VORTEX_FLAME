#!/usr/bin/env python3
"""Cezanne 7B AB Test: vs 8B Cezanne + vs Einstein
Compare on CS knowledge questions
Run: conda activate vortex_flame && python ab_test.py
"""
import os, sys, json, gc, time
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

BASE_7B = "/mnt/d/models/Mistral-7B-Instruct-v0.1"
BASE_8B = "/mnt/d/models/Ministral-8B-Reasoning-Text-34L-ctx1024"

CEZANNE_7B_S1 = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/s1_cs_math_8k/final"
CEZANNE_7B_S2 = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/s2_cs_logic_8k/final"
CEZANNE_8B_S3B = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne_8b/stage3b/final"
EINSTEIN_S5 = "/mnt/d/VORTEX_FLAME/soul_lora_v2/einstein/stage5/final"

AB_QUESTIONS = [
    {"cat": "Algorithm", "q": "Write quicksort in Python with Lomuto partition.", "kw": ["quicksort", "partition", "pivot"]},
    {"cat": "Algorithm", "q": "Write Dijkstra's shortest path algorithm in Python.", "kw": ["dijkstra", "priority", "queue", "distance"]},
    {"cat": "Algorithm", "q": "Implement a binary search that returns the index or -1.", "kw": ["binary", "search", "mid", "index"]},
    {"cat": "Algorithm", "q": "Write a function to detect if a linked list has a cycle.", "kw": ["cycle", "slow", "fast", "pointer", "floyd"]},
    {"cat": "DataStructure", "q": "Implement a min-heap with insert and extract_min in Python.", "kw": ["heap", "insert", "extract", "min", "bubble"]},
    {"cat": "DataStructure", "q": "Implement a hash table with chaining for collision resolution.", "kw": ["hash", "table", "chain", "bucket", "collision"]},
    {"cat": "Complexity", "q": "What is the time complexity of merge sort? Explain why.", "kw": ["n log", "merge", "sort", "divide", "conquer"]},
    {"cat": "Complexity", "q": "Explain the difference between O(n) and O(n log n) with examples.", "kw": ["linear", "log", "complexity", "example"]},
    {"cat": "Debug", "q": "Find the bug: def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1", "kw": ["off-by-one", "mid + 1", "infinite", "boundary"]},
    {"cat": "Debug", "q": "Find the bug: class Counter: count=0; def increment(self): self.count+=1. Two threads calling increment() simultaneously.", "kw": ["race", "thread", "lock", "concurrent", "atomic"]},
    {"cat": "Logic", "q": "Prove using formal logic: if P implies Q and Q implies R, then P implies R.", "kw": ["syllogism", "implies", "transitivity", "modus"]},
    {"cat": "Logic", "q": "What is De Morgan's Law? Give an example in boolean algebra.", "kw": ["de morgan", "not", "and", "or", "complement"]},
    {"cat": "Systems", "q": "Explain virtual memory and how page tables work.", "kw": ["virtual", "page", "tlb", "translation", "memory"]},
    {"cat": "Systems", "q": "Explain how malloc() works internally in C.", "kw": ["malloc", "heap", "free", "chunk", "allocator"]},
    {"cat": "Systems", "q": "What is a deadlock? Give an example and explain prevention.", "kw": ["deadlock", "mutex", "lock", "prevention", "circular"]},
    {"cat": "Systems", "q": "Explain the difference between a process and a thread.", "kw": ["process", "thread", "memory", "address", "space"]},
    {"cat": "Network", "q": "Explain TCP congestion control and slow start.", "kw": ["congestion", "slow start", "window", "tcp"]},
    {"cat": "Network", "q": "What is SQL injection and how to prevent it?", "kw": ["sql", "injection", "parameterized", "prepared"]},
    {"cat": "Math", "q": "What is the modular inverse of 3 mod 7? Show the calculation.", "kw": ["modular", "inverse", "3", "7", "5"]},
    {"cat": "Math", "q": "Calculate the number of edges in a complete graph K_n.", "kw": ["complete", "graph", "edge", "n(n-1)/2"]},
]


def run_test(model, tokenizer, name):
    print(f"\n{'='*60}", flush=True)
    print(f"  Testing: {name}", flush=True)
    print(f"{'='*60}", flush=True)

    results = []
    cat_scores = {}

    for item in AB_QUESTIONS:
        cat = item["cat"]
        prompt = f"<s>[INST] {item['q']} [/INST] "
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=300, temperature=0.7,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        ans_clean = ans.replace(prompt, "").strip()

        matched = [k for k in item["kw"] if k.lower() in ans_clean.lower()]
        passed = len(matched) >= 2

        if cat not in cat_scores:
            cat_scores[cat] = {"pass": 0, "total": 0}
        cat_scores[cat]["total"] += 1
        if passed:
            cat_scores[cat]["pass"] += 1

        results.append({
            "cat": cat,
            "q": item["q"][:80],
            "passed": passed,
            "matched": matched,
            "answer_preview": ans_clean[:150],
        })

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {cat}: {item['q'][:60]}... (matched: {matched})", flush=True)

    total_pass = sum(s["pass"] for s in cat_scores.values())
    total_q = sum(s["total"] for s in cat_scores.values())
    rate = total_pass / total_q if total_q > 0 else 0

    print(f"\n  Summary for {name}:", flush=True)
    print(f"  Overall: {total_pass}/{total_q} ({rate:.1%})", flush=True)
    for cat, s in sorted(cat_scores.items()):
        r = s["pass"] / s["total"] if s["total"] > 0 else 0
        print(f"    {cat}: {s['pass']}/{s['total']} ({r:.1%})", flush=True)

    return {"name": name, "total_pass": total_pass, "total_q": total_q, "rate": rate, "cat_scores": cat_scores, "results": results}


def load_model(base_path, lora_path, name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    print(f"\n  Loading {name}...", flush=True)
    print(f"    Base: {base_path}", flush=True)
    print(f"    LoRA: {lora_path}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(base_path, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"    LoRA loaded!", flush=True)

    model.eval()
    return model, tokenizer


def main():
    all_results = []

    configs = []

    if os.path.exists(CEZANNE_7B_S1):
        configs.append(("7B Cezanne S1 (CS Math)", BASE_7B, CEZANNE_7B_S1))
    if os.path.exists(CEZANNE_7B_S2):
        configs.append(("7B Cezanne S2 (CS Logic)", BASE_7B, CEZANNE_7B_S2))
    if os.path.exists(CEZANNE_8B_S3B):
        configs.append(("8B Cezanne S3b (Full)", BASE_8B, CEZANNE_8B_S3B))
    if os.path.exists(EINSTEIN_S5):
        configs.append(("7B Einstein S5 (Physics)", BASE_7B, EINSTEIN_S5))

    if not configs:
        print("  [ERROR] No LoRA found!", flush=True)
        return

    print(f"  Will test {len(configs)} configurations:", flush=True)
    for name, base, lora in configs:
        print(f"    - {name}", flush=True)

    for name, base_path, lora_path in configs:
        model, tokenizer = load_model(base_path, lora_path, name)
        result = run_test(model, tokenizer, name)
        all_results.append(result)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    print(f"\n{'='*60}", flush=True)
    print(f"  AB TEST FINAL COMPARISON", flush=True)
    print(f"{'='*60}", flush=True)

    for r in all_results:
        print(f"  {r['name']}: {r['total_pass']}/{r['total_q']} ({r['rate']:.1%})", flush=True)

    print(f"\n  Category breakdown:", flush=True)
    all_cats = sorted(set(cat for r in all_results for cat in r["cat_scores"]))
    header = f"  {'Category':<15}" + "".join(f"{r['name'][:20]:>22}" for r in all_results)
    print(header, flush=True)
    for cat in all_cats:
        row = f"  {cat:<15}"
        for r in all_results:
            s = r["cat_scores"].get(cat, {"pass": 0, "total": 0})
            pct = f"{s['pass']}/{s['total']}"
            row += f"{pct:>22}"
        print(row, flush=True)

    out_path = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne/ab_test_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
