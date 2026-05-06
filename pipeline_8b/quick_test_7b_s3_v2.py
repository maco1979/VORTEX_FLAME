#!/usr/bin/env python3
import os, json, gc, time, sys, torch
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE = "/mnt/d/models/Mistral-7B-Instruct-v0.1"
LORA_S3 = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/stage3_cs/final"
LOG = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"

QUESTIONS = [
    {"cat": "debug", "q": "Find the bug: def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1",
     "kw": ["off-by-one", "mid + 1", "mid - 1", "infinite", "boundary"]},
    {"cat": "logic", "q": "Prove using formal logic: if P implies Q and Q implies R, then P implies R. This is hypothetical syllogism.",
     "kw": ["syllogism", "implies", "transitivity", "modus", "ponens"]},
    {"cat": "vm", "q": "Explain virtual memory and page tables. How does TLB speed up address translation?",
     "kw": ["virtual", "page", "TLB", "translation", "frame"]},
    {"cat": "malloc", "q": "Explain how malloc() works internally. What is the difference between malloc, calloc, and realloc?",
     "kw": ["malloc", "calloc", "realloc", "heap", "chunk"]},
    {"cat": "tcp", "q": "Explain TCP congestion control. What is slow start, congestion avoidance, and fast recovery?",
     "kw": ["congestion", "slow start", "window", "threshold", "avoidance"]},
    {"cat": "deadlock", "q": "Explain the concept of a deadlock. What are the four necessary conditions? How to prevent it?",
     "kw": ["deadlock", "mutual exclusion", "hold and wait", "circular", "preemption"]},
    {"cat": "compile", "q": "Explain the phases of a compiler: lexing, parsing, semantic analysis, code generation.",
     "kw": ["lexer", "parser", "semantic", "code generation", "token"]},
    {"cat": "cuda", "q": "Explain how CUDA programming works. What is the relationship between grids, blocks, and threads?",
     "kw": ["CUDA", "grid", "block", "thread", "warp"]},
    {"cat": "sort", "q": "Write quicksort in Python with Lomuto partition. Explain time complexity.",
     "kw": ["quicksort", "partition", "pivot", "O(n log n)", "worst"]},
    {"cat": "graph", "q": "Write Dijkstra algorithm in Python. Explain why priority queue is needed.",
     "kw": ["dijkstra", "priority", "queue", "distance", "shortest"]},
]

def test_lora(lora_path, label):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n  Loading {label}...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    results = []
    for i, item in enumerate(QUESTIONS):
        prompt = f"<s>[INST] {item['q']} [/INST] "
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=400, temperature=0.7,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(out[0], skip_special_tokens=True).replace(
            prompt.replace("<s>", ""), "").strip()
        matched = [k for k in item["kw"] if k.lower() in ans.lower()]
        score = len(matched) / len(item["kw"])
        passed = score >= 0.3 or len(ans) > 150
        results.append({"cat": item["cat"], "score": score, "passed": passed,
                        "matched": matched, "len": len(ans)})
        status = "PASS" if passed else "FAIL"
        print(f"  Q{i+1:2d} [{item['cat']:9s}] [{status}] kw={score:4.0%} len={len(ans):4d} {matched}", flush=True)

    avg = sum(r["score"] for r in results) / len(results)
    pct = sum(1 for r in results if r["passed"])
    print(f"\n  {label}: {pct}/{len(QUESTIONS)} PASS | Avg kw: {avg:.1%}", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    return {"label": label, "avg": avg, "pass": pct, "results": results}


def main():
    print(f"{'='*60}", flush=True)
    print(f"  7B S3_CS Quick Test (mem-optimized)", flush=True)
    print(f"{'='*60}", flush=True)

    r_s3 = test_lora(LORA_S3, "S3_CS")

    log_path = os.path.join(LOG, "s3_cs_quick_test.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(r_s3, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {log_path}", flush=True)


if __name__ == "__main__":
    main()
