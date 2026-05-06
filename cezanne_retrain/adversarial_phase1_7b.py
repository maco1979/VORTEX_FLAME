#!/usr/bin/env python3
"""Phase 1: 7B answers questions → save results
Run separately from Phase 2 to avoid GPU memory issues
"""
import os, json, gc, time, sys
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

BASE_7B = "/mnt/d/models/Mistral-7B-Instruct-v0.1"
LORA_7B_S3 = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/s3_cs_depth_8k/final"
LOG_DIR = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

QUESTIONS_PATH = os.path.join(LOG_DIR, "adversarial_questions_v2.json")
ANSWERS_PATH = os.path.join(LOG_DIR, "adversarial_7b_answers_v2.json")

QUESTIONS = [
    {"cat": "Debug", "q": "Find the bug: def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1", "kw": ["off-by-one", "mid + 1", "infinite", "boundary"]},
    {"cat": "Debug", "q": "Find the bug: class Counter: count=0; def increment(self): self.count+=1. Two threads calling increment() simultaneously.", "kw": ["race", "thread", "lock", "atomic"]},
    {"cat": "Debug", "q": "Find the bug: def reverse_list(head): prev=None; curr=head; while curr: curr.next=prev; prev=curr; curr=curr.next; return prev", "kw": ["next", "temp", "save", "lost"]},
    {"cat": "Debug", "q": "Find the bug: for i in range(len(list)): if list[i] == target: list.remove(target)", "kw": ["skip", "index", "shift", "modify"]},
    {"cat": "Debug", "q": "Find the bug: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)", "kw": ["base case", "infinite", "recursion", "n<=1"]},
    {"cat": "Debug", "q": "Find the bug: def merge(a, b): result=[]; i=j=0; while i<len(a) or j<len(b): if a[i]<b[j]: result.append(a[i]); i+=1; else: result.append(b[j]); j+=1; return result", "kw": ["index", "bounds", "exhausted", "check"]},
    {"cat": "Debug", "q": "Find the bug: d = {}; d['key'] += 1", "kw": ["keyerror", "exist", "defaultdict", "get"]},
    {"cat": "Debug", "q": "Find the bug: try: result = 10/0; except: pass; print(result)", "kw": ["nameerror", "undefined", "silent", "variable"]},
    {"cat": "Debug", "q": "Find the bug: x = [1,2,3]; y = x; y.append(4); print(x)", "kw": ["reference", "copy", "same", "pointer"]},
    {"cat": "Debug", "q": "Find the bug: def average(lst): return sum(lst) / len(lst)", "kw": ["zero", "empty", "division", "guard"]},
    {"cat": "Logic", "q": "What is a contrapositive? Give an example.", "kw": ["contrapositive", "not", "implies", "equivalent"]},
    {"cat": "Logic", "q": "Explain proof by contradiction with an example.", "kw": ["contradiction", "assume", "false", "absurd"]},
    {"cat": "Logic", "q": "Prove using formal logic: if P implies Q and Q implies R, then P implies R.", "kw": ["syllogism", "implies", "transitivity", "modus"]},
    {"cat": "Logic", "q": "What is De Morgan's Law? Give an example.", "kw": ["de morgan", "not", "and", "or"]},
    {"cat": "Logic", "q": "Explain the difference between necessary and sufficient conditions.", "kw": ["necessary", "sufficient", "condition", "implies"]},
    {"cat": "Logic", "q": "Is the argument 'If it rains, the ground is wet. The ground is wet. Therefore it rained.' valid? Explain why or why not.", "kw": ["fallacy", "affirming", "consequent", "invalid"]},
    {"cat": "Logic", "q": "What is the difference between deductive and inductive reasoning?", "kw": ["deductive", "inductive", "certain", "probable"]},
    {"cat": "Logic", "q": "Prove that the square root of 2 is irrational using proof by contradiction.", "kw": ["irrational", "contradiction", "assume", "even"]},
    {"cat": "Systems", "q": "Explain CPU cache and why it matters for performance.", "kw": ["cache", "l1", "l2", "locality"]},
    {"cat": "Systems", "q": "Explain virtual memory and how page tables work.", "kw": ["virtual", "page", "tlb", "translation"]},
    {"cat": "Systems", "q": "Explain how malloc() works internally.", "kw": ["malloc", "heap", "free", "chunk"]},
    {"cat": "Systems", "q": "What is a deadlock? Explain prevention.", "kw": ["deadlock", "mutex", "prevention", "circular"]},
    {"cat": "Systems", "q": "Explain the difference between a process and a thread.", "kw": ["process", "thread", "memory", "address"]},
    {"cat": "Systems", "q": "What is memory-mapped I/O and how does it work?", "kw": ["mmap", "memory", "map", "file"]},
    {"cat": "Systems", "q": "Explain the difference between user space and kernel space.", "kw": ["user", "kernel", "space", "privilege"]},
    {"cat": "Complexity", "q": "Explain the difference between O(n) and O(n log n).", "kw": ["linear", "log", "complexity"]},
    {"cat": "Complexity", "q": "Explain amortized analysis with an example.", "kw": ["amortized", "average", "analysis"]},
    {"cat": "Complexity", "q": "What is the space complexity of quicksort?", "kw": ["log", "space", "quicksort", "recursion"]},
    {"cat": "Complexity", "q": "Compare O(n^2) quadratic time vs O(n log n) linearithmic time.", "kw": ["quadratic", "linearithmic", "n log", "n^2"]},
    {"cat": "Complexity", "q": "What does O(log n) logarithmic time mean? Give an example.", "kw": ["logarithmic", "binary", "half", "search"]},
    {"cat": "Algorithm", "q": "Write Dijkstra's shortest path algorithm in Python.", "kw": ["dijkstra", "priority", "queue", "distance"]},
    {"cat": "Algorithm", "q": "Write quicksort in Python with Lomuto partition.", "kw": ["quicksort", "partition", "pivot"]},
    {"cat": "Algorithm", "q": "Implement BFS for a graph represented as adjacency list.", "kw": ["bfs", "queue", "visited", "adjacency"]},
    {"cat": "Algorithm", "q": "Design an algorithm to find the kth smallest element in O(n) average time.", "kw": ["quickselect", "partition", "kth", "median"]},
    {"cat": "Algorithm", "q": "Explain the difference between DFS and BFS. When would you use each?", "kw": ["dfs", "bfs", "stack", "queue"]},
    {"cat": "Network", "q": "Explain the difference between TCP and UDP.", "kw": ["tcp", "udp", "reliable", "connection"]},
    {"cat": "Network", "q": "What is SQL injection and how to prevent it?", "kw": ["sql", "injection", "parameterized", "prepared"]},
    {"cat": "Network", "q": "Explain TCP congestion control and slow start.", "kw": ["congestion", "slow start", "window", "tcp"]},
    {"cat": "Network", "q": "What is a REST API? Explain the main principles.", "kw": ["rest", "api", "stateless", "http"]},
    {"cat": "Network", "q": "Explain DNS and how domain name resolution works.", "kw": ["dns", "domain", "resolution", "ip"]},
    {"cat": "Math", "q": "What is the modular inverse of 3 mod 7? Show the calculation.", "kw": ["modular", "inverse", "5"]},
    {"cat": "Math", "q": "Calculate the number of edges in a complete graph K_n.", "kw": ["complete", "graph", "edge", "n(n-1)/2"]},
    {"cat": "Math", "q": "What is the probability of getting exactly 3 heads in 5 coin flips?", "kw": ["probability", "binomial", "10", "0.5"]},
    {"cat": "Math", "q": "Explain the Master Theorem for recurrence relations.", "kw": ["master", "theorem", "recurrence", "t(n)"]},
]


def main():
    print(f"PHASE 1: 7B answers {len(QUESTIONS)} questions", flush=True)

    with open(QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(QUESTIONS, f, ensure_ascii=False, indent=2)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_7B, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_7B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, LORA_7B_S3)
    model.eval()
    print(f"7B loaded", flush=True)

    results = []
    for i, q in enumerate(QUESTIONS):
        prompt = f"<s>[INST] {q['q']} [/INST] "
        inp = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=300, temperature=0.7,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        ans = ans.replace(prompt, "").strip()

        matched = [k for k in q["kw"] if k.lower() in ans.lower()]
        passed = len(matched) >= 2
        q["answer_7b"] = ans
        q["7b_matched"] = matched
        q["7b_pass"] = passed
        status = "PASS" if passed else "FAIL"
        print(f"  [{i+1}/{len(QUESTIONS)}] [{q['cat']}] [{status}] matched={matched}", flush=True)
        results.append(q.copy())

        if (i + 1) % 10 == 0:
            with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  [CHECKPOINT] Saved {len(results)} answers", flush=True)

    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = len(results)
    passed = sum(1 for r in results if r["7b_pass"])
    print(f"\n7B Phase 1 done: {passed}/{total} passed ({passed/total:.0%})", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU cleared", flush=True)


if __name__ == "__main__":
    main()
