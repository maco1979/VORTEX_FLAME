#!/usr/bin/env python3
"""7B vs 8B Adversarial Self-Play Learning v2
Simpler approach: 8B answers questions, 8B's answers become training data for 7B
Phase 1: 7B answers diagnostic questions (find weaknesses)
Phase 2: 8B answers same questions (provides correct answers)
Phase 3: Generate training data from 8B's correct answers on 7B's weak spots
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
BASE_8B = "/mnt/d/models/Ministral-8B-Reasoning-Text-34L-ctx1024"
LORA_7B_S3 = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne/s3_cs_depth_8k/final"
LORA_8B_S3B = "/mnt/d/VORTEX_FLAME/soul_lora_v2/cezanne_8b/stage3b/final"
LOG_DIR = "/mnt/d/VORTEX_FLAME/hermes_logs/cezanne"
DATA_DIR = "/mnt/d/VORTEX_FLAME/soul_training_data/cezanne"

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


def load_model(base_path, lora_path, label):
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"  Loading {label}...", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )

    config_path = os.path.join(base_path, "config.json")
    is_mistral3 = False
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        if cfg.get("model_type") == "mistral3" or "Mistral3" in str(cfg.get("architectures", [])):
            is_mistral3 = True

    if is_mistral3:
        from transformers import Mistral3ForConditionalGeneration
        model = Mistral3ForConditionalGeneration.from_pretrained(
            base_path, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_path, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16
        )

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"  {label} LoRA loaded", flush=True)
    else:
        print(f"  {label} base only", flush=True)
    model.eval()
    return model, tokenizer


def generate_7b(model, tokenizer, question, max_tokens=300):
    prompt = f"<s>[INST] {question} [/INST] "
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    return ans.replace(prompt, "").strip()


def generate_8b(model, tokenizer, question, max_tokens=400):
    prompt = f"[INST] {question} [/INST] "
    inp = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_tokens, temperature=0.7,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    return ans.replace(prompt, "").strip()


def main():
    print(f"\n{'#'*60}", flush=True)
    print(f"  7B vs 8B ADVERSARIAL SELF-PLAY v2", flush=True)
    print(f"  7B answers → 8B answers → 8B corrections = training data", flush=True)
    print(f"  Questions: {len(QUESTIONS)}", flush=True)
    print(f"{'#'*60}", flush=True)

    t0 = time.time()

    # Phase 1: 7B answers
    print(f"\n  PHASE 1: 7B answers questions", flush=True)
    model_7b, tok_7b = load_model(BASE_7B, LORA_7B_S3, "7B Cezanne S3")

    results = []
    for i, q in enumerate(QUESTIONS):
        print(f"  [{i+1}/{len(QUESTIONS)}] [{q['cat']}] 7B answering...", flush=True)
        ans_7b = generate_7b(model_7b, tok_7b, q["q"])
        matched = [k for k in q["kw"] if k.lower() in ans_7b.lower()]
        passed = len(matched) >= 2
        q["answer_7b"] = ans_7b
        q["7b_matched"] = matched
        q["7b_pass"] = passed
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] matched={matched}", flush=True)
        results.append(q.copy())

    del model_7b, tok_7b
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(30)
    print(f"  GPU cleared after 7B, VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB", flush=True)

    # Phase 2: 8B answers
    print(f"\n  PHASE 2: 8B answers questions", flush=True)
    model_8b, tok_8b = load_model(BASE_8B, LORA_8B_S3B, "8B Cezanne S3b")

    for i, q in enumerate(results):
        print(f"  [{i+1}/{len(results)}] [{q['cat']}] 8B answering...", flush=True)
        ans_8b = generate_8b(model_8b, tok_8b, q["q"])
        matched = [k for k in q["kw"] if k.lower() in ans_8b.lower()]
        passed = len(matched) >= 2
        q["answer_8b"] = ans_8b
        q["8b_matched"] = matched
        q["8b_pass"] = passed
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] matched={matched}", flush=True)

    del model_8b, tok_8b
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 3: Generate training data
    print(f"\n  PHASE 3: Generate training data", flush=True)
    training_data = []
    stats = {"7b_only_fail": 0, "both_pass": 0, "both_fail": 0, "8b_only_pass": 0}

    for q in results:
        s7 = q["7b_pass"]
        s8 = q["8b_pass"]

        if not s7 and s8:
            stats["8b_only_pass"] += 1
        elif s7 and not s8:
            stats["7b_only_fail"] += 1
        elif s7 and s8:
            stats["both_pass"] += 1
        else:
            stats["both_fail"] += 1

        if not s7 and q.get("answer_8b") and len(q["answer_8b"]) > 30:
            item = {
                "instruction": q["q"],
                "input": "",
                "output": q["answer_8b"],
                "source": f"adversarial_{q['cat']}",
                "soul": "cezanne",
                "_cat": q["cat"],
                "_7b_pass": s7,
                "_8b_pass": s8,
            }
            training_data.append(item)

        if not s8 and q.get("answer_7b") and len(q["answer_7b"]) > 30:
            pass

    cat_counts = {}
    for item in training_data:
        cat = item["_cat"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\n  Stats: {stats}", flush=True)
    print(f"  Training items (7B fail + 8B answer): {len(training_data)}", flush=True)
    print(f"  By category: {cat_counts}", flush=True)

    out_path = os.path.join(DATA_DIR, "cezanne_adversarial_training.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_path}", flush=True)

    # Save full results
    r_path = os.path.join(LOG_DIR, "adversarial_v2_results.json")
    with open(r_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "training_items": len(training_data),
                    "by_category": cat_counts, "results": results,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                   f, ensure_ascii=False, indent=2)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'#'*60}", flush=True)
    print(f"  ADVERSARIAL SELF-PLAY v2 COMPLETE", flush=True)
    print(f"  Time: {elapsed:.1f} minutes", flush=True)
    print(f"  7B fail + 8B pass: {stats['8b_only_pass']}", flush=True)
    print(f"  Training data: {len(training_data)} items", flush=True)
    print(f"  Categories: {cat_counts}", flush=True)
    print(f"{'#'*60}", flush=True)


if __name__ == "__main__":
    main()
