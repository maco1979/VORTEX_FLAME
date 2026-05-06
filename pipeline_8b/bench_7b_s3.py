#!/usr/bin/env python3
"""7B Cezanne S1 → S2 → S3 benchmark with all three stages"""
import os, json, gc, time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TQDM_DISABLE"] = "1"

BASE = r"E:\models\Mistral-7B-Instruct-v0.1-Cezanne"
S1 = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage1\final"
S2 = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage2\final"
S3 = r"D:\VORTEX_FLAME\soul_lora_v2\cezanne\stage3b\final"

Q = [
    {"cat":"sort","q":"Write quicksort in Python with Lomuto partition. Explain time complexity.","kw":["quicksort","partition","pivot","O(n log n)","worst"]},
    {"cat":"tree","q":"Implement BST with insert and search in Python.","kw":["class","Node","insert","search","left","right"]},
    {"cat":"graph","q":"Write Dijkstra algorithm in Python. Explain why priority queue is needed.","kw":["dijkstra","priority","queue","distance","shortest"]},
    {"cat":"dp","q":"Solve 0/1 knapsack with DP. Show the table construction.","kw":["knapsack","DP","table","weight","value"]},
    {"cat":"logic","q":"Prove by contradiction: there is no largest prime number. Label each step.","kw":["contradiction","prime","assume","infinite"]},
    {"cat":"logic","q":"Prove using formal logic: if P implies Q and Q implies R, then P implies R.","kw":["syllogism","implies","transitivity","modus"]},
    {"cat":"math","q":"What is the physical meaning of a derivative? From displacement to velocity to acceleration.","kw":["derivative","velocity","acceleration","rate","limit"]},
    {"cat":"math","q":"Prove: if n is even, then n squared is even. Label each derivation step.","kw":["even","2k","derivation","definition"]},
    {"cat":"debug","q":"Find the bug: def binary_search(arr, target): left,right=0,len(arr); while left<right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid; else: right=mid; return -1","kw":["off-by-one","mid + 1","mid - 1","infinite","boundary"]},
    {"cat":"security","q":"Explain SQL injection. How to prevent it using parameterized queries?","kw":["SQL","injection","parameterized","prepared","input"]},
]

def bench(lora_path, label):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  LoRA: {lora_path}")
    print(f"{'='*50}")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    t = AutoTokenizer.from_pretrained(BASE)
    if t.pad_token is None: t.pad_token = t.eos_token
    m = PeftModel.from_pretrained(m, lora_path)
    m.eval()
    results = []
    for i, item in enumerate(Q):
        prompt = f"<s>[INST] {item['q']} [/INST] "
        inp = t(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=500, temperature=0.7, do_sample=True)
        ans = t.decode(out[0], skip_special_tokens=True).replace(prompt.replace("<s>",""), "").strip()
        matched = [k for k in item["kw"] if k.lower() in ans.lower()]
        score = len(matched)/len(item["kw"])
        passed = score >= 0.3 or len(ans) > 200
        results.append({"cat":item["cat"],"score":score,"passed":passed,"matched":matched,"len":len(ans)})
        status = "PASS" if passed else "FAIL"
        print(f"  Q{i+1} [{item['cat']:8s}] [{status}] kw={score:.0%} len={len(ans)} {matched}")
    avg = sum(r["score"] for r in results)/len(results)
    pct = sum(1 for r in results if r["passed"])
    print(f"\n  {label}: {pct}/10 | Avg kw: {avg:.1%}")
    del m, t
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    return {"label":label,"avg":avg,"pass":pct,"results":results}

# S1 (from existing result)
s1 = json.load(open(r"D:\VORTEX_FLAME\hermes_logs\cezanne\stage1_stage2_benchmark.json","r"))["stage1"]

# S2 (from existing result)
s2 = json.load(open(r"D:\VORTEX_FLAME\hermes_logs\cezanne\stage1_stage2_benchmark.json","r"))["stage2"]

# S3 (run fresh)
s3 = bench(S3, "Stage3 (Math+Logic+Code+Proof)")

# Compare
print(f"\n{'='*50}")
print(f"  7B S1 vs S2 vs S3 TREND")
print(f"{'='*50}")
print(f"  {'Category':10s} {'S1':>6s} {'S2':>6s} {'S3':>6s} {'趋势':>6s}")
print(f"  {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
for i, item in enumerate(Q):
    s1v = s1["results"][i]["score"]
    s2v = s2["results"][i]["score"]
    s3v = s3["results"][i]["score"]
    d = s3v - s2v
    trend = f"+{d:.0%}" if d > 0.05 else (f"{d:.0%}" if d < -0.05 else "=")
    print(f"  {item['cat']:10s} {s1v:5.0%}  {s2v:5.0%}  {s3v:5.0%}  [{trend}]")

print(f"  {'AVG':10s} {s1['avg_score']:5.0%}  {s2['avg_score']:5.0%}  {s3['avg']:5.0%}")

log_path = r"D:\VORTEX_FLAME\hermes_logs\cezanne\stage3_benchmark.json"
with open(log_path,"w",encoding="utf-8") as f:
    json.dump({"s1_avg":s1["avg_score"],"s2_avg":s2["avg_score"],"s3_avg":s3["avg"],"s3_results":s3["results"]},f,indent=2)
print(f"\nSaved: {log_path}")
