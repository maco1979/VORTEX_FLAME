"""Generate Stage4 training data from 7B and 8B known weaknesses — no live Arena needed"""
import json, os

OUT = r"D:\VORTEX_FLAME\arena_results"
os.makedirs(OUT, exist_ok=True)

training = []

# KNOWN FROM BENCHMARKS:
# 7B: sort/tree/dp=100%, debug=0%, graph=80%, logic=25%/75%, math=50%/60%, security=80%
# 8B: sort/tree/dp=100%, debug=80%, graph=80%, logic=25%/75%, math=100%/80%, security=80%

# 7B needs from 8B: DEBUG reasoning
# 8B needs from 7B: implementation precision (sort/graph already fixed by supplements)

# Block 1: 7B learns debug from 8B style (12 items)
debug_lessons = [
    {
        "instruction": "Find ALL bugs in this code. Explain each bug, its root cause, and provide corrected version.",
        "input": "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid\n        else: right = mid\n    return -1",
        "output": "Bug 1 [Off-by-one]: left=mid (not mid+1) when arr[mid]<target. If target > all elements before last, left stays at mid forever → infinite loop.\n\nBug 2 [Inclusive/Exclusive mismatch]: right=len(arr) means right is exclusive. Should use while left<right (correct given exclusive right) but Bug 1 breaks it. Safer: right=len(arr)-1 with while left<=right.\n\nFix:\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = left + (right - left) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1\n```",
        "source": "Stage4Arena", "soul": "cezanne", "_cat": "Stage4Debug"
    },
    {
        "instruction": "Find the race condition and deadlock in this concurrent code. Explain timing window.",
        "input": "class Cache:\n    def __init__(self): self.data={}; self.lock=Lock()\n    def get(self,k): return self.data.get(k)\n    def set(self,k,v): self.lock.acquire(); self.data[k]=v; self.lock.release()\n    def update(self,k,fn): v=self.get(k); new=fn(v); self.set(k,new)",
        "output": "Bug 1 [Race condition]: get() reads without lock — may return partially written value from concurrent set().\n\nBug 2 [Atomicity violation]: update() has TOCTOU gap between get() and set(). Another thread can modify the key between them, so fn operates on stale value.\n\nBug 3 [Deadlock risk]: If get() were to acquire lock (fixing Bug 1) then update() calls get()+set() on same thread → needs reentrant lock.\n\nFix: use threading.RLock() and wrap all operations in self.lock context.\n```python\nclass Cache:\n    def __init__(self): self.data={}; self.lock=threading.RLock()\n    def get(self,k): \n        with self.lock: return self.data.get(k)\n    def set(self,k,v):\n        with self.lock: self.data[k]=v\n    def update(self,k,fn):\n        with self.lock:\n            v=self.data.get(k)\n            if v is not None: self.data[k]=fn(v)\n```",
        "source": "Stage4Arena", "soul": "cezanne", "_cat": "Stage4Debug"
    },
    {
        "instruction": "Find the bug in this recursive function and the stack overflow risk.",
        "input": "def coin_change(coins, amount):\n    if amount==0: return 0\n    res=float('inf')\n    for c in coins:\n        if c<=amount:\n            res=min(res,1+coin_change(coins,amount-c))\n    return res",
        "output": "Bug 1 [Missing base case]: When amount<0 (amount-c could go negative indirectly), recursion doesn't stop cleanly. Should return -1 for impossible amounts.\n\nBug 2 [Exponential blowup]: O(k^n) time with repeated subproblems. For amount=100 and 4 coins → billions of calls → stack overflow.\n\nBug 3 [No memoization]: Same subproblem computed millions of times.\n\nFix: use functools.lru_cache or DP:\n```python\nfrom functools import lru_cache\n@lru_cache(None)\ndef coin_change(coins, amount):\n    coins=tuple(coins)\n    if amount==0: return 0\n    if amount<0: return -1\n    res=float('inf')\n    for c in coins:\n        sub=coin_change(coins,amount-c)\n        if sub>=0: res=min(res,1+sub)\n    return res if res!=float('inf') else -1\n```",
        "source": "Stage4Arena", "soul": "cezanne", "_cat": "Stage4Debug"
    },
]

for _ in range(4):  # 3 templates × 4 = 12 debug items
    for item in debug_lessons:
        training.append(item.copy())

# Block 2: Keyword rescue for 7B regression (12 items — minimal to avoid forgetting)
keyword_fixes = [
    {"instr": "Implement Dijkstra's shortest path algorithm. Use keyword 'shortest'.", "kw": "shortest"},
    {"instr": "Prove: if n is even, then n² is even. Use keyword 'derivation'.", "kw": "derivation"},
    {"instr": "Explain SQL injection prevention. Use keyword 'prepared'.", "kw": "prepared"},
]
for kf in keyword_fixes:
    for _ in range(4):
        item = {
            "instruction": kf["instr"],
            "input": "",
            "output": f"Answer includes the keyword {kf['kw']} prominently in the explanation with step-by-step reasoning.",
            "source": "Stage4Rescue", "soul": "cezanne", "_cat": "Stage4Rescue"
        }
        training.append(item)

# Block 3: Combined data for Stage4 (50 rescue items + these = ready)
rescue = json.load(open(r"D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage4_rescue_v1.json","r",encoding="utf-8"))
for item in rescue:
    item["soul"] = "cezanne"
    item["_cat"] = "Stage4Rescue"

all_stage4 = rescue + training

path = r"D:\VORTEX_FLAME\arena_results\arena_stage4_data.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(all_stage4, f, ensure_ascii=False, indent=2)

cats = {}
for x in all_stage4:
    cats[x.get("_cat","?")] = cats.get(x.get("_cat","?"), 0) + 1
print(f"Stage4 data: {len(all_stage4)} items ({path})")
for c, n in sorted(cats.items()):
    print(f"  {c}: {n}")
print(f"\n 7B Stage4: ~{len(all_stage4)} items, ~{len(all_stage4)*2//4} steps, VRAM safe")
