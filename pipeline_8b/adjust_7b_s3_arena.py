#!/usr/bin/env python3
"""
7B Stage3b data adjustment + Arena mutual improvement analysis
Based on: 7B S1=56% S2=73.5%, 8B S1=76% S2=74% S3=82%

Key findings:
  7B strengths: sort=100, graph=100, dp=100, tree=100, security=100
  7B weaknesses: debug=0% (CRITICAL), math_odd=75%, logic=25%
  8B S2 weaknesses (learned): debug=20%, sort=80%, graph=80%, logic=25%

Arena mutual improvement:
  Mode 2 (Attack-Defense): 8B debug skill + 7B precise implementation = mutual gain
  Mode 4 (Dual Soul): Einstein math + 7B Cezanne code > 8B alone
  Generated training data feeds BOTH models
"""
import json, os

# Load current 7B Stage3b
S3B_PATH = r"D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage3b_supplement_v1.json"
with open(S3B_PATH, "r", encoding="utf-8") as f:
    s3b = json.load(f)

print(f"Current 7B Stage3b: {len(s3b)} items")

# ==============
# EXTRA SUPPLEMENTS for 7B specific weaknesses
# ==============

extra = []

# DEBUG: 7B at 0% - need MORE than 8B's 12 items
# Add 8 additional debug items with varied patterns
debug_extra = [
    {
        "instruction": "Find ALL bugs in this code. Explain each bug, its root cause, and provide the corrected version with line-by-line explanation.",
        "input": "def palindrome_check(s): return s == s[::-1]; def longest_palindrome(s): n=len(s); max_len=1; start=0; for i in range(n): for j in range(i,n): if palindrome_check(s[i:j+1]) and j-i+1>max_len: max_len=j-i+1; start=i; return s[start:start+max_len]",
        "output": "Bugs:\n\n1. palindrome_check: works correctly for ASCII but fails for unicode with combining characters (e.g. 'cafe\\u0301'). Fix: use unicodedata.normalize first.\n\n2. longest_palindrome: O(n³) brute force. Works but extremely slow for n>200. Should use Manacher's algorithm O(n) or expand-around-center O(n²).\n\n3. substring slicing s[i:j+1] creates n² string copies → memory bloat. Should use index comparison instead.\n\n4. No edge case handling: empty string, single char, all same chars.\n\nCorrected version uses expand-around-center with indices instead of string slices.",
        "source": "DebugExtra", "soul": "cezanne", "_cat": "DebugSupplement",
    },
    {
        "instruction": "This function has a subtle logical error that causes incorrect results for certain inputs. Find the bug and fix it.",
        "input": "def majority_element(nums): candidate=None; count=0; for n in nums: if count==0: candidate=n; count+=1 if n==candidate else -1; return candidate",
        "output": "Bug: Boyer-Moore majority vote algorithm is correct but return value is wrong when no majority exists. The function always returns the last candidate even if it doesn't appear > n/2 times.\n\nFix: After finding candidate, verify count(candidate) > len(nums)//2 in a second pass. If not, return None or raise ValueError.\n\n```python\ndef majority_element(nums):\n    candidate = count = 0\n    for n in nums:\n        if count == 0: candidate = n\n        count += 1 if n == candidate else -1\n    if nums.count(candidate) > len(nums) // 2:\n        return candidate\n    return None\n```",
        "source": "DebugExtra", "soul": "cezanne", "_cat": "DebugSupplement",
    },
    {
        "instruction": "Find ALL bugs. This code aims to rotate a matrix 90 degrees clockwise in-place.",
        "input": "def rotate_matrix(matrix): n=len(matrix); for i in range(n): for j in range(i): matrix[i][j],matrix[j][i]=matrix[j][i],matrix[i][j]; for i in range(n): matrix[i].reverse()",
        "output": "Bugs:\n\n1. Inner loop: range(i) should be range(i, n) for transpose. As written, it transposes only half the matrix (below diagonal), missing the upper triangle elements.\n\n2. The transpose loop bounds are wrong but the reverse step is conceptually correct.\n\nCorrect version:\n```python\ndef rotate_matrix(matrix):\n    n = len(matrix)\n    for i in range(n):\n        for j in range(i, n):  # FIXED: i to n, not 0 to i\n            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]\n    for i in range(n):\n        matrix[i].reverse()\n```\n\n3. Edge case: n=0 or n=1 should be handled (n=1 case: transpose does nothing, reverse is fine).",
        "source": "DebugExtra", "soul": "cezanne", "_cat": "DebugSupplement",
    },
    {
        "instruction": "Debug the concurrent code. Find the race condition and the deadlock, explain the timing window.",
        "input": "class ThreadSafeCache:\n    def __init__(self): self.data={}; self.lock=threading.Lock()\n    def get(self,key): return self.data.get(key)\n    def set(self,key,value): self.lock.acquire(); self.data[key]=value; self.lock.release()\n    def update(self,key,fn): val=self.get(key); new=fn(val); self.set(key,new)",
        "output": "Bug 1 (Race condition): get() reads without lock → may read partially-written value during concurrent set(). Fix: acquire lock in get() too.\n\nBug 2 (Race in update): Between get() and set(), another thread may modify the key → fn operates on stale value. Fix: wrap entire update() in lock context.\n\nBug 3 (Deadlock risk): if get() acquires lock then update() calls get()+set() → reentrant lock needed or restructure.\n\nFixed version:\n```python\nclass ThreadSafeCache:\n    def __init__(self): self.data={}; self.lock=threading.RLock()\n    def get(self,key):\n        with self.lock: return self.data.get(key)\n    def set(self,key,value):\n        with self.lock: self.data[key]=value\n    def update(self,key,fn):\n        with self.lock:\n            val = self.data.get(key)\n            self.data[key] = fn(val) if val is not None else None\n```",
        "source": "DebugExtra", "soul": "cezanne", "_cat": "DebugSupplement",
    },
    {
        "instruction": "Find the off-by-one error and the overflow bug in this binary search variant.",
        "input": "def search_insert_position(arr, target): left,right=0,len(arr)-1; while left<=right: mid=(left+right)//2; if arr[mid]==target: return mid; elif arr[mid]<target: left=mid+1; else: right=mid-1; return left+1",
        "output": "Bug 1 (overflow in mid): (left+right)//2 can overflow in languages with fixed-width integers (not Python but bad practice). Use left+(right-left)//2.\n\nBug 2 (return value): returns left+1 which may be off by one. Should return left (the insertion index, not insertion index + 1). When target > all elements, left becomes len(arr) after loop → left is the correct insert position (end of array). left+1 would be out of bounds.\n\nCorrected:\n```python\ndef search_insert_position(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = left + (right - left) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return left  # FIXED: not left+1\n```",
        "source": "DebugExtra", "soul": "cezanne", "_cat": "DebugSupplement",
    },
    {
        "instruction": "This regex-based input validator has multiple security vulnerabilities. Find them all.",
        "input": "def validate_email(email): import re; return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$', email))",
        "output": "Bug 1 (ReDoS): The regex is vulnerable to ReDoS (Regular Expression Denial of Service) with crafted inputs like 'a'*1000+'@a.'+'a'*1000 due to backtracking in the nested quantifiers.\n\nBug 2 (No length limit): Accepts emails up to any length, enabling buffer overflow attacks in downstream systems.\n\nBug 3 (Unicode bypass): Only matches ASCII. Internationalized email addresses (RFC 6531) with unicode characters are rejected but some systems accept them, causing inconsistency.\n\nBug 4 (Domain validation insufficient): Accepts '..' in domain, domains starting/ending with '.', and domains like 'a.-b.com'.\n\nFix: Use email-validator library instead of regex. Add max length check (254 per RFC 5321).",
        "source": "DebugExtra", "soul": "cezanne", "_cat": "DebugSupplement",
    },
    {
        "instruction": "Find the silent data corruption bug in this file copy function.",
        "input": "def copy_file(src, dst): data=open(src,'rb').read(); open(dst,'wb').write(data)",
        "output": "Bug 1 (File not closed): open(src) and open(dst) leak file handles. With many files → 'too many open files' error.\n\nBug 2 (No error handling): If src doesn't exist → FileNotFoundError with no recovery. If dst path doesn't exist → FileNotFoundError.\n\nBug 3 (Partial write risk): write() may not write all data at once (returns bytes written). For large files, this silently loses data.\n\nBug 4 (Memory exhaustion): read() loads entire file into memory. For >1GB files on 8GB system → OOM.\n\nCorrected:\n```python\ndef copy_file(src, dst):\n    os.makedirs(os.path.dirname(dst), exist_ok=True)\n    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:\n        while chunk := fsrc.read(8192):\n            fdst.write(chunk)\n```",
        "source": "DebugExtra", "soul": "cezanne", "_cat": "DebugSupplement",
    },
    {
        "instruction": "Debug the recursive function: find the missing base case and the stack overflow risk.",
        "input": "def coin_change(coins, amount): if amount==0: return 0; res=float('inf'); for c in coins: if c<=amount: res=min(res,1+coin_change(coins,amount-c)); return res",
        "output": "Bugs:\n\n1. Missing base case: when amount < 0, returns float('inf') for all recursive calls. Should explicitly check and return -1 for impossible amounts.\n\n2. Exponential time: O(k^n) where k=len(coins), n=amount. For amount=100 and 4 coins → billions of calls → stack overflow.\n\n3. No memoization: recomputes same subproblems millions of times.\n\nFix: use DP or memoized recursion:\n```python\nfrom functools import lru_cache\n@lru_cache(None)\ndef coin_change(coins, amount):\n    coins = tuple(coins)\n    if amount == 0: return 0\n    if amount < 0: return -1\n    res = float('inf')\n    for c in coins:\n        sub = coin_change(coins, amount - c)\n        if sub >= 0:\n            res = min(res, 1 + sub)\n    return res if res != float('inf') else -1\n```",
        "source": "DebugExtra", "soul": "cezanne", "_cat": "DebugSupplement",
    },
]

# LOGIC: 7B same as 8B at 25% on formal logic - 2 additional proof items
logic_extra = [
    {
        "instruction": "Prove the following statement using formal logic with labeled deduction steps.",
        "input": "Prove: ((P→Q)∧(Q→R))→(P→R) is a tautology using truth table.",
        "output": "Truth table (8 rows for P,Q,R):\n\nP Q R | P→Q | Q→R | (P→Q)∧(Q→R) | P→R | ((P→Q)∧(Q→R))→(P→R)\nT T T | T   | T   | T            | T   | T\nT T F | T   | F   | F            | F   | T (F→F is True)\nT F T | F   | T   | F            | T   | T (F→T is True)\nT F F | F   | T   | F            | F   | T (F→F is True)\nF T T | T   | T   | T            | T   | T\nF T F | T   | F   | F            | T   | T (F→T is True)\nF F T | T   | T   | T            | T   | T\nF F F | T   | T   | T            | T   | T\n\nAll 8 rows yield True in the final column. Therefore ((P→Q)∧(Q→R))→(P→R) is a tautology (always true, regardless of P,Q,R values). This proves the transitivity of implication.",
        "source": "LogicExtra", "soul": "cezanne", "_cat": "LogicSupplement",
    },
    {
        "instruction": "Prove the statement using formal logic with each step labeled by the inference rule used.",
        "input": "Prove the logical equivalence: ¬(P∨Q) ≡ ¬P∧¬Q (one of De Morgan's Laws) using a truth table.",
        "output": "Truth table:\n\nP Q | P∨Q | ¬(P∨Q) | ¬P | ¬Q | ¬P∧¬Q | ¬(P∨Q)≡¬P∧¬Q\nT T | T   | F      | F  | F  | F     | T (both F)\nT F | T   | F      | F  | T  | F     | T (both F)\nF T | T   | F      | T  | F  | F     | T (both F)\nF F | F   | T      | T  | T  | T     | T (both T)\n\nThe columns for ¬(P∨Q) and ¬P∧¬Q are identical in all 4 rows. Therefore ¬(P∨Q) ≡ ¬P∧¬Q is logically equivalent. This is De Morgan's Law: 'not (P or Q) equals not P and not Q'.",
        "source": "LogicExtra", "soul": "cezanne", "_cat": "LogicSupplement",
    },
]

# MATH: 7B at 75% on even number proof (missing "derivation")
math_extra = [
    {
        "instruction": "Prove the statement step by step. Label each step with the definition or theorem used.",
        "input": "Prove: if n is odd, then n² is odd.",
        "output": "Step 1: Assume n is odd. [Premise]\nStep 2: By definition of odd: n = 2k + 1 for some integer k. [Definition of odd integer]\nStep 3: Square both sides: n² = (2k+1)². [Algebra, substitution from step 2]\nStep 4: Expand: n² = 4k² + 4k + 1 = 2(2k² + 2k) + 1. [Algebraic expansion and factoring]\nStep 5: Let m = 2k² + 2k. m is an integer (sum/product of integers). [Closure of integers under addition and multiplication]\nStep 6: Then n² = 2m + 1. [Substitution from step 5]\nStep 7: By definition, a number of the form 2m+1 is odd. [Definition of odd integer]\nStep 8: Therefore, if n is odd, n² is odd. QED. [Conditional proof: steps 1→7]",
        "source": "MathExtra", "soul": "cezanne", "_cat": "MathSupplement",
    },
    {
        "instruction": "Explain the physical meaning of each step. Derive from first principles using limits.",
        "input": "Explain why the derivative of position s(t) gives velocity v(t), and the second derivative gives acceleration a(t). Use the limit definition.",
        "output": "1. Position: s(t) = location at time t.\n\n2. Average velocity over interval Δt: v_avg = [s(t+Δt) - s(t)] / Δt\n   This is the slope of the secant line on the s-t graph.\n\n3. Instantaneous velocity (derivative): v(t) = s'(t) = lim[Δt→0] [s(t+Δt)-s(t)]/Δt\n   As Δt shrinks to zero, the secant becomes the tangent line. The slope of the tangent = instantaneous rate of change of position = velocity.\n\n4. Physical meaning: v(t) measures how fast s changes per unit time AT that exact instant, not averaged over an interval.\n\n5. Acceleration (second derivative): a(t) = v'(t) = s''(t) = lim[Δt→0] [v(t+Δt)-v(t)]/Δt\n   This is the instantaneous rate of change of velocity.\n\n6. Chain: s(t) → differentiate → v(t)=s'(t) → differentiate → a(t)=s''(t)\n\n7. Example: s(t)=5t²+2t. v(t)=s'(t)=10t+2 (velocity increases linearly). a(t)=v'(t)=10 (constant acceleration 10 m/s²). The 'rate' and 'limit' concepts are fundamental to connecting these three physical quantities.",
        "source": "MathExtra", "soul": "cezanne", "_cat": "MathSupplement",
    },
]

extra.extend(debug_extra)
extra.extend(logic_extra)
extra.extend(math_extra)

# Dedup against existing entries
existing_fps = set()
for item in s3b:
    fp = item.get("instruction", "")[:30].lower().strip() + "|||" + item.get("input", "")[:80].lower().strip()
    existing_fps.add(fp)

added = 0
for item in extra:
    fp = item.get("instruction", "")[:30].lower().strip() + "|||" + item.get("input", "")[:80].lower().strip()
    if fp not in existing_fps:
        s3b.append(item)
        existing_fps.add(fp)
        added += 1

print(f"Added {added} extra supplements (targeting 7B weaknesses)")
print(f"  Debug: {len(debug_extra)} items (7B at 0% → target 60%+)")
print(f"  Logic: {len(logic_extra)} items")
print(f"  Math: {len(math_extra)} items")

# Save
with open(S3B_PATH, "w", encoding="utf-8") as f:
    json.dump(s3b, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {S3B_PATH} ({len(s3b)} total items)")

# Category summary
cats = {}
for item in s3b:
    c = item.get("_cat", "?")
    cats[c] = cats.get(c, 0) + 1
print(f"\nStage3b final categories ({len(s3b)} total):")
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

# VRAM recalc
steps = len(s3b) * 2 // 4
print(f"\nStage3b steps: ~{steps} (VRAM ~7GB, safe)")

# ==============
# ARENA ANALYSIS
# ==============
print(f"\n{'='*60}")
print(f"  ARENA MUTUAL IMPROVEMENT ANALYSIS")
print(f"{'='*60}")
print(f"""
7B S2=73.5% vs 8B S2=74%  (pre-Stage3, both are equal!)

7B ADVANTAGES over 8B:
  sort: 100% vs 80%  (+20pp) — precise algorithm implementation
  graph: 100% vs 80% (+20pp) — complete Dijkstra including 'shortest'
  security: 100% vs 100% — tied

8B ADVANTAGES over 7B:
  debug: 20% vs 0% (+20pp) — has seen some debug patterns
  math even: 100% vs 75% (+25pp) — formal proof structure

MUTUAL WEAKNESS:
  logic假言: BOTH at 25% — needs Arena to break this

ARENA SCENARIOS:

  Mode 1 (Code Battle):
    7B wins on sort/graph/dp/security (implementation precision)
    8B wins on debug/math (reasoning depth)
    → 7B's correct code → training data for 8B's code quality
    → 8B's analytical approach → training data for 7B's reasoning

  Mode 2 (Attack-Defense) — HIGHEST VALUE:
    7B writes correct implementation → 8B tries to find edge case bugs
    → 8B finds bugs 7B missed → debug training data for 7B
    8B writes buggy implementation → 7B tries to fix
    → 7B fails (0% debug) → 8B's fix → multi-step debug data for 7B
    → This directly addresses 7B's CRITICAL 0% debug weakness

  Mode 3 (Reasoning Duel):
    Both at 75%/25% on logic → little difference
    → Both need formal proof training, not Arena

  Mode 4 (Dual Soul: Einstein + 7B vs 8B):
    Einstein analyses math → 7B codes solution
    → 7B's code precision (100%) + Einstein's math depth > 8B alone
    → Generates "analysis→code" bridge data useful for 8B too

CONCLUSION: Arena CAN iterate 8B through 7B
  - 7B's precise implementation is data 8B needs (sort/graph were 80%)
  - 8B's debug reasoning is data 7B desperately needs (0%!)
  - Mutual weakness (logic 25%) needs external data, not Arena
  - Estimated improvement: 7B S3→82-85%, 8B gets code quality boost from 7B
""")
