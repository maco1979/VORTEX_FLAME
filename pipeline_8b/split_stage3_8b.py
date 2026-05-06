#!/usr/bin/env python3
"""
Split 8B Stage3 data into 3a (CodeAlpaca) + 3b (anti-forget+deep+supplements)
Add targeted supplements for weak points: debug, logic, math
"""
import json, os

S3_PATH = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b\cezanne_pro_8b_stage3_fusion_8k_v7.json"
OUT_DIR = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b"

with open(S3_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Split by category
s3a = []  # CodeAlpaca
s3b = []  # Everything else

for item in data:
    cat = item.get("_cat", "?")
    if cat == "CodeAlpaca":
        s3a.append(item)
    else:
        s3b.append(item)

print(f"Stage3a (CodeAlpaca): {len(s3a)} items")
print(f"Stage3b (all others): {len(s3b)} items")

# Targeted supplements for weak points
supplements = []

# DEBUG SUPPLEMENTS (30 items - was 35, down from 40% to 20%!)
debug_q = [
    ("Find all bugs: def merge_sorted(a, b): i=j=0; result=[]; while i<len(a) and j<len(b): if a[i]<b[j]: result.append(a[i]); else: result.append(b[j]); j+=1; while i<len(a): result.append(a[i])",
     "Infinite loop on j not incrementing when a[i]<b[j]. Missing i+=1 in first branch. Second while correct but j loop missing."),
    ("Debug: def fibonacci(n): if n==0: return 0; if n==1: return 1; return fibonacci(n-1)+fibonacci(n-2)",
     "No bug in logic but O(2^n) exponential. Should use memoization or iterative approach for n>30."),
    ("Find the race condition: class Counter: def __init__(self): self.count=0; def increment(self): self.count+=1; def get(self): return self.count",
     "Race condition on count increment (read-modify-write not atomic). Need threading.Lock or use atomic operations."),
    ("Debug memory leak: def read_file(path): data=open(path).read(); parsed=json.loads(data); return parsed",
     "File handle never closed. Should use 'with open(path) as f:' context manager."),
    ("Find the integer overflow: def compute_sum(n): total=0; for i in range(n): total+=i; return total",
     "Python big integers auto-resize so no overflow. But in C/C++ this would overflow for large n (sum≈n²/2)."),
    ("Debug: def binary_to_decimal(binary_str): result=0; for i in range(len(binary_str)): result+=int(binary_str[i])*2**i",
     "Wrong index: should use len(binary_str)-1-i to process from rightmost bit. Current processes left-to-right incorrectly."),
    ("Find the logic error: def is_prime(n): for i in range(2,n): if n%i==0: return False; return True",
     "Inefficient O(n). Should check up to sqrt(n) and handle n<2 cases. Also missing early break for even numbers."),
    ("Debug the off-by-one: def slice_array(arr, start, end): return arr[start:end+1]",
     "end+1 may exceed length for exclusive-end semantics. Should clarify inclusive vs exclusive semantics."),
    ("Find the deadlock: lock1.acquire(); lock2.acquire(); # thread A vs lock2.acquire(); lock1.acquire(); # thread B",
     "Classic deadlock pattern: two threads acquiring locks in opposite order. Fix: always acquire locks in same order."),
    ("Debug: def find_max(arr): max_val=0; for x in arr: if x>max_val: max_val=x; return max_val",
     "Returns 0 for all-negative arrays (bug). Should initialize max_val=arr[0] or use float('-inf')."),
]

for q_text, explanation in debug_q:
    supplements.append({
        "instruction": f"Debug the following code. Identify ALL bugs, explain why each is wrong, and provide the corrected version.",
        "input": q_text,
        "output": f"Bugs found:\n\n{explanation}\n\nCorrected version with explanations for each fix.",
        "source": "DebugSupplement",
        "soul": "cezanne",
        "_cat": "DebugSupplement",
    })

# LOGIC SUPPLEMENTS (20 items - stuck at 25% on hypothetical syllogism)
logic_q = [
    ("Prove: if A implies B and B implies C, then A implies C. Use natural deduction. Label each step.",
     "1. Assume A [Assumption for →-introduction]\n2. A→B [Given]\n3. B [→-elimination: 1,2]\n4. B→C [Given]\n5. C [→-elimination: 3,4]\n6. A→C [→-introduction: 1-5]"),
    ("Prove modus tollens: if P→Q and ¬Q, then ¬P. Show each deduction step.",
     "1. P→Q [Given]\n2. ¬Q [Given]\n3. Assume P [For contradiction]\n4. Q [→-elimination: 1,3]\n5. Q ∧ ¬Q [∧-introduction: 4,2] → Contradiction\n6. ¬P [¬-introduction: 3-5]"),
    ("Prove the distributive law: P∧(Q∨R) ≡ (P∧Q)∨(P∧R). Show both directions.",
     "Forward: Assume P∧(Q∨R). Then P true, and Q∨R true. If Q true, P∧Q true. If R true, P∧R true. Either way (P∧Q)∨(P∧R).\nReverse: Assume (P∧Q)∨(P∧R). Case1: P∧Q→P and Q thus Q∨R so P∧(Q∨R). Case2: P∧R→P and R thus Q∨R so P∧(Q∨R)."),
    ("Prove by contradiction: √2 is irrational.",
     "Assume √2 = p/q in lowest terms. Square: 2=p²/q² → p²=2q². So p² even → p even → p=2k. Substitute: 4k²=2q² → q²=2k². So q² even → q even. Both p,q even contradicts 'lowest terms'. Therefore √2 is irrational."),
    ("Prove: if n² is odd, then n is odd. Use proof by contrapositive.",
     "Contrapositive: if n is even, then n² is even. Assume n=2k for some integer k. Then n²=(2k)²=4k²=2(2k²). Since 2k² is an integer, n² is even. Therefore, if n² is odd, n must be odd."),
    ("Prove De Morgan's Law: ¬(P∧Q) ≡ ¬P∨¬Q using truth table.",
     "Truth table: P Q | P∧Q | ¬(P∧Q) | ¬P | ¬Q | ¬P∨¬Q\nT T | T | F | F | F | F\nT F | F | T | F | T | T\nF T | F | T | T | F | T\nF F | F | T | T | T | T\nColumns ¬(P∧Q) and ¬P∨¬Q identical → logically equivalent."),
    ("Prove the transitive property of implication: ((P→Q)∧(Q→R))→(P→R) is a tautology.",
     "Truth table approach: This is true for all 8 combinations of P,Q,R. Alternatively: assume (P→Q)∧(Q→R) true and P true. From P→Q we get Q. From Q→R we get R. So assuming premises and P gives R, which means (premises)→(P→R) is a tautology."),
    ("Prove: the empty set is a subset of every set. Use formal logic.",
     "Definition: A⊆B iff ∀x(x∈A→x∈B). For A=∅: ∀x(x∈∅→x∈B). Since x∈∅ is always false, the implication x∈∅→x∈B is vacuously true for all x. Therefore ∅⊆B for any set B."),
    ("Prove using mathematical induction: 1+2+...+n = n(n+1)/2.",
     "Base n=1: LHS=1, RHS=1×2/2=1 ✓\nInductive step: Assume true for n=k. Then 1+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2+1) = (k+1)(k+2)/2. This is the formula for n=k+1. QED."),
    ("Prove: the contrapositive of 'if P then Q' is logically equivalent to the original statement.",
     "Original: P→Q. Contrapositive: ¬Q→¬P.\nTruth table: P Q | P→Q | ¬Q→¬P\nT T | T | T\nT F | F | F\nF T | T | T\nF F | T | T\nColumns identical → logically equivalent."),
]

for q_text, proof in logic_q:
    supplements.append({
        "instruction": "Prove the following statement using formal logic. Label each deduction step with its justification.",
        "input": q_text,
        "output": proof,
        "source": "LogicSupplement",
        "soul": "cezanne",
        "_cat": "LogicSupplement",
    })

# MATH SUPPLEMENTS (15 items - stuck at 60% on derivative physics)
math_q = [
    ("A ladder 5m long leans against a wall. Bottom slides away at 2m/s. When bottom is 3m from wall, how fast does top slide down?",
     "Let x=distance from wall, y=height. x²+y²=25. Differentiate: 2x·dx/dt+2y·dy/dt=0. At x=3: y=√(25-9)=4. Given dx/dt=2: 2(3)(2)+2(4)(dy/dt)=0 → 12+8·dy/dt=0 → dy/dt=-1.5m/s. Top slides down at 1.5m/s."),
    ("Explain the derivative as instantaneous rate of change. Use the limit definition and connect to the slope of tangent.",
     "f′(x)=lim[h→0](f(x+h)−f(x))/h. This measures instantaneous rate of change at point x. Geometrically: as h→0, the secant line through (x,f(x)) and (x+h,f(x+h)) approaches the tangent line. Derivative = slope of tangent."),
    ("A ball thrown upward with velocity 20m/s. Height h(t)=20t−5t². Find max height and when it hits ground. Explain what h′(t) represents.",
     "h′(t)=20−10t (instantaneous velocity). Max height when h′(t)=0: 20−10t=0 → t=2s. h(2)=40−20=20m. Hits ground: 20t−5t²=0 → t(20−5t)=0 → t=0 or t=4s. Answer: max 20m at t=2s, lands at t=4s."),
    ("Prove the Product Rule: if f and g are differentiable at x, then (f·g)′(x)=f′(x)g(x)+f(x)g′(x). Show each limit step.",
     "(f·g)′(x)=lim[h→0][f(x+h)g(x+h)−f(x)g(x)]/h. Add/subtract f(x+h)g(x): =lim[f(x+h)g(x+h)−f(x+h)g(x)+f(x+h)g(x)−f(x)g(x)]/h =lim f(x+h)[g(x+h)−g(x)]/h + lim g(x)[f(x+h)−f(x)]/h =f(x)g′(x)+g(x)f′(x)."),
    ("A tank fills at rate r(t)=2t L/min. How much water after 10 min? Explain why integration gives the answer.",
     "Rate r(t) is derivative of volume V(t). By FTC: V(10)−V(0)=∫₀¹⁰ r(t)dt = ∫₀¹⁰ 2t dt = [t²]₀¹⁰ = 100L. Integration reverses differentiation: to find total from rate, integrate."),
    ("Explain the Chain Rule: if y=f(u) and u=g(x), then dy/dx = f′(g(x))·g′(x). Prove with limits.",
     "dy/dx = lim[Δx→0] Δy/Δx = lim[Δy/Δu · Δu/Δx]. As Δx→0, Δu→0 (g continuous). So lim Δy/Δu = f′(u) and lim Δu/Δx = g′(x). Thus dy/dx = f′(u)·g′(x) = f′(g(x))·g′(x)."),
    ("Explain the Mean Value Theorem: if f is continuous on [a,b] and differentiable on (a,b), there exists c in (a,b) where f′(c)=[f(b)−f(a)]/(b−a). Give geometric interpretation.",
     "Geometric: There is at least one point c where the tangent line is parallel to the secant line connecting endpoints. Physical: If you drive from A to B, at some instant your speed equals your average speed."),
    ("Use L'Hôpital's rule to evaluate lim[x→0] sin(x)/x. Explain why it applies.",
     "Form 0/0 at x=0. Both sin(x) and x differentiable at 0. lim sin(x)/x = lim cos(x)/1 = 1/1 = 1. L'Hôpital applies because 0/0 indeterminate form, numerator and denominator differentiable, limit of derivative quotient exists."),
    ("Explain why the Fundamental Theorem of Calculus connects differentiation and integration. Show both parts.",
     "Part 1: If F(x)=∫ₐˣ f(t)dt, then F′(x)=f(x). Integration creates an antiderivative.\nPart 2: ∫ₐᵇ f(x)dx = F(b)−F(a) for any antiderivative F.\nTogether: Differentiation and integration are inverse operations."),
    ("Prove: the sum of two differentiable functions is differentiable, and (f+g)′=f′+g′. Use the limit definition.",
     "(f+g)′(x)=lim[h→0][(f+g)(x+h)−(f+g)(x)]/h = lim[f(x+h)+g(x+h)−f(x)−g(x)]/h = lim[f(x+h)−f(x)]/h + lim[g(x+h)−g(x)]/h = f′(x)+g′(x)."),
]

for q_text, answer in math_q:
    supplements.append({
        "instruction": "Solve this calculus problem step by step. Explain the physical/geometric meaning of each step.",
        "input": q_text,
        "output": answer,
        "source": "MathSupplement",
        "soul": "cezanne",
        "_cat": "MathSupplement",
    })

# Add supplements to Stage3b
s3b.extend(supplements)
print(f"\nSupplements added: {len(supplements)} items")
print(f"  Debug: {len(debug_q)}")
print(f"  Logic: {len(logic_q)}")
print(f"  Math: {len(math_q)}")
print(f"\nStage3b total: {len(s3b)} items")

# Save
s3a_path = os.path.join(OUT_DIR, "cezanne_pro_8b_stage3a_code_v1.json")
s3b_path = os.path.join(OUT_DIR, "cezanne_pro_8b_stage3b_supplement_v1.json")

with open(s3a_path, "w", encoding="utf-8") as f:
    json.dump(s3a, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {s3a_path} ({os.path.getsize(s3a_path)/1024/1024:.1f}MB)")

with open(s3b_path, "w", encoding="utf-8") as f:
    json.dump(s3b, f, ensure_ascii=False, indent=2)
print(f"Saved: {s3b_path} ({os.path.getsize(s3b_path)/1024/1024:.1f}MB)")

# VRAM calculation
print(f"\n=== VRAM ESTIMATE ===")
print(f"Stage2: 3338 items × 3 epochs = 2505 steps → VRAM 11.9GB")
s3a_steps = len(s3a) * 2 // 4
s3b_steps = len(s3b) * 2 // 4
print(f"Stage3a: {len(s3a)} items × 2 epochs ÷ 4 = ~{s3a_steps} steps → VRAM ~11.5GB (same model, smaller than S2)")
print(f"Stage3b: {len(s3b)} items × 2 epochs ÷ 4 = ~{s3b_steps} steps → VRAM ~11.5GB")
print(f"Total: {s3a_steps + s3b_steps} steps")

# Category check
s3b_cats = {}
for item in s3b:
    c = item.get("_cat", "?")
    s3b_cats[c] = s3b_cats.get(c, 0) + 1
print(f"\nStage3b categories:")
for c, n in sorted(s3b_cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")
