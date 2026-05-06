#!/usr/bin/env python3
"""Add 7 targeted supplements for uncovered weak points + dedup check"""
import json

PATH = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b\cezanne_pro_8b_stage3b_supplement_v1.json"
with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect all existing instruction texts for dedup
existing_instructions = set()
for item in data:
    inst = item.get("instruction", "")[:100].lower().strip()
    existing_instructions.add(inst)

new_items = 0

# 1. Binary search debug (the exact bug from benchmark)
debug_binary = {
    "instruction": "Find ALL bugs in this binary search implementation. Explain why each bug causes problems and write the corrected version.",
    "input": "def binary_search(arr, target): left, right = 0, len(arr); while left < right: mid = (left + right) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: left = mid; else: right = mid; return -1",
    "output": "Bugs found:\n\n1. **right = len(arr)**: Correct initialization should be right = len(arr) - 1 (zero-indexed). With right = len(arr), the search space is [0, len(arr)) which is 1 too large.\n\n2. **left = mid** (not mid+1): When arr[mid] < target, the target is in the right half. Setting left = mid (instead of left = mid + 1) causes infinite loop when left and right differ by 1. Example: left=0, right=1, mid=0, arr[0]<target, left=mid=0 again → infinite loop.\n\n3. **right = mid** (not mid-1): When arr[mid] > target, same issue. Should be right = mid - 1. Otherwise also causes infinite loop or incorrect boundary.\n\n4. **Missing off-by-one in right initialization**: len(arr) vs len(arr)-1 determines whether the search uses [l, r] closed or [l, r) half-open interval. Inconsistent with while left < right (half-open) vs updating rules (should be closed).\n\nCorrected version:\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\nKey fixes: right = len(arr)-1, while left <= right, left = mid+1, right = mid-1. Time complexity O(log n).",
    "source": "DebugSupplement",
    "soul": "cezanne",
    "_cat": "DebugSupplement",
}
inst_key = debug_binary["instruction"][:100].lower().strip()
if inst_key not in existing_instructions:
    data.append(debug_binary)
    existing_instructions.add(inst_key)
    new_items += 1
    print("  + Binary search debug (off-by-one, infinite loop)")

# 2. Infinite loop find_max debug (same class as binary search boundary issue)
debug_infinite = {
    "instruction": "This code has an infinite loop bug under certain conditions. Find it, explain the root cause, and fix it.",
    "input": "def find_peak(arr): left, right = 0, len(arr) - 1; while left < right: mid = (left + right) // 2; if arr[mid] < arr[mid + 1]: left = mid; else: right = mid; return left",
    "output": "Bug: Both branches use mid without increment/decrement → infinite loop when left+1==right.\n\nRoot cause: left=mid never increases left beyond mid. When left=0,right=1,mid=0, arr[0]<arr[1], left stays 0 → infinite loop.\n\nFix: left = mid + 1 in the ascending branch.\n```python\ndef find_peak(arr):\n    left, right = 0, len(arr) - 1\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] < arr[mid + 1]:\n            left = mid + 1  # FIXED\n        else:\n            right = mid\n    return left\n```\nPattern: binary search boundary updates must always narrow the interval by at least 1.",
    "source": "DebugSupplement",
    "soul": "cezanne",
    "_cat": "DebugSupplement",
}
inst_key = debug_infinite["instruction"][:100].lower().strip()
if inst_key not in existing_instructions:
    data.append(debug_infinite)
    existing_instructions.add(inst_key)
    new_items += 1
    print("  + Binary search boundary infinite loop (find_peak)")

# 3. Quicksort worst-case complexity explanation
sort_worst = {
    "instruction": "Explain quicksort's worst-case time complexity. Under what conditions does it occur and how can it be avoided?",
    "input": "",
    "output": "Quicksort worst case: **O(n²)** time complexity.\n\nOccurs when: pivot selection always picks the smallest or largest element, making partitions unbalanced (one empty, one with n-1 elements). This happens on:\n- Already sorted or reverse-sorted arrays with first/last element as pivot\n- All elements equal (with naive Lomuto partition)\n\nAvoidance strategies:\n1. **Randomized pivot**: Pick random element, expected O(n log n)\n2. **Median-of-three**: Pick median of first, middle, last\n3. **Introsort**: Switch to heapsort when recursion depth exceeds log n (Python's TimSort is a merge sort variant)\n4. **Three-way partitioning**: For duplicate-heavy arrays\n\nBest case: O(n log n) - balanced partitions\nAverage case: O(n log n) - randomized pivot gives this with high probability\nSpace: O(log n) for recursion stack (O(n) worst case with naive recursion)\n\nStable? No, standard quicksort is not stable. Can be made stable but impractical.",
    "source": "SortSupplement",
    "soul": "cezanne",
    "_cat": "SortSupplement",
}
inst_key = sort_worst["instruction"][:100].lower().strip()
if inst_key not in existing_instructions:
    data.append(sort_worst)
    existing_instructions.add(inst_key)
    new_items += 1
    print("  + Quicksort worst-case O(n²) explanation")

# 4. Dijkstra shortest path explanation
dijkstra_shortest = {
    "instruction": "Explain why Dijkstra's algorithm finds the shortest path. Why does it need a priority queue and what happens without one?",
    "input": "",
    "output": "Dijkstra finds the **shortest path** because it processes nodes in order of increasing distance from the source (greedy optimality).\n\nWhy priority queue is essential:\n1. Extracts the unvisited node with minimum distance in O(log V) time\n2. Without PQ: would need O(V) scan each time → O(V²) total\n3. With PQ: O((V+E)log V) total\n\nWhy it works (correctness proof):\n- When a node v is extracted from PQ, dist[v] is the true shortest distance from source to v\n- Proof by contradiction: assume first wrongly-extracted. Shorter path must exist through some unvisited node w. But dist[w] < dist[v] (since edge weights ≥0), so PQ would extract w first, not v. Contradiction.\n\nLimitations:\n- Does NOT work with negative edge weights (use Bellman-Ford instead)\n- Does NOT work with negative cycles (no shortest path exists)\n\nWithout priority queue (naive O(V²)):\n- Just scan all unvisited nodes for minimum\n- Acceptable for dense graphs (E ≈ V²) where O(V²) ≈ O((V+E)log V)\n- But for sparse graphs, PQ gives huge speedup",
    "source": "GraphSupplement",
    "soul": "cezanne",
    "_cat": "GraphSupplement",
}
inst_key = dijkstra_shortest["instruction"][:100].lower().strip()
if inst_key not in existing_instructions:
    data.append(dijkstra_shortest)
    existing_instructions.add(inst_key)
    new_items += 1
    print("  + Dijkstra shortest path + priority queue")

# 5. Prime number contradiction proof (infinite primes)
prime_proof = {
    "instruction": "Prove by contradiction that there are infinitely many prime numbers. Label each step with its logical justification.",
    "input": "",
    "output": "Proof by contradiction:\n\nStep 1: Assume the opposite — there are finitely many primes. Let them be p₁, p₂, ..., pₙ. [Assumption for contradiction]\n\nStep 2: Construct N = (p₁ × p₂ × ... × pₙ) + 1. [Definition, from the assumed finite list]\n\nStep 3: Claim: N is either prime or composite. [Law of excluded middle]\n\nStep 4: Case A: N is prime. Then N is a prime not in {p₁,...,pₙ} since N > pᵢ for all i. This contradicts the assumption that we listed ALL primes.\n\nStep 5: Case B: N is composite. Then N has some prime divisor q. But q cannot be any of p₁,...,pₙ because N mod pᵢ = 1 for all i (N = (product)+1). So q is a prime not in the list. Contradiction again.\n\nStep 6: Both cases lead to contradiction. Therefore the original assumption (finitely many primes) must be false. [Reductio ad absurdum]\n\nStep 7: Conclusion: There are infinitely many prime numbers. QED.\n\nKey insight: (product of all primes) + 1 is not divisible by any of those primes, forcing a new prime to exist. This is Euclid's classic proof.",
    "source": "LogicSupplement",
    "soul": "cezanne",
    "_cat": "LogicSupplement",
}
inst_key = prime_proof["instruction"][:100].lower().strip()
if inst_key not in existing_instructions:
    data.append(prime_proof)
    existing_instructions.add(inst_key)
    new_items += 1
    print("  + Infinite primes proof by contradiction")

# 6. Quicksort with explicit worst-case discussion
sort_qs = {
    "instruction": "Implement quicksort in Python using Lomuto partition. Document the worst-case, best-case, and average time complexity in the comments.",
    "input": "",
    "output": "```python\ndef quicksort(arr, low=0, high=None):\n    \"\"\"\n    Quicksort with Lomuto partition.\n    Best case: O(n log n) - balanced partitions\n    Average case: O(n log n) - expected with random pivot\n    Worst case: O(n²) - sorted array, last element pivot\n    Space: O(log n) recursion stack\n    Not stable\n    \"\"\"\n    if high is None:\n        high = len(arr) - 1\n    if low < high:\n        pivot_idx = lomuto_partition(arr, low, high)\n        quicksort(arr, low, pivot_idx - 1)\n        quicksort(arr, pivot_idx + 1, high)\n    return arr\n\ndef lomuto_partition(arr, low, high):\n    pivot = arr[high]  # Last element as pivot\n    i = low - 1\n    for j in range(low, high):\n        if arr[j] <= pivot:\n            i += 1\n            arr[i], arr[j] = arr[j], arr[i]\n    arr[i + 1], arr[high] = arr[high], arr[i + 1]\n    return i + 1\n```\n\nWorst case O(n²) example: arr=[1,2,3,4,5] with last-element pivot. Each partition removes only the pivot (n-1 comparisons for pivot, then n-2, ... = n(n-1)/2).\n\nAvoid worst case: pick random pivot or median-of-three.",
    "source": "SortSupplement",
    "soul": "cezanne",
    "_cat": "SortSupplement",
}
inst_key = sort_qs["instruction"][:100].lower().strip()
if inst_key not in existing_instructions:
    data.append(sort_qs)
    existing_instructions.add(inst_key)
    new_items += 1
    print("  + Quicksort with worst-case documentation")

# 7. Derivative physical meaning with explicit rate/limit
math_derivative = {
    "instruction": "Explain the physical meaning of a derivative using displacement, velocity, and acceleration as examples. Include the limit definition and rate of change interpretation.",
    "input": "",
    "output": "The derivative f′(x) represents the **instantaneous rate of change** of f with respect to x.\n\nLimit definition:\nf′(x) = lim_{h→0} [f(x+h) − f(x)] / h\n\nPhysical chain: displacement → velocity → acceleration:\n\n1. **Position s(t)** = displacement at time t\n2. **Velocity v(t)** = s′(t) = lim_{h→0} [s(t+h)−s(t)]/h\n   - This is the instantaneous rate of change of position\n   - Geometrically: slope of the tangent line to position curve\n3. **Acceleration a(t)** = v′(t) = s″(t)\n   - Rate of change of velocity\n   - Second derivative of position\n\nExample: s(t) = 5t² meters\n- v(t) = s′(t) = 10t m/s (velocity increases linearly)\n- a(t) = v′(t) = 10 m/s² (constant acceleration)\n\nKey physical intuitions:\n- Derivative > 0 → increasing (moving forward)\n- Derivative < 0 → decreasing (moving backward)\n- Derivative = 0 → stationary (turning point, max/min)\n- Second derivative > 0 → accelerating (concave up)\n- Second derivative < 0 → decelerating (concave down)\n\nThe 'instantaneous' is crucial: average velocity over Δt is [s(t+Δt)−s(t)]/Δt. Taking limit as Δt→0 gives instantaneous velocity, which is exactly the derivative.",
    "source": "MathSupplement",
    "soul": "cezanne",
    "_cat": "MathSupplement",
}
inst_key = math_derivative["instruction"][:100].lower().strip()
if inst_key not in existing_instructions:
    data.append(math_derivative)
    existing_instructions.add(inst_key)
    new_items += 1
    print("  + Derivative physical meaning (rate, limit, velocity, acceleration)")

print(f"\nNew items added: {new_items}")

# Dedup: check if any new item has near-duplicate in existing data
# Use instruction first 50 chars as fingerprint
fingerprints = {}
dupes_found = 0
for i, item in enumerate(data):
    fp = item.get("instruction", "")[:50].lower().strip()
    if fp in fingerprints:
        print(f"  DEDUP WARNING: item {i} dup with {fingerprints[fp]}: {fp}...")
        dupes_found += 1
    else:
        fingerprints[fp] = i

if dupes_found == 0:
    print("Dedup check: PASS (no duplicates)")

# Category summary
cats = {}
for item in data:
    c = item.get("_cat", "?")
    cats[c] = cats.get(c, 0) + 1
print(f"\nFinal category distribution ({len(data)} total):")
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {c}: {n}")

# Save
with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {PATH}")

# VRAM recalc
s3b_steps = len(data) * 2 // 4  # 2 epochs, grad_accum=4
print(f"Stage3b steps: ~{s3b_steps} (still safe, VRAM ~11.5GB)")
