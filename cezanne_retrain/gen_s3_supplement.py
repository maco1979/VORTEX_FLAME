#!/usr/bin/env python3
"""Generate S3 supplement data: Debug + Complexity + Anti-forgetting review"""
import json, os, random

random.seed(3407)

supplement = []

# ===== DEBUG SUPPLEMENT (20 items) =====
debug_items = [
    {
        "instruction": "Find the bug in this binary search: def binary_search(arr, target): left, right = 0, len(arr); while left < right: mid = (left + right) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: left = mid; else: right = mid; return -1",
        "output": "Bug: off-by-one error. When arr[mid] < target, setting left = mid causes an infinite loop when left = mid = right-1. Fix: left = mid + 1. Similarly, when arr[mid] > target, right should be set to mid (not mid-1) because right is exclusive. Corrected: elif arr[mid] < target: left = mid + 1. Also, the initial right should be len(arr) for exclusive bounds, which is correct. The key bug is left=mid should be left=mid+1."
    },
    {
        "instruction": "Find the bug: def reverse_list(head): prev = None; curr = head; while curr: curr.next = prev; prev = curr; curr = curr.next; return prev",
        "output": "Bug: curr.next is overwritten before curr is saved. When curr.next = prev executes, the original curr.next (next node) is lost. Then curr = curr.next sets curr to prev (which is None on first iteration). Fix: save next_node before modifying: while curr: next_node = curr.next; curr.next = prev; prev = curr; curr = next_node."
    },
    {
        "instruction": "Find the bug: for i in range(len(lst)): if lst[i] == target: lst.remove(target)",
        "output": "Bug: modifying a list while iterating over it with index causes elements to be skipped. When remove() shifts elements left, the next i skips the element that moved into position i. Fix options: (1) iterate backwards: for i in range(len(lst)-1, -1, -1); (2) create a new list: lst = [x for x in lst if x != target]; (3) use while loop: while target in lst: lst.remove(target)."
    },
    {
        "instruction": "Find the bug: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)",
        "output": "Bug: missing base case causes infinite recursion and RecursionError. Without a termination condition, the function calls itself forever with decreasing n, going into negative infinity. Fix: add base cases: if n <= 0: return 0; if n == 1: return 1; return fibonacci(n-1) + fibonacci(n-2). Also, this naive recursion has O(2^n) time complexity; use memoization or iteration for efficiency."
    },
    {
        "instruction": "Find the bug: class Counter: count = 0; def increment(self): self.count += 1",
        "output": "Bug: race condition in multi-threaded environment. self.count += 1 is not atomic — it reads, increments, and writes in separate steps. Two threads can read the same value before either writes. Fix: use threading.Lock: class Counter: def __init__(self): self.count = 0; self.lock = Lock(); def increment(self): with self.lock: self.count += 1. Or use atomic types from the multiprocessing module."
    },
    {
        "instruction": "Find the bug: def remove_duplicates(lst): for item in lst: if lst.count(item) > 1: lst.remove(item); return lst",
        "output": "Bug: modifying list while iterating causes skipped elements and incorrect results. Also, remove() only removes the first occurrence, so it may remove the wrong duplicate. Fix: return list(set(lst)) for unordered, or list(dict.fromkeys(lst)) for order-preserving deduplication."
    },
    {
        "instruction": "Find the bug: result = 0; for i in range(10): result += 1/i",
        "output": "Bug: division by zero when i=0. The loop starts at 0, and 1/0 raises ZeroDivisionError. Fix: start from 1: for i in range(1, 10), or add a check: if i != 0: result += 1/i."
    },
    {
        "instruction": "Find the bug: def is_sorted(arr): for i in range(len(arr)): if arr[i] > arr[i+1]: return False; return True",
        "output": "Bug: IndexError when i reaches the last element, because arr[i+1] is out of bounds. Fix: iterate to len(arr)-1: for i in range(len(arr)-1): if arr[i] > arr[i+1]: return False."
    },
    {
        "instruction": "Find the bug: def merge(a, b): result = []; i = j = 0; while i < len(a) or j < len(b): if a[i] < b[j]: result.append(a[i]); i += 1; else: result.append(b[j]); j += 1; return result",
        "output": "Bug: when one list is exhausted (i >= len(a) or j >= len(b)), accessing a[i] or b[j] causes IndexError. The condition uses 'or' so the loop continues even when one list is done. Fix: add bounds checking: while i < len(a) and j < len(b) for the comparison loop, then extend remaining elements: result.extend(a[i:]); result.extend(b[j:])."
    },
    {
        "instruction": "Find the bug: def insert_sorted(arr, val): arr.append(val); arr.sort(); return arr",
        "output": "Bug: this modifies the original list in place (append and sort are in-place operations). If the caller still holds a reference, they see unexpected changes. Fix: work on a copy: result = arr.copy(); result.append(val); result.sort(); return result. Or use bisect.insort for O(n) insertion into a sorted list."
    },
    {
        "instruction": "Find the bug: def copy_list(lst): return lst",
        "output": "Bug: this returns a reference to the same list, not a copy. Modifications to the returned list will affect the original. Fix: return lst.copy() or return lst[:] or return list(lst) for shallow copy. For nested structures, use copy.deepcopy(lst)."
    },
    {
        "instruction": "Find the bug: x = [1, 2, 3]; y = x; y.append(4); print(x)",
        "output": "Bug: y = x does not create a copy — y points to the same list object. When y.append(4) executes, x also becomes [1, 2, 3, 4] because x and y reference the same list. Fix: y = x.copy() or y = x[:] to create an independent copy."
    },
    {
        "instruction": "Find the bug: def average(lst): return sum(lst) / len(lst)",
        "output": "Bug: ZeroDivisionError when lst is empty. len([]) returns 0, causing division by zero. Fix: add a guard: if not lst: return 0 (or None, or raise a meaningful exception)."
    },
    {
        "instruction": "Find the bug: d = {}; d['key'] += 1",
        "output": "Bug: KeyError because 'key' does not exist in the dictionary yet. The += operator tries to read d['key'] first, which fails. Fix: use d['key'] = d.get('key', 0) + 1, or use collections.defaultdict(int): d = defaultdict(int); d['key'] += 1."
    },
    {
        "instruction": "Find the bug: def flatten(matrix): return [item for row in matrix for item in row]",
        "output": "Not strictly a bug for 2D lists, but if any row element is itself a list (3D+ nesting), it won't be fully flattened. For true deep flattening, use recursion: def flatten(lst): result = []; for item in lst: if isinstance(item, list): result.extend(flatten(item)); else: result.append(item); return result."
    },
    {
        "instruction": "Find the bug: try: result = 10 / 0; except: pass; print(result)",
        "output": "Bug: the exception is silently caught and ignored (bare except with pass), and then result is used without being defined, causing NameError. Fix: either handle the exception properly (set a default value in the except block), or let it propagate, or at minimum log the error instead of pass."
    },
    {
        "instruction": "Find the bug: def get_value(d, key): if d.has_key(key): return d[key]",
        "output": "Bug: dict.has_key() was removed in Python 3. This raises AttributeError. Fix: use 'if key in d:' instead. Also, the function returns None implicitly if the key is not found, which may be unexpected. Better: return d.get(key) or return d.get(key, default_value)."
    },
    {
        "instruction": "Find the bug: files = []; for f in os.listdir('.'): files.append(open(f).read())",
        "output": "Bug: file handles are never closed, causing resource leaks. Each open() allocates a file descriptor that remains open. Fix: use with statement: for f in os.listdir('.'): with open(f) as fh: files.append(fh.read()). Or use list comprehension with contextlib."
    },
    {
        "instruction": "Find the bug: def search(arr, target): i = 0; while i < len(arr): if arr[i] == target: return i; i += 1; return -1  # but called as: search((1,2,3), 2)",
        "output": "Bug: tuples are immutable but the function assumes it can work with any sequence. Actually this works fine for tuples. The real bug would be if someone tries to modify: arr[i] = new_val. But for search only, it works. A subtle issue: using while instead of for loop is unpythonic. Fix: for i, val in enumerate(arr): if val == target: return i."
    },
    {
        "instruction": "Find the bug: import threading; balance = 0; def deposit(amount): global balance; temp = balance; temp += amount; balance = temp; t1 = threading.Thread(target=deposit, args=(100,)); t2 = threading.Thread(target=deposit, args=(200,)); t1.start(); t2.start(); print(balance)",
        "output": "Bug: race condition between reading balance and writing it back. Thread t1 reads balance=0, t2 also reads balance=0 before t1 writes. Both compute temp=0+amount, so final balance may be 200 instead of 300. Fix: use Lock to make the read-modify-write atomic: lock = threading.Lock(); def deposit(amount): global balance; with lock: temp = balance; temp += amount; balance = temp."
    },
]

for item in debug_items:
    supplement.append({
        "instruction": item["instruction"],
        "input": "",
        "output": item["output"],
        "source": "debug_supplement",
        "soul": "cezanne",
        "_cat": "Debug",
    })

# ===== COMPLEXITY SUPPLEMENT (10 items) =====
complexity_items = [
    {
        "instruction": "Explain the difference between O(n) linear time and O(n log n) linearithmic time. Give an example algorithm for each.",
        "output": "O(n) linear time means the running time grows directly proportional to input size n. Example: linear search scans each element once. O(n log n) linearithmic time means the running time grows as n times log(n). Example: merge sort divides the input (log n levels) and merges at each level (n work per level). The key difference: O(n log n) is slightly slower than O(n) but much faster than O(n^2). For n=1 million, O(n) does 1M operations, O(n log n) does about 20M operations, while O(n^2) does 1 trillion."
    },
    {
        "instruction": "What does O(1) constant time mean? Give three examples of O(1) operations.",
        "output": "O(1) constant time means the operation takes the same amount of time regardless of input size. Examples: (1) Accessing an array element by index: arr[5] is O(1). (2) Inserting at the head of a linked list: just update the head pointer. (3) Hash table lookup: computing hash and accessing the bucket is O(1) on average. Contrast with O(n) linear time where you must scan all elements."
    },
    {
        "instruction": "Compare O(n^2) quadratic time vs O(n log n) linearithmic time. When does the difference matter?",
        "output": "O(n^2) quadratic time grows as the square of input size. O(n log n) grows as n times log n. The difference is enormous for large n. For n=10,000: O(n^2) = 100M operations, O(n log n) = ~130K operations — 770x faster. For n=1M: O(n^2) = 1 trillion, O(n log n) = 20M — 50,000x faster. The difference matters whenever n > ~100. Bubble sort (O(n^2)) vs merge sort (O(n log n)) is the classic example."
    },
    {
        "instruction": "Explain what O(log n) logarithmic time means with an example.",
        "output": "O(log n) logarithmic time means the running time grows as the logarithm of input size — each step cuts the problem size roughly in half. Example: binary search on a sorted array of 1 million elements takes at most log2(1,000,000) ≈ 20 comparisons. This is extremely efficient: doubling the input size only adds one more step. Other examples: balanced BST operations, finding an element in a skip list."
    },
    {
        "instruction": "What is the time complexity of accessing, searching, inserting, and deleting in: (1) array, (2) linked list, (3) hash table, (4) BST?",
        "output": "Array: access O(1), search O(n), insert O(n), delete O(n). Linked list: access O(n), search O(n), insert O(1) at head, delete O(n). Hash table: access N/A, search O(1) avg, insert O(1) avg, delete O(1) avg. BST (balanced): access O(log n), search O(log n), insert O(log n), delete O(log n). Key insight: hash tables give O(1) average but O(n) worst case; BSTs give O(log n) guaranteed for balanced trees."
    },
    {
        "instruction": "Explain the difference between best case, average case, and worst case time complexity using quicksort as an example.",
        "output": "Quicksort: Best case O(n log n) — pivot always divides array in half. Average case O(n log n) — random pivots give balanced partitions on average. Worst case O(n^2) — pivot is always min or max (e.g., already sorted array with first element as pivot). This is why quicksort uses random pivot selection or median-of-three to avoid worst case. In practice, quicksort is often faster than merge sort due to better cache locality, despite same average complexity."
    },
    {
        "instruction": "What is amortized analysis? Explain with the dynamic array (ArrayList) example.",
        "output": "Amortized analysis averages the cost of operations over a sequence. Dynamic array example: appending is usually O(1), but when the array is full, it must resize (copy all elements to a new larger array), costing O(n). However, resizing happens rarely — after n/2 appends. So the amortized cost per append is O(1): total cost for n appends is O(n) (n cheap appends + a few expensive resizes), divided by n operations = O(1) per operation on average."
    },
    {
        "instruction": "Explain space complexity. What is the space complexity of merge sort vs quicksort?",
        "output": "Space complexity measures how much extra memory an algorithm uses relative to input size. Merge sort: O(n) space because it creates temporary arrays during merging. Quicksort: O(log n) space for the recursion stack (average case), O(n) worst case with unbalanced partitions. This is why quicksort is preferred when memory is limited — it uses significantly less auxiliary space than merge sort."
    },
    {
        "instruction": "What does O(n!) factorial time mean? Give an example.",
        "output": "O(n!) factorial time means the running time grows as n factorial. For n=10, that's 3,628,800 operations; for n=20, it's about 2.4×10^18 — completely intractable. Example: generating all permutations of n elements (n! permutations), or the brute-force solution to the traveling salesman problem checking all possible routes. These problems require heuristic or approximation algorithms for any practical input size."
    },
    {
        "instruction": "Explain the Big-O complexity classes from fastest to slowest: O(1), O(log n), O(n), O(n log n), O(n^2), O(2^n), O(n!)",
        "output": "From fastest to slowest: O(1) constant — hash lookup. O(log n) logarithmic — binary search. O(n) linear — simple loop. O(n log n) linearithmic — efficient sorting (merge sort, heap sort). O(n^2) quadratic — nested loops, bubble sort. O(2^n) exponential — recursive Fibonacci, subset enumeration. O(n!) factorial — permutation generation. Rule of thumb: O(n log n) and faster are efficient for large inputs. O(n^2) is acceptable for n < 10,000. O(2^n) and O(n!) are only feasible for very small n (< 30)."
    },
]

for item in complexity_items:
    supplement.append({
        "instruction": item["instruction"],
        "input": "",
        "output": item["output"],
        "source": "complexity_supplement",
        "soul": "cezanne",
        "_cat": "Complexity",
    })

# ===== ANTI-FORGETTING REVIEW (10 items) =====
review_items = [
    {
        "instruction": "Find the bug: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2). Explain why this causes infinite recursion.",
        "output": "Bug: missing base case. Without a termination condition (e.g., if n <= 0: return 0; if n == 1: return 1), the function calls itself forever with decreasing n values, going into negative infinity until Python's recursion limit is hit (RecursionError). Fix: def fibonacci(n): if n <= 0: return 0; if n == 1: return 1; return fibonacci(n-1) + fibonacci(n-2). Better yet, use memoization or iterative approach for O(n) instead of O(2^n)."
    },
    {
        "instruction": "Explain proof by contradiction (reductio ad absurdum) with a mathematical example.",
        "output": "Proof by contradiction works by assuming the opposite of what you want to prove, then showing this leads to a logical contradiction. Example: Prove there is no largest prime number. Assume there IS a largest prime p. Then consider N = (2×3×5×...×p) + 1, the product of all primes plus 1. N is not divisible by any prime up to p (each leaves remainder 1). So either N is prime (contradicting p being largest), or N has a prime factor larger than p (also contradicting p being largest). Contradiction! Therefore, there is no largest prime."
    },
    {
        "instruction": "Explain CPU cache hierarchy (L1, L2, L3) and why it matters for program performance.",
        "output": "CPU cache is small, fast memory between the CPU and main RAM. L1 cache: 32-64KB per core, ~1ns access, fastest. L2 cache: 256KB-1MB per core, ~4ns access. L3 cache: 4-64MB shared across cores, ~10ns access. Main RAM: ~100ns access — 10-100x slower than cache. Performance impact: cache-friendly code (sequential access, small working sets) can be 10-100x faster than cache-unfriendly code (random access, large strides). This is why iterating through a 2D array row-by-row is much faster than column-by-column in C — row access is cache-friendly."
    },
    {
        "instruction": "What is a contrapositive? Show that 'If P then Q' is logically equivalent to 'If not Q then not P'.",
        "output": "The contrapositive of 'If P then Q' is 'If not Q then not P'. They are logically equivalent — both have the same truth table. Example: Original: 'If it rains (P), then the ground is wet (Q).' Contrapositive: 'If the ground is not wet (not Q), then it did not rain (not P).' Both statements are true. This is useful in proofs: instead of proving P→Q directly, prove ¬Q→¬P, which is often easier."
    },
    {
        "instruction": "Write Dijkstra's shortest path algorithm in Python using a priority queue.",
        "output": "import heapq\ndef dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    pq = [(0, start)]\n    while pq:\n        current_dist, current = heapq.heappop(pq)\n        if current_dist > distances[current]:\n            continue\n        for neighbor, weight in graph[current]:\n            distance = current_dist + weight\n            if distance < distances[neighbor]:\n                distances[neighbor] = distance\n                heapq.heappush(pq, (distance, neighbor))\n    return distances\nTime complexity: O((V + E) log V) with binary heap."
    },
    {
        "instruction": "Explain virtual memory, page tables, and TLB in detail.",
        "output": "Virtual memory gives each process an illusion of having a large, contiguous address space. When a process accesses a virtual address, the MMU translates it to a physical address using page tables. Page tables map virtual page numbers to physical frame numbers. Translation Lookaside Buffer (TLB) is a hardware cache that stores recent page table entries, avoiding the need to walk the page table on every access. TLB hit: 1 cycle. TLB miss: requires page table walk (multiple memory accesses). Page fault: the page is not in physical memory, requiring disk I/O — extremely slow (~10ms). This is why TLB hit rate is critical for performance."
    },
    {
        "instruction": "Explain how malloc() works internally. Describe the heap structure and common allocation strategies.",
        "output": "malloc() allocates memory from the heap. Internally, the heap is organized as a list of blocks, each with a header containing size and status (free/allocated). Common strategies: (1) First-fit: scan from start, use first block large enough. (2) Best-fit: find smallest block that fits. (3) Next-fit: continue from last allocation. When free() is called, the block is marked free and adjacent free blocks may be coalesced. Fragmentation is a key problem: external fragmentation (free space is scattered) and internal fragmentation (allocated block is larger than needed). dlmalloc and jemalloc are popular malloc implementations."
    },
    {
        "instruction": "Write quicksort in Python with Lomuto partition scheme.",
        "output": "def quicksort(arr, low=0, high=None):\n    if high is None:\n        high = len(arr) - 1\n    if low < high:\n        pivot_index = partition(arr, low, high)\n        quicksort(arr, low, pivot_index - 1)\n        quicksort(arr, pivot_index + 1, high)\n\ndef partition(arr, low, high):\n    pivot = arr[high]\n    i = low - 1\n    for j in range(low, high):\n        if arr[j] <= pivot:\n            i += 1\n            arr[i], arr[j] = arr[j], arr[i]\n    arr[i + 1], arr[high] = arr[high], arr[i + 1]\n    return i + 1\nTime: O(n log n) average, O(n^2) worst. Space: O(log n) for recursion stack."
    },
    {
        "instruction": "Explain the difference between TCP and UDP protocols. When would you use each?",
        "output": "TCP (Transmission Control Protocol): connection-oriented, reliable, ordered delivery. Uses 3-way handshake, acknowledgments, retransmission, flow control, congestion control. Overhead: higher latency, more bandwidth. Use for: web (HTTP), email (SMTP), file transfer (FTP). UDP (User Datagram Protocol): connectionless, unreliable, no ordering guarantee. No handshake, no retransmission. Overhead: minimal, very fast. Use for: video streaming, online gaming, DNS queries, VoIP. Key tradeoff: TCP guarantees delivery; UDP prioritizes speed over reliability."
    },
    {
        "instruction": "What is SQL injection? Show an example attack and explain prevention methods.",
        "output": "SQL injection occurs when user input is concatenated directly into SQL queries, allowing attackers to execute arbitrary SQL. Example attack: query = \"SELECT * FROM users WHERE name='\" + username + \"'\"; if username = \"' OR '1'='1\", the query becomes: SELECT * FROM users WHERE name='' OR '1'='1' — returns all rows. Prevention: (1) Parameterized queries (prepared statements): cursor.execute(\"SELECT * FROM users WHERE name=?\", (username,)). (2) Input validation and sanitization. (3) ORM frameworks that handle escaping automatically. (4) Least-privilege database accounts. Parameterized queries are the gold standard — they separate code from data."
    },
]

for item in review_items:
    supplement.append({
        "instruction": item["instruction"],
        "input": "",
        "output": item["output"],
        "source": "anti_forget_review",
        "soul": "cezanne",
        "_cat": "Review",
    })

out_path = r"D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_s3_supplement.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(supplement, f, ensure_ascii=False, indent=2)

from collections import Counter
cats = Counter(item["_cat"] for item in supplement)
print(f"Total supplement: {len(supplement)} items")
print(f"Categories: {dict(cats)}")
print(f"Saved: {out_path}")
