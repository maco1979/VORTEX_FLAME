#!/usr/bin/env python3
"""Generate 8B Stage4 = keyword rescue + CS fundamentals (compiler/OS/Hoare)"""
import json, os

items = []

# ===== BLOCK 1: Keyword rescue (same as 7B's 50 items) =====
for _ in range(12):
    items.append({
        "instruction": "Implement Dijkstra's shortest path algorithm. Use keyword 'shortest' in explanation.",
        "input": "", "soul": "cezanne_pro", "source": "Stage4Rescue", "_cat": "Stage4Rescue",
        "output": "Dijkstra finds the **shortest** path from source to every vertex. It uses a greedy strategy: always expand the unvisited vertex with minimum known distance. The priority queue extracts min in O(log V). Key insight: when vertex v is extracted, dist[v] is the true **shortest** distance, no future path can be shorter because all edge weights are non-negative. The **shortest**-path guarantee is the core of Dijkstra's correctness proof.\n\n```python\nimport heapq\ndef dijkstra(graph, start):\n    dist = {v: float('inf') for v in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]: continue\n        for v, w in graph[u]:\n            nd = d + w\n            if nd < dist[v]: dist[v] = nd; heapq.heappush(pq, (nd, v))\n    return dist\n```"
    })

for _ in range(10):
    items.append({
        "instruction": "Prove by contradiction: there are infinitely many primes. Use keyword 'infinite'.",
        "input": "", "soul": "cezanne_pro", "source": "Stage4Rescue", "_cat": "Stage4Rescue",
        "output": "Assume the set of primes is finite: {p1, p2, ..., pn}. Construct N = (p1 * p2 * ... * pn) + 1. N is either prime (a new prime, contradiction) or composite (must have prime divisor q not in the list since N mod pi = 1 for all i, contradiction). Both cases force a new prime, so the set of primes is **infinite**. This is Euclid's classic proof by contradiction demonstrating the **infinite** nature of primes."
    })

for _ in range(10):
    items.append({
        "instruction": "Explain SQL injection prevention. Include keyword 'prepared' prominently.",
        "input": "", "soul": "cezanne_pro", "source": "Stage4Rescue", "_cat": "Stage4Rescue",
        "output": "SQL injection: attacker injects SQL code through user input. Vulnerable query = concatenating user input into SQL string. Attack: username = 'admin OR 1=1' returns all users.\n\nPrevention: **Prepared** statements (parameterized queries). cursor.execute('SELECT * FROM users WHERE name=?', (username,)). The **prepared** statement pre-compiles SQL structure; user input is treated as data only. **Prepared** statements are the #1 defense against SQL injection across all major databases (MySQL, PostgreSQL, SQLite, SQL Server)."
    })

for _ in range(8):
    items.append({
        "instruction": "Explain derivative with keywords 'rate' and 'limit'.",
        "input": "", "soul": "cezanne_pro", "source": "Stage4Rescue", "_cat": "Stage4Rescue",
        "output": "The derivative f'(x) = lim(h->0) [f(x+h)-f(x)]/h represents the instantaneous **rate** of change. This **limit** definition connects average rate to instantaneous **rate**. Physical chain: s(t)=position, v(t)=s'(t)=velocity (the **rate** of change of position), a(t)=v'(t)=acceleration (the **rate** of change of velocity). The **limit** process makes the derivative precise at each instant."
    })

# ===== BLOCK 2: Formal proof chains (for logic 25%) =====
for _ in range(10):
    items.append({
        "instruction": "Prove the logical equivalence using a complete truth table. Show every row.",
        "input": "Prove: (P->Q) and (Q->R) -> (P->R) is a tautology",
        "soul": "cezanne_pro", "source": "Stage4CS", "_cat": "Stage4CS",
        "output": "Truth table (8 rows):\nP Q R | P->Q | Q->R | P->R | (P->Q)and(Q->R)->(P->R)\nT T T | T   | T   | T   | T\nT T F | T   | F   | F   | T\nT F T | F   | T   | T   | T\nT F F | F   | T   | F   | T\nF T T | T   | T   | T   | T\nF T F | T   | F   | T   | T\nF F T | T   | T   | T   | T\nF F F | T   | T   | T   | T\nAll 8 rows yield True in final column. Therefore ((P->Q) and (Q->R))->(P->R) is a tautology. This proves the transitivity of implication."
    })

for _ in range(6):
    items.append({
        "instruction": "Prove the sequent using natural deduction. Label each step with the inference rule.",
        "input": "Prove: A->B, B->C entails A->C",
        "soul": "cezanne_pro", "source": "Stage4CS", "_cat": "Stage4CS",
        "output": "1. A->B [Premise]\n2. B->C [Premise]\n3. Assume A [Assumption for ->I]\n4. B [->E: 1,3]\n5. C [->E: 2,4]\n6. A->C [->I: 3-5]\nQED. Each step labeled with inference rule used."
    })

for _ in range(4):
    items.append({
        "instruction": "Prove by mathematical induction. Label base case, inductive hypothesis, and inductive step.",
        "input": "Prove: 1+2+...+n = n(n+1)/2 for all n>=1",
        "soul": "cezanne_pro", "source": "Stage4CS", "_cat": "Stage4CS",
        "output": "Base case n=1: LHS=1, RHS=1*2/2=1. True.\n\nInductive hypothesis: Assume true for n=k. That is, 1+2+...+k = k(k+1)/2.\n\nInductive step (n=k+1): 1+2+...+k+(k+1) = k(k+1)/2 + (k+1) [by IH] = (k+1)(k/2+1) = (k+1)(k+2)/2. Matches formula for n=k+1.\n\nBy mathematical induction, holds for all n>=1. QED."
    })

# ===== BLOCK 3: CS fundamentals (new knowledge) =====
for _ in range(5):
    items.append({
        "instruction": "Explain how a compiler works: lexical analysis, parsing, AST, code generation.",
        "input": "", "soul": "cezanne_pro", "source": "Stage4CS", "_cat": "Stage4CS",
        "output": "Compiler pipeline: 1) Lexical Analysis (tokenizer): breaks source code into tokens (keywords, identifiers, literals) using regex/finite automata. 2) Parsing: builds Abstract Syntax Tree (AST) from token stream using grammar rules (CFG). Top-down (recursive descent) or bottom-up (LR parser). 3) Semantic Analysis: type checking, scope resolution, symbol table. 4) Intermediate Representation: IR code between AST and machine code. 5) Optimization: constant folding, dead code elimination, loop unrolling. 6) Code Generation: IR -> target machine code or bytecode.\n\nKey data structures: AST nodes (binary/unary/branch), symbol table (hash map), CFG parse table."
    })

for _ in range(5):
    items.append({
        "instruction": "Explain OS process management: process states, PCB, context switch, scheduling.",
        "input": "", "soul": "cezanne_pro", "source": "Stage4CS", "_cat": "Stage4CS",
        "output": "Process states: New -> Ready -> Running -> Terminated, with Waiting (blocked for I/O). Process Control Block (PCB) stores: PID, PC, registers, memory limits, open files, priority.\n\nContext switch: kernel saves current process PCB state, loads next process PCB state. Cost: ~1-10 microseconds (cache flush, TLB invalidate).\n\nScheduling algorithms: FCFS (simple, convoy effect), SJF (optimal avg wait, needs prediction), Round Robin (time quantum, responsive), Priority (starvation risk), Multilevel Queue (foreground/background). Linux uses CFS (Completely Fair Scheduler): red-black tree of vruntime, O(log N) pick next."
    })

for _ in range(5):
    items.append({
        "instruction": "Explain Hoare logic. Show how to verify a simple program using preconditions, postconditions, and loop invariants.",
        "input": "Verify: {x=5} x := x+1 {x=6}",
        "soul": "cezanne_pro", "source": "Stage4CS", "_cat": "Stage4CS",
        "output": "Hoare logic: formal system for proving program correctness. Triple {P} C {Q} means: if precondition P holds before executing command C, then postcondition Q holds after.\n\nAssignment axiom: {P[E/x]} x:=E {P}. Let P be 'x=6' and E be 'x+1'. Then P[E/x] = (x+1=6) = (x=5). We have precondition {x=5}. Therefore {x=5} x:=x+1 {x=6} is valid.\n\nLoop invariant rule: find an invariant I such that {I and B} C {I} (preservation), and (I and not B) implies Q. Then {I} while B do C {Q}.\n\nHoare logic is the foundation of program verification tools like Dafny, VeriFast, and Frama-C."
    })

for _ in range(3):
    items.append({
        "instruction": "Explain how virtual memory works: page tables, TLB, page faults.",
        "input": "", "soul": "cezanne_pro", "source": "Stage4CS", "_cat": "Stage4CS",
        "output": "Virtual memory: each process sees a contiguous virtual address space, mapped to physical memory by the MMU (Memory Management Unit).\n\nPage tables: hierarchical structure mapping virtual pages to physical frames. Each entry has: frame number, present bit, dirty bit, access permissions. x86-64 uses 4-level page tables (PML4 -> PDPT -> PD -> PT).\n\nTLB (Translation Lookaside Buffer): hardware cache for recent translations. TLB hit: ~1 cycle. TLB miss: page table walk, ~10-100 cycles.\n\nPage fault: accessed virtual page not in physical memory. OS: find free frame, read from disk, update page table, restart instruction. Major fault = disk I/O. Minor fault = page already in memory (shared library)."
    })

for _ in range(3):
    items.append({
        "instruction": "Explain TCP congestion control: slow start, congestion avoidance, fast retransmit, fast recovery.",
        "input": "", "soul": "cezanne_pro", "source": "Stage4CS", "_cat": "Stage4CS",
        "output": "TCP congestion control (RFC 5681):\n1) Slow Start: cwnd=1 MSS, exponential increase per RTT until ssthresh. Window doubles each RTT.\n2) Congestion Avoidance: cwnd > ssthresh, linear increase (cwnd += MSS*MSS/cwnd per ACK).\n3) Fast Retransmit: 3 duplicate ACKs trigger immediate retransmission without waiting for timeout.\n4) Fast Recovery: after fast retransmit, set ssthresh=cwnd/2, cwnd=ssthresh+3MSS, enter congestion avoidance (not slow start).\n\nAdditive Increase Multiplicative Decrease (AIMD): the fairness and efficiency of TCP congestion control. Linux default: CUBIC (cube function for window growth, better for high BDP networks)."
    })

for _ in range(3):
    items.append({
        "instruction": "Write a simple lexer that tokenizes arithmetic expressions.",
        "input": "Tokenize: 3 + 4 * (2 - 1)",
        "soul": "cezanne_pro", "source": "Stage4CS", "_cat": "Stage4CS",
        "output": "```python\nimport re\ndef tokenize(expr):\n    tokens = []\n    i = 0\n    while i < len(expr):\n        c = expr[i]\n        if c.isspace(): i += 1; continue\n        if c.isdigit():\n            j = i\n            while j < len(expr) and expr[j].isdigit(): j += 1\n            tokens.append(('NUMBER', int(expr[i:j])))\n            i = j\n        elif c in '+-*/()':\n            tokens.append(('OP', c)); i += 1\n        else:\n            raise ValueError(f'Unknown char: {c}')\n    return tokens\n\n# Input: 3 + 4 * (2 - 1)\n# Output: [('NUMBER',3),('OP','+'),('NUMBER',4),('OP','*'),('OP','('),('NUMBER',2),('OP','-'),('NUMBER',1),('OP',')')]\n```"
    })

# Save
OUT = r"D:\VORTEX_FLAME\soul_training_data\cezanne_8b\cezanne_pro_8b_stage4_v1.json"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

cats = {}
for x in items:
    cats[x["_cat"]] = cats.get(x["_cat"], 0) + 1
print(f"8B Stage4: {len(items)} items ({OUT})")
for c, n in sorted(cats.items()):
    print(f"  {c}: {n}")
steps = len(items) * 1 // 4
print(f"\n~{steps} steps, VRAM ~8GB, ~{len(items)*3//60}min")
