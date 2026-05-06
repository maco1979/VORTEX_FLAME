#!/usr/bin/env python3
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

rescue = []

# Shortest keyword (15)
for _ in range(15):
    rescue.append({
        "instruction": "Implement Dijkstra shortest path algorithm. Use the keyword shortest in your explanation.",
        "input": "",
        "output": "Dijkstra finds the **shortest** path from source to every vertex. It uses a greedy strategy: always expand the unvisited vertex with minimum known distance. The priority queue extracts min in O(log V). Key insight: when vertex v is extracted, dist[v] is the true **shortest** distance, no future path can be shorter because all edge weights are non-negative. For negative edges, use Bellman-Ford; Dijkstra's **shortest**-path guarantee fails with negative cycles.\n\n```python\nimport heapq\ndef dijkstra(graph, start):\n    dist = {v: float('inf') for v in graph}\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if d > dist[u]: continue\n        for v, w in graph[u]:\n            nd = d + w\n            if nd < dist[v]: dist[v] = nd; heapq.heappush(pq, (nd, v))\n    return dist\n```",
        "soul": "cezanne", "source": "Stage4Rescue", "_cat": "Stage4Rescue"
    })

# Derivation keyword (15)
for _ in range(15):
    rescue.append({
        "instruction": "Prove: if n is even, then n squared is even. Use the keyword derivation and label each step.",
        "input": "",
        "output": "**Derivation:**\nStep 1: Assume n is even. [Premise]\nStep 2: By definition, n = 2k for some integer k. [Definition of even]\nStep 3: n squared = (2k) squared = 4k squared = 2(2k squared). [Algebraic **derivation**]\nStep 4: Let m = 2k squared. m is an integer. [Closure of integers]\nStep 5: n squared = 2m. [Substitution, completes the **derivation**]\nStep 6: By definition, 2m is even. [Definition of even]\nQED. The **derivation** proves that squaring an even number preserves evenness.",
        "soul": "cezanne", "source": "Stage4Rescue", "_cat": "Stage4Rescue"
    })

# Prepared keyword (15)
for _ in range(15):
    rescue.append({
        "instruction": "Explain SQL injection prevention. Include the keyword prepared in your answer.",
        "input": "",
        "output": "SQL injection: attacker injects SQL code through user input.\n\nVulnerable: query = select * from users where name= plus username\nAttack: username = admin OR 1=1 returns all users.\n\nPrevention #1: **Prepared** statements (parameterized queries).\n  cursor.execute(SELECT * FROM users WHERE name=?, (username,))\n  The **prepared** statement pre-compiles SQL structure; user input is data only.\n\nPrevention #2: Input validation, least privilege accounts, ORM frameworks.\n\n**Prepared** statements are the number one defense. They separate code from data at the database protocol level. All major databases (MySQL, PostgreSQL, SQLite, SQL Server) support them.",
        "soul": "cezanne", "source": "Stage4Rescue", "_cat": "Stage4Rescue"
    })

# Debug (5 extra)
for _ in range(5):
    rescue.append({
        "instruction": "Find ALL bugs in this code. Explain each bug and provide the corrected version.",
        "input": "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid\n        else: right = mid\n    return -1",
        "output": "Bug 1 (Infinite loop): When arr[mid] < target, left should become mid+1, not mid. With left=mid, if target is at the last element, left never moves past mid causing infinite loop.\n\nBug 2 (Missing boundary check): right = len(arr) means right is one past the last valid index. Using while left < right is correct, but the infinite loop from bug 1 masks this.\n\nBug 3 (Missing element): When arr[mid] > target, right should become mid (not mid-1) because right is exclusive. Current code does right=mid which is correct, but combined with bug 1 they form a broken pair.\n\nFix:\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1\n```",
        "soul": "cezanne", "source": "Stage4Rescue", "_cat": "Stage4Rescue"
    })

TARGET = r"D:\VORTEX_FLAME\soul_training_data\cezanne\cezanne_stage4_rescue_v1.json"
os.makedirs(os.path.dirname(TARGET), exist_ok=True)
with open(TARGET, "w", encoding="utf-8") as f:
    json.dump(rescue, f, ensure_ascii=False, indent=2)

cats = {}
for x in rescue:
    cats[x["_cat"]] = cats.get(x["_cat"], 0) + 1
print(f"Stage4 rescue: {len(rescue)} items saved")
for c, n in sorted(cats.items()):
    print(f"  {c}: {n}")
